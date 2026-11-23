"""Package and run the omnidirectional clock and phased foot pose network."""
import argparse
import math
from pathlib import Path

import torch

from .mlp_omni_clock import FORMAT as CLOCK_FORMAT, load_clock
from .phase_pose import FORMAT as POSE_FORMAT, network, predict
from .phase_x import features
from ..motion import CONTROL_DT

FORMAT = "omni_phase_planner_v1"


class OmniPlanner:
    """Stateful, single-robot planner with cadence updates at left-foot cycle boundaries."""

    def __init__(self, checkpoint, device, *, heading_feedback=False):
        if checkpoint.get("format") != FORMAT:
            raise ValueError("Expected a packaged omni planner checkpoint")
        self.heading_feedback = heading_feedback
        self.clock_weights = checkpoint["clock"]
        self.pose_weights = checkpoint["pose"]
        self.pose_model = network(self.pose_weights["mean"].numel()).to(device).eval()
        self.pose_model.load_state_dict(self.pose_weights["weights"])
        self.pose_weights = {key: self.pose_weights[key].to(device) for key in ("mean", "scale")}
        self.hist = checkpoint["history_length"]
        self.horizons = checkpoint["horizons"]
        self.history = None
        self.clock = None

    def reset(self, position, heading, *, body_yaw=None):
        """Provide the standing feet as local XYZ and sin/cos heading."""
        if position.shape != (1, 2, 3) or heading.shape != (1, 2, 2):
            raise ValueError("Expected initial poses for two feet of one robot")
        if self.heading_feedback and (body_yaw is None or not math.isfinite(body_yaw)):
            raise ValueError("Heading feedback requires a measured initial body yaw")
        self.target_yaw = body_yaw
        self.heading_error = 0.
        self.initial_position = position.clone()
        self.initial_heading = heading.clone()
        self.history = None
        self.clock = None
        self.frame = 0
        self.pending_clock = None
        self.cycle_origin = None
        self.cadence_updates = 0

    @torch.inference_mode()
    def step(self, observation, velocity, query_dt, *, body_yaw=None):
        """Advance one 20 ms control step and return a 13-D contact target."""
        if not hasattr(self, "initial_position"):
            raise RuntimeError("Reset the planner with the initial feet before stepping")
        if observation.ndim != 2 or observation.shape[0] != 1 or observation.shape[1] < 48:
            raise ValueError("Expected one proprioceptive observation with at least 48 fields")
        if len(velocity) != 3 or not all(math.isfinite(v) for v in velocity):
            raise ValueError("Expected a finite (vx, vy, vyaw) command")
        if not any(abs(query_dt-h) < 1e-6 for h in self.horizons):
            raise ValueError("Query time is outside the packaged horizons")
        if self.heading_feedback and (body_yaw is None or not math.isfinite(body_yaw)):
            raise ValueError("Heading feedback requires a measured body yaw at each step")
        command_key = tuple(float(v) for v in velocity)
        if self.clock is None:
            self.clock = load_clock(self.clock_weights,command_key)
            self.command_key = self.active_command = self.initial_command = command_key
            self.query_dt = query_dt
            self.startup_delay = self.clock.delay
        elif command_key != self.command_key:
            if abs(query_dt-self.query_dt)>1e-6:
                raise ValueError("Keep the query horizon fixed when changing speed")
            self.validate_direction_change(self.initial_command,command_key)
            candidate = load_clock(self.clock_weights,command_key)
            self.command_key = command_key
            self.pending_clock = None if command_key == self.active_command else candidate
        if (self.pending_clock is not None or self.cadence_updates) and abs(query_dt-self.query_dt)>1e-6:
            raise ValueError("Keep the query horizon fixed when changing speed")
        obs = observation[:, None, :48]
        self.history = (obs.expand(-1,self.hist,-1).clone() if self.history is None else
                        torch.cat((self.history[:,1:],obs),1))
        phase, code, waiting = self._query_clock(query_dt)
        self.last_timing = phase, code, waiting
        self.frame += 1
        command = observation.new_tensor([[*command_key,query_dt]])
        support = observation.new_tensor([[bool(code&1),bool(code&2)]])
        if waiting:
            position,heading = self.initial_position,self.initial_heading
        else:
            angle = 2*math.pi*phase
            future_phase = observation.new_tensor([math.sin(angle),math.cos(angle)])
            inputs = features(self.history,command,support,future_phase)
            position,heading = predict(self.pose_model,self.pose_weights,inputs)
            if self.heading_feedback:
                # Blend towards the original toe-out and commanded future
                # turn, then add bounded error feedback in the local frame.
                error = self.target_yaw-body_yaw
                self.heading_error = math.atan2(math.sin(error),math.cos(error))
                planned_yaw = torch.atan2(heading[...,0],heading[...,1])
                toe_out = torch.atan2(self.initial_heading[...,0],self.initial_heading[...,1])
                elapsed = (self.frame-1)*CONTROL_DT
                lookahead = min(query_dt,max(0.,elapsed+query_dt-self.startup_delay))
                delta = toe_out+command_key[2]*lookahead-planned_yaw
                nominal = torch.atan2(delta.sin(),delta.cos())
                correction = (.45*nominal+.75*self.heading_error).clamp(-.26,.26)
                yaw = planned_yaw+correction
                heading = torch.stack((yaw.sin(),yaw.cos()),-1)
        if self.heading_feedback and (self.frame-1)*CONTROL_DT >= self.startup_delay:
            target = self.target_yaw+command_key[2]*CONTROL_DT
            self.target_yaw = math.atan2(math.sin(target),math.cos(target))
        mask = support[...,None]
        return torch.cat((command[:,3:],support,(position*mask).flatten(1),(heading*mask).flatten(1)),1)

    @staticmethod
    def validate_direction_change(initial, command):
        axis = max(range(3),key=lambda i:abs(initial[i]))
        if abs(initial[axis]) < 1e-6:
            raise ValueError("Initial command must be nonzero")
        ratio = command[axis]/initial[axis]
        if ratio <= 0 or any(abs(v-ratio*u)>1e-6 for u,v in zip(initial,command)):
            raise ValueError("This version can change speed within the current direction")

    def _query_clock(self, query_dt):
        elapsed = self.frame*CONTROL_DT
        future = elapsed+query_dt
        if self.pending_clock is not None and future >= self.startup_delay:
            if self.cycle_origin is None:
                origin = self.startup_delay-self.clock.offset
            else:
                origin = self.cycle_origin
            boundary = origin+(math.floor((future-origin)/self.clock.period)+1)*self.clock.period
            # Use the first left touchdown beyond the most recent emitted query.
            if self.frame and boundary-self.clock.period > future-CONTROL_DT:
                self.cycle_origin = boundary-self.clock.period
                self.clock = self.pending_clock
                self.active_command = self.command_key
                self.pending_clock = None
                self.cadence_updates += 1
        if self.cycle_origin is None:
            return self.clock.query(elapsed,query_dt)
        phase_time = (future-self.cycle_origin)%self.clock.period
        stage = min(3,sum(phase_time >= self.clock.ends))
        return phase_time/self.clock.period,(3,1,3,2)[stage],False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-model",required=True)
    parser.add_argument("--phase-pose-model",required=True)
    parser.add_argument("--tracker",required=True,help="Tracker checkpoint supplying history length and query horizons")
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    load = lambda path: torch.load(path,map_location="cpu",weights_only=True)
    clock,pose,tracker = load(args.clock_model),load(args.phase_pose_model),load(args.tracker)
    config = tracker["config"]
    hist = config["hist"]
    if clock.get("format") != CLOCK_FORMAT or pose.get("format") != POSE_FORMAT:
        parser.error("Expected an omni MLP clock and a phase pose model")
    if pose["mean"].shape != (48*hist+8,) or pose["scale"].shape != pose["mean"].shape:
        parser.error("Pose model dimensions must match tracker history length")
    bundle = dict(format=FORMAT,clock=clock,pose=pose,history_length=hist,
                  horizons=config["horizons"],robot_fingerprint=tracker["robot"]["fingerprint"])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("xb") as stream:
        torch.save(bundle,stream)
    print(args.out)


if __name__=="__main__":
    main()
