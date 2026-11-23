"""S17 clip selection, mirroring and simulation-root playback for the RoBan player."""
import math
import re

import mujoco
import numpy as np
from scipy.signal import savgol_filter

from .motion_matching import Database, LOOKAHEAD, heading, planar_heading, rotate


def angle_delta(target, current):
    delta = target-current
    return np.arctan2(np.sin(delta), np.cos(delta))


def spring_angle(angle, velocity, goal, rate, dt):
    error = angle_delta(angle, goal)
    slope = velocity+rate*error
    decay = math.exp(-rate*dt)
    return goal+(error+slope*dt)*decay, (velocity-rate*slope*dt)*decay


class S17Stick:
    """Drive and steer independently, including fixed travel during world yaw."""

    def __init__(self, limits, run_speed, deadzone, dt):
        self.limits, self.run_speed, self.deadzone, self.dt = limits, run_speed, deadzone, dt
        self.rate = 2*math.log(2.)/.27
        self.body_goal = self.travel_goal = self.travel_yaw = self.travel_rate = 0.
        self.body_turn = self.forward = self.side = 0.
        self.walk = self.walk_rate = 0.
        self.world_turn_held = False

    def update(self, axes, walk, orbit, facing, root):
        drive, strafe, yaw, world_yaw = np.where(
            np.abs(axes) > self.deadzone, np.asarray(axes)*np.abs(axes), 0.)
        self.walk, self.walk_rate = spring_angle(self.walk, self.walk_rate, float(walk),
                                                  2*math.log(2.)/.1, self.dt)
        walk_speed, side_speed, max_turn = self.limits
        forward_speed = self.run_speed+(walk_speed-self.run_speed)*self.walk
        side_speed *= 1.-(1.-.75/1.15)*self.walk
        scale = max(1., abs(drive)+abs(strafe))
        self.forward = forward_speed*drive/scale
        self.side = side_speed*strafe/scale

        if self.world_turn_held and world_yaw == 0.:
            self.travel_goal = self.body_goal
            self.travel_yaw, self.travel_rate = root.yaw, root.yaw_rate
            self.world_turn_held = False
        self.body_turn = 0. if orbit else max_turn*yaw
        turn = self.body_turn if orbit else np.clip(self.body_turn+max_turn*world_yaw, -max_turn, max_turn)
        if not orbit and world_yaw != 0.:
            self.world_turn_held = True

        velocity = self.world_velocity(facing, 0.)
        self.body_goal += turn*self.dt
        self.travel_goal += self.body_turn*self.dt
        self.travel_yaw, self.travel_rate = spring_angle(
            self.travel_yaw, self.travel_rate, self.travel_goal, self.rate, self.dt)
        return np.r_[rotate(velocity, -facing), turn], self.body_goal

    def world_velocity(self, facing, lookahead):
        travel = self.travel_yaw
        if lookahead:
            # Solve the heading spring with a travel goal that advances at body_turn.
            goal = self.travel_goal-2*self.body_turn/self.rate
            travel, _ = spring_angle(travel, self.travel_rate-self.body_turn, goal, self.rate, lookahead)
            travel += self.body_turn*lookahead
        return rotate([self.forward, 0.], travel)+rotate([0., self.side], facing)

    def orbit(self, axes, azimuth, elevation):
        right = np.asarray(axes[2:])
        strength = np.linalg.norm(right)
        if strength <= self.deadzone:
            return azimuth, elevation
        left, up = right*(min(1., strength)**2/strength)
        rate = math.degrees(2.*self.dt)
        return (azimuth+left*rate+180.) % 360.-180., float(np.clip(elevation+up*rate, -72., 0.))


class Mirror:
    def __init__(self, robot):
        model = robot.model
        self.root = robot.root
        self.columns = robot.joint_qpos
        self.order = []
        self.sign = []
        for name in robot.joints:
            other = re.sub(r"_([lr])(?=\d)", lambda m: "_r" if m[1] == "l" else "_l", name)
            self.order.append(robot.joints.index(other))
            axis = model.jnt_axis[model.joint(name).id]
            opposite = model.jnt_axis[model.joint(other).id]
            parity = np.dot(axis, opposite*[-1, 1, -1])
            if abs(parity) < .999:
                raise ValueError(f"Joint axes do not mirror for {name}")
            self.sign.append(np.sign(parity))
        self.sign = np.array(self.sign)

    def poses(self, poses):
        result = poses.copy()
        result[..., [self.root+i for i in (1, 4, 6)]] *= -1
        result[..., self.columns] = poses[..., self.columns[self.order]]*self.sign
        return result

    @staticmethod
    def motion(features):
        result = features.copy()
        for start in (0, 6):
            section = features[..., start:start+6]
            result[..., start:start+6] = (section.reshape(*section.shape[:-1], 2, 3)[..., ::-1, :]
                                               * [1., -1., 1.]).reshape(section.shape)
        result[..., [13, 16, 18, 20, 22, 24, 26]] *= -1
        return result

    def posture(self, pose):
        result = pose.copy()
        result[..., :21] = pose[..., self.order]*self.sign
        result[..., 21:] = pose[..., 21:][..., ::-1]
        return result


def selected_sections(db, directory, selection):
    """Intersect the S17 selection and error masks with the existing checkpoint's source frames."""
    take = {name: i for i, name in enumerate(db.names)}
    use = {}
    for line in selection.read_text().splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        name, start, stop, gait, *_ = fields
        if gait != "idle" and name in take:
            use.setdefault(name, []).append((int(start), int(stop), gait))
    mask = np.zeros(len(db.pose), dtype=bool)
    ends = db.end.copy()
    labels = []
    for name, ranges in use.items():
        rows = np.flatnonzero(db.clip == take[name])
        path = directory/(name+".csv")
        with path.open(encoding="utf-8-sig") as file:
            header = file.readline().strip().split(",")
        error_fields = ("LeftFoot_pos_err", "LeftToe_pos_err", "RightFoot_pos_err", "RightToe_pos_err")
        columns = [header.index("Frame")] + [header.index(field) for field in error_fields if field in header]
        quality = np.loadtxt(path, delimiter=",", skiprows=1, usecols=columns, ndmin=2)
        if len(quality) != len(rows):
            raise ValueError(f"S17 selection and cached track differ for {name}")
        frame = quality[:, 0]
        good = db.end[rows] > rows
        if len(columns) > 1:
            good &= np.max(quality[:, 1:], axis=1) <= .10
        chosen = np.zeros(len(rows), dtype=bool)
        for start, stop, _ in ranges:
            chosen |= (frame >= start) & (frame < stop)
        sections = np.flatnonzero(np.diff(np.r_[False, good & chosen, False])).reshape(-1, 2)
        for start, stop in sections:
            if stop-start <= round(db.fps*1.33):
                continue
            mask[rows[start:stop]] = True
            ends[rows[start:stop]] = min(rows[stop-1], int(db.end[rows[start]]))
            labels.append((rows[start], rows[stop-1]+1))
    return mask, ends, labels


def _smoothed(values, seconds, fps):
    width = 2*round(seconds*fps/2)+1
    return savgol_filter(values, width, 3, axis=0, mode="interp") if len(values) >= width else values


class AlignedDatabase(Database):
    def __init__(self, base, robot, directory, selection):
        self.__dict__.update(base.__dict__)
        self.base = base
        self.n = len(base.pose)
        self.mirror = Mirror(robot)
        mask, ends, sections = selected_sections(base, directory, selection)
        horizon = round(base.fps)
        usable = mask & (np.arange(self.n)+horizon <= ends)
        chosen = np.flatnonzero(usable & np.isin(np.arange(self.n), base.candidates))
        if not len(chosen):
            raise ValueError("No S17 selections overlap the current motion database")
        self.sections = len(sections)
        self.end = np.r_[ends, ends+self.n]
        self.names = base.names+[name+" mirrored" for name in base.names]
        self.clip = np.r_[base.clip, base.clip+len(base.names)]
        self.running_clips = np.array([name.startswith("run") for name in self.names])
        self.pose = np.concatenate((base.pose, self.mirror.poses(base.pose)))
        self.stance = np.concatenate((base.stance, base.stance[:, ::-1]))
        self.pose_features = np.concatenate((base.pose_features, self.mirror.posture(base.pose_features)))
        self.intent = np.concatenate((base.intent, base.intent*[1., -1., -1.])).astype(np.float32)
        feats = base.features.copy()
        idx = robot.root
        offsets = np.rint(LOOKAHEAD*base.fps).astype(int)
        for start, end in sections:
            raw = base.pose[start:end, idx:idx+7]
            yaw = heading(raw[:, 3:7])
            root = _smoothed(raw[:, :2], .50, base.fps)
            directions = _smoothed(planar_heading(yaw), 1., base.fps)
            sim_yaw = np.arctan2(directions[:, 1], directions[:, 0])
            row = np.arange(len(raw))
            future = np.minimum(row[:, None]+offsets, len(raw)-1)
            original = base.features[start:end]
            relative = original[:, :6].reshape(-1, 2, 3)
            foot_xy = rotate(relative[..., :2], yaw[:, None]) + raw[:, None, :2]
            foot_xy = rotate(foot_xy-root[:, None], -sim_yaw[:, None])
            feats[start:end, :6] = np.dstack((foot_xy, relative[..., 2])).reshape(-1, 6)
            feet_vel = rotate(original[:, 6:12].reshape(-1, 2, 3)[..., :2], (yaw-sim_yaw)[:, None])
            feats[start:end, 6:12] = np.dstack((feet_vel, original[:, [8, 11]])).reshape(-1, 6)
            feats[start:end, 12:14] = rotate(np.gradient(root, axis=0)*base.fps, -sim_yaw)
            feats[start:end, 15:21] = rotate(root[future]-root[:, None], -sim_yaw[:, None]).reshape(-1, 6)
            feats[start:end, 21:27] = planar_heading(sim_yaw[future]-sim_yaw[:, None]).reshape(-1, 6)
        self.features = np.concatenate((feats, self.mirror.motion(feats)))
        # A known neutral seed remains available even if it sits outside a curated moving range.
        self.end[self.start] = base.end[self.start]
        self.candidates = np.r_[chosen, chosen+self.n]
        self.candidate_intent = self.intent[self.candidates]
        self.candidate_pose = self.pose_features[self.candidates]
        self.normalized = (self.features[self.candidates]*self.scale).astype(np.float32)
        self.curated_columns = np.flatnonzero(np.isin(base.candidates, chosen))
        self.projector = None

        future = np.minimum(np.arange(self.n)[:, None]+np.arange(horizon+1), ends[:, None])
        contact = base.stance[future] >= .40
        # Require three consecutive swing and support frames around a landing.
        air = ~(contact[:, :-5] | contact[:, 1:-4] | contact[:, 2:-3])
        ground = contact[:, 3:-2] & contact[:, 4:-1] & contact[:, 5:]
        alternating = (air & ground).any(axis=1).all(axis=-1)
        self.alternating = np.tile(alternating, 2)
        self.displacements = self.features[:, 15:21].reshape(-1, 3, 2)
        self.facing = self.features[:, 21:27].reshape(-1, 3, 2)
        self.filtered_queries = 0
        self.shortlist_fallbacks = 0

    def eligibility(self, command, current):
        speed = np.linalg.norm(command[:2])
        # For a sustained translation, exclude pivots, braking, stops and foot-shuffle windows.
        if speed < .23 or abs(command[2]) >= .18:
            return None, True
        direction = command[:2]/speed
        ids = np.r_[self.candidates, current]
        future = self.displacements[ids]
        forward = future @ direction
        side = future @ np.array([-direction[1], direction[0]])
        increments = np.diff(np.column_stack((np.zeros(len(ids)), forward)), axis=1)
        straight = ((forward[:, -1] >= .38*speed)
                    & (increments.min(axis=1) > -.015)
                    & (np.abs(side).max(axis=1) <= .35*forward[:, -1]+.07)
                    & (self.facing[ids, :, 0].min(axis=1) >= np.cos(.40)))
        stepped = straight & self.alternating[ids]
        if speed >= 1.85:
            running = stepped & self.running_clips[self.clip[ids]] & (forward[:, -1] >= .62*speed)
            if running[:-1].any():
                self.filtered_queries += 1
                return running[:-1], bool(running[-1])
        if stepped[:-1].any():
            self.filtered_queries += 1
            return stepped[:-1], bool(stepped[-1])
        if straight[:-1].any():
            self.filtered_queries += 1
            return straight[:-1], bool(straight[-1])
        return None, True

    def pick(self, query, command, pose_query, current, force, command_changed, current_cost, intent_scale,
             pool, allowed, keep_current):
        result = super().pick(query, command, pose_query, current, force, command_changed, current_cost,
                              intent_scale, pool, allowed, keep_current)
        if result is None and not isinstance(pool, slice):
            self.shortlist_fallbacks += 1
        return result

    def shortlist(self, query, command, pose_query, mode):
        count = len(self.curated_columns)
        direct = self.projector.propose(self.base.projection_inputs(query, command, pose_query), mode)
        reflected = self.base.projection_inputs(self.mirror.motion(query),
                                                command*np.array([1., -1., -1.]),
                                                self.mirror.posture(pose_query))
        mirror = self.projector.propose(reflected, mode)
        return np.r_[direct, mirror+count]


class MirroredStepper:
    def __init__(self, original, db):
        self.original, self.db = original, db
        self.window = original.window
        self.decoded_root = original.decoded_root

    def seed(self, frame):
        self.side = self.db.n if frame >= self.db.n else 0
        self.original.seed(frame-self.side)
        self.last = min(self.original.last+self.side, self.db.end[frame]-1)

    def pose(self, frame):
        pose = self.original.pose(frame-self.side)
        return self.db.mirror.poses(pose) if self.side else pose


class SimulationRoot:
    """Spring trajectory prediction and bounded root correction in the S17 convention."""

    def __init__(self, robot, dt, *, velocity_source=None):
        self.robot, self.dt = robot, dt
        self.velocity_source = velocity_source
        self.rate = 2*math.log(2.)/.27
        self.reset()

    def reset(self):
        self.xy = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.yaw = self.goal = self.yaw_rate = 0.

    def _advance(self, xy, v, a, yaw, rate, goal, command, dt, lookahead=0.):
        r = self.rate
        next_yaw, next_rate = spring_angle(yaw, rate, goal, r, dt)
        decay = math.exp(-r*dt)
        desired = (rotate(command[:2], next_yaw) if self.velocity_source is None
                   else self.velocity_source(next_yaw, lookahead))
        j0 = v-desired
        j1 = a+j0*r
        displacement = (1-decay)*(j1/r**2+j0/r)-decay*j1*dt/r+desired*dt
        return xy+displacement, (j0+j1*dt)*decay+desired, (a-j1*r*dt)*decay, next_yaw, next_rate

    def target_heading(self, target, rate):
        delta = angle_delta(target, self.yaw)
        if abs(delta) > np.pi-.10 and abs(rate) > .1:
            delta = math.copysign(min(abs(delta), np.pi-1e-6), rate)
        return self.yaw+delta

    def query(self, command, xy, yaw, target):
        goal = self.goal if target is None else self.target_heading(target, command[2])
        state = (self.xy, self.velocity, self.acceleration, self.yaw, self.yaw_rate)
        positions, angles = [], []
        lookahead = 0.
        for dt in np.diff(np.r_[0., LOOKAHEAD]):
            lookahead += dt
            destination = goal if target is not None and self.velocity_source is None else goal+command[2]*dt
            state = self._advance(*state, destination, command, dt, lookahead)
            goal = destination
            positions.append(state[0])
            angles.append(state[3])
        return rotate(np.asarray(positions)-xy, -yaw), planar_heading(np.asarray(angles)-yaw)

    def correct(self, player, pose, command, target):
        absolute = target is not None
        self.goal = (self.target_heading(target, command[2]) if absolute else self.goal+command[2]*self.dt)
        state = self._advance(self.xy, self.velocity, self.acceleration, self.yaw, self.yaw_rate,
                              self.goal, command, self.dt)
        self.xy, self.velocity, self.acceleration, self.yaw, self.yaw_rate = state
        root = pose[self.robot.root:self.robot.root+7]
        old_xy = root[:2].copy()
        delta = self.xy-old_xy
        damped = delta*(1-math.exp(-math.log(2.)*self.dt/.1))
        max_step = .5*np.linalg.norm(player.velocity[:2])*self.dt
        root[:2] += damped*min(1., max_step/max(np.linalg.norm(damped), 1e-8))
        error = root[:2]-self.xy
        root[:2] = self.xy+error*min(1., .15/max(np.linalg.norm(error), 1e-8))
        offset = angle_delta(self.yaw, heading(root[3:7]))
        if not absolute:
            cap = .5*abs(player.velocity[5])*self.dt
            small = np.clip(offset*(1-math.exp(-math.log(2.)*self.dt/.2)), -cap, cap)
            offset = small + (offset-small)-np.clip(offset-small, -np.pi/2, np.pi/2)
        # Carry corrected SE2 into the alignment so the next frame starts in the same place.
        player.shift = root[:2]+rotate(player.shift-old_xy, offset)
        player.yaw += offset
        turn = np.array([np.cos(offset/2), 0., 0., np.sin(offset/2)])
        mujoco.mju_mulQuat(root[3:7], turn, root[3:7].copy())


class SupportAwareSimulationRoot(SimulationRoot):
    """Reduce excess planar foot motion from blending and root correction."""

    def correct(self, player, pose, command, target):
        super().correct(player, pose, command, target)
        db = player.db
        rows = slice(player.frame-1, player.frame+1)
        # Use support in both consecutive reference frames to preserve swing and toe-off.
        support = db.stance[rows].min(axis=0)
        weight = np.clip((support-.18)/.52, 0., 1.)
        if not weight.any():
            return

        robot = self.robot
        robot.data.qpos[:] = pose
        mujoco.mj_kinematics(robot.model, robot.data)
        feet = robot.foot_positions()
        height = np.maximum(player.feet[:, 2], feet[:, 2])
        weight *= np.clip((.09-height)/.06, 0., 1.)

        roots = db.pose[rows, robot.root:robot.root+7]
        local_feet = db.features[rows, :6].reshape(2, 2, 3)[..., :2]
        source_feet = roots[:, None, :2] + rotate(local_feet, heading(roots[:, 3:7])[:, None])
        # Preserve the source foot's own motion through stance, including rolling the sole.
        source_delta = rotate(source_feet[1]-source_feet[0], player.yaw)
        error = player.feet[:, :2]+source_delta-feet[:, :2]
        shift = .8*(error*weight[:, None]).sum(0)/max(1., weight.sum())
        shift *= min(1., .85*self.dt/max(np.linalg.norm(shift), 1e-8))
        pose[robot.root:robot.root+2] += shift
        player.shift += shift
