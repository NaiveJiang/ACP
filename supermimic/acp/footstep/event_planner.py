"""Supervised next-contact planner for two feet on flat ground."""
import torch
from torch import nn
from torch.nn import functional as F

from ..motion import CONTROL_DT
from ..task import to_base
from ...rotations import yaw_mat

FORMAT = "two_foot_contact_event_planner_v1"
WINDOW = 75
MIN_SUPPORT = 10


def _next(mask):
    index = torch.arange(len(mask), device=mask.device)[:, None, None]
    marked = torch.where(mask, index, len(mask)).flip(0).cummin(0).values.flip(0)
    return torch.cat((marked[1:], torch.full_like(marked[:1],len(mask))))


def hindsight(positions, angles, roots, rotations, contact, velocity, history, accepted):
    """Label later observed support events, with no censored positive or negative windows."""
    # A one-frame drop inside a stance is usually contact-point handoff.
    middle = (contact[:-2].int()+contact[1:-1].int()+contact[2:].int()) >= 2
    contact = torch.cat((contact[:1],middle,contact[-1:]))
    t = torch.arange(len(contact),device=contact.device)[:,None,None]
    release = _next(~contact)
    starts = contact & ~torch.cat((torch.zeros_like(contact[:1]),contact[:-1]))
    touchdown = _next(starts & (release-t >= MIN_SUPPORT))
    n = len(contact) - WINDOW - MIN_SUPPORT + 1
    if n <= 0:
        raise ValueError("The rollout is shorter than the touchdown window")
    observed = touchdown[:n] <= t[:n] + WINDOW
    end = release.gather(0,touchdown[:n].clamp_max(len(contact)-1))
    lookup = touchdown[:n].clamp_max(len(contact)-1)
    pos = positions.gather(0,lookup[...,None].expand(-1,-1,-1,3))
    heading = angles.gather(0,lookup)
    pos = to_base(pos.flatten(0,1),roots[:n].flatten(0,1),rotations[:n].flatten(0,1)).unflatten(0,(n,-1))
    yaw = torch.atan2(rotations[:n,...,1,0],rotations[:n,...,0,0])
    heading = heading - yaw[...,None]
    delta = (velocity[1:]-velocity[:-1]).abs().amax(-1) > 1e-5
    changes = torch.cat((torch.zeros_like(delta[:1],dtype=torch.long),delta.long().cumsum(0)))
    clean = (changes[:n] == changes[WINDOW:WINDOW+n]) & accepted[None]
    end_command = changes[...,None].expand(-1,-1,2).gather(0,end.clamp_max(len(contact)-1))
    return {
        "history":history[:n][clean],
        "velocity":velocity[:n][clean],
        "support":contact[:n][clean].float(),
        "valid":observed[clean].float(),
        "release":((release[:n]-t[:n]).clamp(0,WINDOW+1)*CONTROL_DT)[clean],
        "touchdown":((touchdown[:n]-t[:n]).clamp(0,WINDOW+1)*CONTROL_DT)[clean],
        "duration":((end-touchdown[:n]).clamp(0,WINDOW)*CONTROL_DT)[clean],
        # Never supervise a truncated stance duration near the rollout boundary.
        "duration_known":((end < len(contact)) & (end_command == changes[:n,...,None]))[clean].float(),
        "position":pos[clean],
        "heading":torch.stack((heading.sin(),heading.cos()),dim=-1)[clean],
    }


class EventPlanner(nn.Module):
    def __init__(self, hist):
        super().__init__()
        self.hist = hist
        self.net = nn.Sequential(nn.Linear(48*hist+3,256),nn.LayerNorm(256),nn.SiLU(),
                                 nn.Linear(256,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,20))

    def forward(self, history, velocity):
        x = history[...,:48]
        # The fixed scales leave observed positions in radians and rates in rad/s.
        x = torch.cat((x[...,:3],x[...,3:6]/3.,x[...,6:27],x[...,27:48]/6.),dim=-1)
        out = self.net(torch.cat((x.flatten(1),velocity),dim=-1)).reshape(-1,2,10)
        times = out[...,2:5].sigmoid()*((WINDOW+2)*CONTROL_DT)
        return out[...,:2], times, out[...,5:8], F.normalize(out[...,8:10],dim=-1,eps=1e-6)

    def loss(self, batch):
        logits,times,pos,yaw = self(batch["history"],batch["velocity"])
        release,touch,duration = times.unbind(-1)
        valid = batch["valid"]
        support = batch["support"]
        def mean(value, mask):
            return (value*mask).sum()/mask.sum().clamp_min(1)
        def timing(value, target, mask):
            return mean(F.smooth_l1_loss(value/.20,target/.20,reduction="none"),mask)
        loss = F.binary_cross_entropy_with_logits(logits,torch.stack((support,valid),dim=-1))
        loss = loss + timing(release,batch["release"],support)
        loss = loss + timing(touch,batch["touchdown"],valid)
        loss = loss + timing(duration,batch["duration"],valid*batch["duration_known"])
        position_loss = F.smooth_l1_loss(pos/.10,batch["position"]/.10,reduction="none").mean(-1)
        return loss + mean(position_loss + (1-(yaw*batch["heading"]).sum(-1)),valid)

    @torch.no_grad()
    def plan(self, history, velocity, query_dt, state, task):
        """Adapt the next event per foot to the existing tracker query interface."""
        logits,times,next_pos,next_heading = self(history,velocity)
        release,touch,duration = times.unbind(-1)
        current = (logits[...,0] >= 0) & (query_dt < release)
        arriving = (logits[...,1] >= 0) & (query_dt >= touch) & (query_dt < touch+duration)
        # The walking controller expects at least one supported foot.
        support = current | arriving
        fallback = F.one_hot(logits[...,0].argmax(-1),2).bool()
        support = support | (~support.any(-1,keepdim=True) & fallback)
        feet,R,_ = task.foot_geometry(state)
        current_pos = to_base(feet,state[0],state[1])
        angle = torch.atan2(R[...,1,0],R[...,0,0]) - torch.atan2(state[1][:,1,0],state[1][:,0,0])[:,None]
        current_yaw = torch.stack((angle.sin(),angle.cos()),-1)
        choose_event = (arriving & ~current)[...,None]
        position = torch.where(choose_event,next_pos,current_pos)
        heading = torch.where(choose_event,next_heading,current_yaw)
        return torch.cat((query_dt,support,(position*support[...,None]).flatten(1),
                          (heading*support[...,None]).flatten(1)),dim=-1)


def handover(iteration, start, duration):
    return min(1.,max(0.,(iteration-start)/duration))


@torch.no_grad()
def world_plan(plan, state):
    basis = yaw_mat(state[1])
    position = state[0][:,None] + (basis[:,None] @ plan[:,3:9].reshape(-1,2,3,1)).squeeze(-1)
    yaw = torch.atan2(basis[:,1,0],basis[:,0,0])[:,None]
    heading = yaw + torch.atan2(plan[:,9::2],plan[:,10::2])
    return plan[:,1:3],position,heading


def contact_reward_delta(task, state, frames, support, target, heading):
    """Difference between planned and reference contact rewards at the deadline."""
    c = task.cfg
    feet,R,gap = task.foot_geometry(state)
    probability = torch.sigmoid((c.contact_distance-gap)/c.contact_temperature)
    contact = (support*probability+(1-support)*(1-probability)).mean(-1)
    count = support.sum(-1).clamp_min(1.)
    position = (torch.exp(-(feet-target).square().sum(-1)/c.position_std**2)*support).sum(-1)/count
    yaw = torch.atan2(R[...,1,0],R[...,0,0])
    error = torch.atan2((yaw-heading).sin(),(yaw-heading).cos())
    angle = (torch.exp(-error.square()/c.yaw_std**2)*support).sum(-1)/count
    reference = task.reward_terms(state,frames)
    return CONTROL_DT*(c.contact_weight*(contact-reference["contact"])
                       +c.position_weight*(position-reference["position"])
                       +c.yaw_weight*(angle-reference["yaw"]))


class EventHandover:
    def __init__(self, planner, task):
        self.planner, self.task = planner,task
        motion = task.motion
        changes = (motion.velocity_command[1:]-motion.velocity_command[:-1]).abs().amax(-1)>1e-5
        changes |= motion.clip_id[1:]!=motion.clip_id[:-1]
        self.command_ids = F.pad(changes.long().cumsum(0),(1,0))

    @torch.no_grad()
    def plan(self, history, state, frame, query):
        m = self.task.motion
        return self.planner.plan(history,m.velocity_command[frame],m.horizons[query,None],state,self.task)

    def imagined_delta(self, predicted, frames, query, plans, use):
        """Match the plan sent on a rollout step to the state at its deadline."""
        steps = self.task.motion.horizon_steps[query]
        horizon = len(plans)
        elapsed = torch.arange(1,horizon+1,device=query.device)[:,None]
        origin = (elapsed-steps).clamp_min(0)
        columns = torch.arange(len(query),device=query.device)
        support,position,heading = (torch.stack(series)[origin,columns].flatten(0,1)
                                    for series in zip(*plans))
        f = torch.stack(frames)
        same = self.command_ids[f[:-1][origin,columns]] == self.command_ids[f[1:]]
        valid = ((elapsed>=steps)&use&same).flatten()
        return contact_reward_delta(self.task,predicted,f[1:].flatten(),support,position,heading)*valid


class DuePlans:
    """Small real-rollout queue keyed by environment and policy query deadline."""
    def __init__(self, event, envs, device):
        self.event = event
        self.size = int(event.task.motion.horizon_steps.max())+1
        self.slots = torch.zeros((self.size,envs,11),device=device)
        self.valid = torch.zeros((self.size,envs),device=device,dtype=torch.bool)
        self.ids = torch.arange(envs,device=device)
        self.t = 0

    @torch.no_grad()
    def append(self, plan, state, frames, eligible):
        support,pos,yaw = world_plan(plan,state)
        self.slots[self.t] = torch.cat((support,pos.flatten(1),yaw,self.event.command_ids[frames,None]),dim=1)
        self.valid[self.t] = eligible
        self.t = (self.t+1)%self.size

    def reward(self, state, frames, query, episode_step, weight):
        steps = self.event.task.motion.horizon_steps[query]
        origin = (self.t-steps)%self.size
        x = self.slots[origin,self.ids]
        same = x[:,10] == self.event.command_ids[frames]
        live = self.valid[origin,self.ids] & (episode_step+1>=steps) & same
        delta = contact_reward_delta(self.event.task,state,frames,x[:,:2],x[:,2:8].reshape(-1,2,3),x[:,8:10])
        return weight*delta*live
