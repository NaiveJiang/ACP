"""Static masked updates and replayable GPU optimizer steps.

CUDA-graph/finite-step strategy follows BlinkMimic; inactive episode suffixes
are sanitized BEFORE model/loss evaluation, retaining ACP's quarantine semantics.
"""
import copy

import torch

from ...mdp import normalize_xi, unflatten_state
from ...networks import WorldModel
from .models import flat_scan


def freeze(module, flag):
    module.requires_grad_(not flag)


def gated_step(loss, model, optimizer, eligible=None):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    good = torch.isfinite(loss) & torch.isfinite(grad)
    if eligible is not None:
        good = good & eligible
    for p in model.parameters():
        if p.grad is not None:
            p.grad.nan_to_num_(nan=0., posinf=0., neginf=0.)
            p.grad.mul_(good)
    if isinstance(optimizer, torch.optim.Optimizer):
        # Fused Adam skips parameters, moments AND step counter on bad gradients.
        optimizer.found_inf = (~good).float()
        try:
            optimizer.step()
        finally:
            optimizer.found_inf = None
    else:
        optimizer.step(ok=good)  # vendored capturable Muon + fused Adam
    return good, grad


class Updates:
    def __init__(self, wm, policy, critic, mdp, task, opt_wm, opt_pi, opt_v,
                 horizon=16, gamma=.99, lam=.95, critic_epochs=2, compile=True, ema=0.,
                 critic_minibatches=1, event=None):
        self.wm, self.policy, self.critic, self.mdp, self.task = wm, policy, critic, mdp, task
        self.opt_wm, self.opt_pi, self.opt_v = opt_wm, opt_pi, opt_v
        self.horizon, self.gamma, self.lam, self.critic_epochs = horizon, gamma, lam, critic_epochs
        self.critic_minibatches = critic_minibatches
        self.event = event
        self.ema = ema
        self.average = copy.deepcopy(policy).requires_grad_(False) if ema else None
        device = next(wm.parameters()).device
        self.errors = torch.zeros(3,device=device,dtype=torch.long)
        self.runs = torch.zeros(3,device=device,dtype=torch.long)
        self.worst = torch.zeros(3,device=device,dtype=torch.long)
        self.fallback_position = torch.tensor([0.,0.,.77],device=device)
        J = mdp.default_q.numel()
        self.error_floor = torch.cat((torch.full((3,),.003),torch.full((3,),.005),torch.full((3,),.02),
                                      torch.full((3,),.05),torch.full((J,),.003),torch.full((J,),.05))).to(device)
        self.wm_loss = self._wm_loss
        self.imag_step = self._imag_step
        self.reward = self._reward
        self.critic_input = mdp.critic_input
        self.value_targets = self._value_targets
        self.upper_supervision = task.supervision_loss if hasattr(task, 'supervision_loss') else None
        self.joint_supervision = task.joint_supervision_loss if hasattr(task,'joint_supervision_loss') else None
        self.trajectory_supervision=getattr(task,'trajectory_supervision_loss',None)
        if compile and self.trajectory_supervision is not None:
            self.trajectory_supervision=torch.compile(self.trajectory_supervision,fullgraph=True,dynamic=False,
                                                     options={'max_fusion_size':32})
        self.query_reward = hasattr(mdp,'reward_with_query')
        if compile and self.joint_supervision is not None:
            self.joint_supervision=torch.compile(self.joint_supervision,fullgraph=True,dynamic=False,
                                                options={'max_fusion_size':32})
        if compile and self.upper_supervision is not None:
            self.upper_supervision = torch.compile(self.upper_supervision, fullgraph=True, dynamic=False,
                                                    options={'max_fusion_size':32})
        if compile:
            options = {'max_fusion_size':32}
            for attr in ('wm_loss','imag_step','reward','critic_input','value_targets'):
                setattr(self,attr,torch.compile(getattr(self,attr),fullgraph=True,dynamic=False,options=options))

    def note(self, index, good):
        self.errors[index].add_((~good).long())
        self.runs[index].copy_(torch.where(good,0,self.runs[index]+1))
        self.worst[index].copy_(torch.maximum(self.worst[index],self.runs[index]))

    def fallback(self, S):
        n = S[0].shape[0]
        p = torch.zeros_like(S[0])+self.fallback_position
        return (p,torch.eye(3,device=p.device).expand(n,-1,-1),torch.zeros_like(S[2]),torch.zeros_like(S[3]),
                self.mdp.default_q.expand(n,-1),torch.zeros_like(S[5]))

    @staticmethod
    def choose(live, yes, no):
        return tuple(torch.where(live.reshape((-1,)+(1,)*(a.ndim-1)),a,b) for a,b in zip(yes,no))

    def _wm_loss(self, batch):
        S = unflatten_state(batch['s'][:,0])
        fallback = self.fallback(S)
        S = self.choose(batch['mask'][:,0].bool(),S,fallback)
        total = S[0].new_zeros(())
        std = torch.maximum(self.wm.inc_norm.std(),self.error_floor)
        for j in range(batch['a'].shape[1]):
            live = batch['mask'][:,j].bool()
            safe = self.choose(live,S,fallback)
            a = torch.where(live[:,None],batch['a'][:,j],0.)
            xi = normalize_xi(torch.where(live[:,None],batch['xi'][:,j],0.))
            prediction = self.wm.step(safe,a,xi,flat_scan(safe,self.mdp.scan_dim))
            S = self.choose(live,prediction,S)
            target = self.choose(live,unflatten_state(batch['s1'][:,j]),fallback)
            predicted = self.choose(live,prediction,fallback)
            error = (WorldModel.state_error(predicted,target)/std).square().mean(-1)
            total = total+torch.where(live,error,0.).sum()/batch['a'].shape[0]
        return total/batch['a'].shape[1]

    def world(self, batch):
        loss = self.wm_loss(batch)
        good, grad = gated_step(loss,self.wm,self.opt_wm,batch['mask'].bool().any())
        self.note(0,good)
        return torch.stack((loss.detach(),grad.detach(),good.float()))

    def _imag_step(self, S, hist, f, query, xi, use_plan=None):
        if self.event is not None:
            from ..event_planner import world_plan
            plan = self.event.plan(hist,S,f,query)
            targets = torch.where(use_plan[:,None],plan,self.task.command(S,f,query))
            action = self.policy(hist,targets)
            plan = (self.event.world_plan(plan,S,f,query) if hasattr(self.event,'world_plan')
                    else world_plan(plan,S))
        elif self.trajectory_supervision is not None:
            plan=self.policy.planning_outputs(hist,self.task.command(S,f,query))
            action=self.policy.controller_action(hist,plan[:,:13])
        else:action = self.policy(hist,self.task.command(S,f,query))
        S1 = self.wm.step(S,action,normalize_xi(xi),flat_scan(S,self.mdp.scan_dim))
        obs = self.mdp.add_noise(self.mdp.obs(S1,action,xi))
        result=(S1,torch.cat((hist[:,1:],obs[:,None]),1),action)
        return (*result,plan) if self.event is not None or self.trajectory_supervision is not None else result

    def _reward(self, S, f, a, prev, query=None):
        reward=self.mdp.reward_with_query(S,f,a,prev,query) if self.query_reward else self.mdp.reward(S,f,a,prev)
        return reward,self.mdp.soft_termination(S,f)

    def _value_targets(self, values, rewards, dones):
        target = values[-1]
        targets = []
        for t in range(self.horizon-1,-1,-1):
            target = rewards[t]+self.gamma*(1-dones[t])*((1-self.lam)*values[t+1]+self.lam*target)
            targets.append(target)
        return torch.stack(targets[::-1])

    def actor(self, batch):
        S = unflatten_state(batch['s'])
        hist,f,query,xi = batch['hist'],batch['f'],batch['goal_steps'],batch['xi']
        states,frames,actions,previous = [S],[f],[],[batch['a_prev']]
        plans=[]
        if self.event is not None:
            use_plan = torch.rand(len(f),device=f.device) < batch["eta_obs"]
        freeze(self.wm,True); freeze(self.critic,True)
        try:
            for _ in range(self.horizon):
                if self.event is not None:
                    S,hist,a,plan = self.imag_step(S,hist,f,query,xi,use_plan)
                    plans.append(plan)
                elif self.trajectory_supervision is not None:
                    S,hist,a,plan=self.imag_step(S,hist,f,query,xi);plans.append(plan)
                else:S,hist,a = self.imag_step(S,hist,f,query,xi)
                f = f+1
                states.append(S);frames.append(f);actions.append(a);previous.append(a)
            flat = lambda xs: torch.stack(xs).flatten(0,1)
            predicted = tuple(flat([s[k] for s in states[1:]]) for k in range(6))
            if self.query_reward:
                r,d=self.reward(predicted,flat(frames[1:]),flat(actions),flat(previous[:-1]),query.repeat(self.horizon))
            else:r,d = self.reward(predicted,flat(frames[1:]),flat(actions),flat(previous[:-1]))
            if self.event is not None:
                r = r + batch["eta_reward"]*self.event.imagined_delta(predicted,frames,query,plans,use_plan)
            r,d = r.view(self.horizon,-1),d.view(self.horizon,-1)
            # BlinkMimic: explicit products avoid cumprod-backward host checks.
            alive,trajectory = torch.ones_like(d[0]),[]
            for t in range(self.horizon):
                trajectory.append(alive)
                alive = alive*(1-d[t])
            survival = torch.stack(trajectory)
            discount = (self.gamma**torch.arange(self.horizon,device=r.device))[:,None]
            boot = self.critic(self.critic_input(S,f,query,normalize_xi(xi)))
            objective = (discount*survival*r).sum(0)+self.gamma**self.horizon*alive*boot
            usable = batch['usable'].float()
            loss = -(objective*usable).sum()/usable.sum().clamp_min(1)
            if self.upper_supervision is not None:
                upper_loss = self.upper_supervision(self.policy, batch)
                loss = loss+self.task.cfg.upper_supervision_weight*upper_loss
            if self.joint_supervision is not None:
                joint_loss=self.joint_supervision(self.policy,batch)
                loss=loss+joint_loss
            if self.trajectory_supervision is not None:
                stacked=tuple(torch.stack([s[k] for s in states]) for k in range(6))
                realization_loss=self.trajectory_supervision(stacked,query,torch.stack(plans),batch['usable'],survival,alive)
                loss=loss+realization_loss
            eligible=(batch['usable'].any() if self.joint_supervision is None
                      else torch.ones((),device=loss.device,dtype=torch.bool))
            good,grad = gated_step(loss,self.policy,self.opt_pi,eligible)
            self.note(1,good)
        finally:
            freeze(self.wm,False);freeze(self.critic,False)
        with torch.no_grad():
            all_states = tuple(flat([s[k].detach() for s in states]) for k in range(6))
            all_f = flat(frames)
            all_query = query.repeat(self.horizon+1)
            all_xi = normalize_xi(xi).repeat(self.horizon+1,1)
            X = self.critic_input(all_states,all_f,all_query,all_xi).view(self.horizon+1,-1,self.critic.norm.mean.numel())
            self.critic.norm.update(X,batch['usable'].repeat(self.horizon+1))
            targets = self.value_targets(self.critic(X),r.detach(),d.detach()).flatten()
            X = X[:-1].flatten(0,1)
        weights = batch['usable'].float().repeat(self.horizon)
        for _ in range(self.critic_epochs):
            groups = ((slice(None),) if self.critic_minibatches == 1 else
                      torch.randperm(len(X), device=X.device).chunk(self.critic_minibatches))
            for indices in groups:
                weight = weights[indices]
                v_loss = ((self.critic(X[indices])-targets[indices]).square()
                          * weight).sum()/weight.sum().clamp_min(1)
                eligible = weight.bool().any()
                vg, _ = gated_step(v_loss,self.critic,self.opt_v,eligible)
                self.note(2,vg | (~eligible & weights.bool().any()))
        if self.average is not None:
            with torch.no_grad():
                for avg,live in zip(self.average.parameters(),self.policy.parameters()):
                    avg.copy_(torch.where(good,avg.lerp(live,1-self.ema),avg))
                for avg,live in zip(self.average.buffers(),self.policy.buffers()):
                    avg.copy_(live)
        metrics = [loss.detach(),v_loss.detach(),grad.detach(),alive.mean().detach(),good.float()]
        if self.upper_supervision is not None:
            metrics.append(upper_loss.detach())
        if self.joint_supervision is not None:metrics.append(joint_loss.detach())
        if self.trajectory_supervision is not None:metrics.append(realization_loss.detach())
        return torch.stack(metrics)


class CudaUpdate:
    """Capture updates with fixed inputs; warm-up does NOT perform hidden training.

    Restore model/optimizer/counters/RNG after warm-up. Capture records the work;
    every call, including the first, replays exactly one optimizer update.
    """
    def __init__(self, fn, modules, optimizers, counters=(), enabled=True):
        self.fn,self.modules,self.optimizers,self.counters = fn,modules,optimizers,counters
        self.enabled,self.graph,self.static,self.output = enabled,None,None,None

    def __call__(self, batch):
        if not self.enabled:
            return self.fn(batch)
        if self.graph is None:
            self.static = {k:v.clone() for k,v in batch.items()}
            saved = [{k:v.clone() for k,v in m.state_dict().items()} for m in self.modules]
            optimizers = []
            for opt in self.optimizers:
                optimizers.extend([opt] if isinstance(opt,torch.optim.Optimizer) else [opt.muon,opt.adam])
            slots = [{p:{k:(v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k,v in s.items()}
                      for p,s in opt.state.items()} for opt in optimizers]
            counts = [t.clone() for t in self.counters]
            rng = torch.cuda.get_rng_state()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):self.fn(self.static)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            for module,state in zip(self.modules,saved):module.load_state_dict(state)
            for opt,old in zip(optimizers,slots):
                for p,state in opt.state.items():
                    for k,value in state.items():
                        if torch.is_tensor(value):
                            value.copy_(old[p][k] if p in old and k in old[p] else torch.zeros_like(value))
            for tensor,value in zip(self.counters,counts):tensor.copy_(value)
            torch.cuda.set_rng_state(rng)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.output = self.fn(self.static)
            self.graph.replay()
            return self.output.clone()
        for key,value in batch.items():self.static[key].copy_(value)
        self.graph.replay()
        return self.output.clone()
