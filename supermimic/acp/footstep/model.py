"""Three-head MLP; no dependency on the original ACP planner or simulator."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .data import STATE_DIM


class FootstepPlanner(nn.Module):
    def __init__(self, hist=5, hidden=(256, 256, 128)):
        super().__init__()
        self.hist = hist
        self.register_buffer("state_mean", torch.zeros(STATE_DIM))
        self.register_buffer("state_std", torch.ones(STATE_DIM))
        self.register_buffer("command_mean", torch.zeros(4))
        self.register_buffer("command_std", torch.ones(4))
        layers, dim = [], hist*STATE_DIM+4
        for width in hidden:
            layers += [nn.Linear(dim, width), nn.ELU()]
            dim = width
        self.encoder = nn.Sequential(*layers)
        self.contact = nn.Linear(dim, 2)
        self.position = nn.Linear(dim, 6)
        self.heading = nn.Linear(dim, 4)
        with torch.no_grad():
            self.heading.bias.copy_(torch.tensor([0., 1., 0., 1.]))

    @torch.no_grad()
    def set_normalization(self, state, command):
        self.state_mean.copy_(state.mean(dim=0))
        self.state_std.copy_(state.std(dim=0, unbiased=False).clamp_min(.01))
        self.command_mean.copy_(command.mean(dim=0))
        self.command_std.copy_(command.std(dim=0, unbiased=False).clamp_min(.01))

    def forward(self, state_history, velocity_command, query_dt):
        state = ((state_history-self.state_mean)/self.state_std).flatten(1)
        cmd = (torch.cat((velocity_command, query_dt), dim=-1)-self.command_mean)/self.command_std
        z = self.encoder(torch.cat((state, cmd), dim=-1))
        heading = self.heading(z).reshape(-1, 2, 2)
        return dict(contact_logits=self.contact(z), position=self.position(z).reshape(-1, 2, 3),
                    heading=F.normalize(heading, dim=-1, eps=1e-6))

    @torch.no_grad()
    def command(self, state_history, velocity_command, query_dt):
        """Decode to the agreed 16-D tracker instruction; no tracker is run here."""
        pred = self(state_history, velocity_command, query_dt)
        contact = pred["contact_logits"].sigmoid() > .5
        pos = torch.where(contact[..., None], pred["position"], 0.)
        head = torch.where(contact[..., None], pred["heading"], 0.)
        return torch.cat((velocity_command, query_dt, contact.float(), pos.flatten(1), head.flatten(1)), dim=-1)


def planner_loss(pred, batch, position_scale=.2):
    mask = batch["support_mask"]
    denom = mask.sum().clamp_min(1)
    contact = F.binary_cross_entropy_with_logits(pred["contact_logits"], batch["target_contact"])
    position = (F.smooth_l1_loss(pred["position"]/position_scale,
                                batch["target_position"]/position_scale, reduction="none").mean(-1)*mask).sum()/denom
    heading = ((1-(pred["heading"]*batch["target_heading"]).sum(-1))*mask).sum()/denom
    return dict(total=contact+position+heading, contact=contact, position=position, heading=heading)
