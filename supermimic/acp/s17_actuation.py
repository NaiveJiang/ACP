"""S17 motor parameters from kuavo_amp_mjlab e639e8fa92a2."""

import math

import torch

# kp, kd, armature for waist, the two legs, and the two arms.
WAIST = (33.458, 3.195, 0.0339)
HIP_KNEE = (125.147, 11.951, 0.1268)
ANKLE = (59.020, 5.636, 0.0598)
ARM = (29.510, 2.818, 0.0299)
LEG = (HIP_KNEE, HIP_KNEE, WAIST, HIP_KNEE, ANKLE, ANKLE)
ARM_GAINS = ((4.935, 0.471, 0.0050), ARM, ARM, ARM)
GAINS = (WAIST, *LEG, *LEG, *ARM_GAINS, *ARM_GAINS)
DELAY_STEPS = 10
ACTION_SCALE = 0.25

# Output-shaft speed in rpm and available driving torque relative to the peak.
LARGE = (
    (0, 1),
    (70, 1),
    (80, 114 / 128),
    (90, 94 / 128),
    (100, 59.3 / 128),
    (110, 22 / 128),
    (120, 5 / 128),
    (130, 1.5 / 128),
    (140, 0),
)
YAW = (
    (0, 1),
    (30, 1),
    (40, 49.8 / 53),
    (50, 45.9 / 53),
    (60, 42.1 / 53),
    (70, 38.1 / 53),
    (80, 34 / 53),
    (90, 25.2 / 53),
    (100, 9.1 / 53),
    (110, 0),
)
SMALL = (
    (0, 1),
    (115, 1),
    (125, 22.8 / 29.1),
    (135, 17.1 / 29.1),
    (145, 12.3 / 29.1),
    (155, 7.24 / 29.1),
    (165, 0),
)
SHOULDER = (
    (0, 1),
    (50, 1),
    (60, 17.1 / 18.6),
    (70, 15.7 / 18.6),
    (80, 13.6 / 18.6),
    (90, 12.5 / 18.6),
    (100, 11.5 / 18.6),
    (110, 9.9 / 18.6),
    (120, 7.8 / 18.6),
    (130, 5.4 / 18.6),
    (140, 3.8 / 18.6),
    (150, 1.9 / 18.6),
    (158, 0),
)
LEG_CURVES = (LARGE, LARGE, YAW, LARGE, SMALL, SMALL)
ARM_CURVES = (SHOULDER, SMALL, SMALL, SMALL)
CURVES = (None, *LEG_CURVES, *LEG_CURVES, *ARM_CURVES, *ARM_CURVES)


def torque_curve(points, speed_limit, device):
    if points is None:
        return torch.tensor((0.0, speed_limit), device=device), torch.tensor(
            (1.0, 0.0), device=device
        )
    rpm, ratio = torch.tensor(points, device=device).T
    return (rpm * (math.tau / 60)).contiguous(), ratio.contiguous()


def limit_torque(torque, velocity, effort, speed, ratio):
    """Derate driving torque with speed and retain peak braking torque."""
    magnitude = velocity.abs().clamp(max=speed[-1])
    upper = torch.searchsorted(speed, magnitude.contiguous()).clamp(1, len(speed) - 1)
    lower = upper - 1
    weight = (magnitude - speed[lower]) / (speed[upper] - speed[lower])
    driving = effort * torch.lerp(ratio[lower], ratio[upper], weight)
    minimum = torch.where(velocity <= 0, -driving, -effort)
    maximum = torch.where(velocity >= 0, driving, effort)
    return torch.clamp(torque, min=minimum, max=maximum)


def append_targets(queue, action, steps):
    """Shift an oldest-first physics FIFO by repeated 50 Hz position targets."""
    targets = action.unsqueeze(-2).expand(*action.shape[:-1], steps, action.shape[-1])
    return torch.cat((queue[..., steps:, :], targets), dim=-2)
