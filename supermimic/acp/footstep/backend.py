"""Explicit checkpoint dispatch; legacy velocity and contact-only tasks stay distinct."""
from types import SimpleNamespace

VELOCITY_FORMAT = "two_foot_oracle_tracker_v1"
CONTACT_FORMAT = "two_foot_contact_only_tracker_v2"
STABLE_CONTACT_FORMAT = "two_foot_stable_contact_tracker_v3"


def backend_for_format(format_name):
    if format_name == STABLE_CONTACT_FORMAT:
        from .stable_contact import (StableContactTask, StableContactMDP, StableContactPolicy,
                                     StableContactRewardCfg)
        return SimpleNamespace(Task=StableContactTask, MDP=StableContactMDP, Policy=StableContactPolicy,
                               RewardCfg=StableContactRewardCfg, obs_dim=69, command_dim=13,
                               contact_offset=1, format=format_name)
    if format_name == VELOCITY_FORMAT:
        from .task import FootstepTask, FootstepMDP, TrackerPolicy, ContactRewardCfg, OBS_DIM, COMMAND_DIM
        return SimpleNamespace(Task=FootstepTask, MDP=FootstepMDP, Policy=TrackerPolicy,
                               RewardCfg=ContactRewardCfg, obs_dim=OBS_DIM, command_dim=COMMAND_DIM,
                               contact_offset=4, format=format_name)
    if format_name == CONTACT_FORMAT:
        from .contact_only.task import (ContactOnlyTask, ContactOnlyMDP, ContactOnlyPolicy,
                                        ContactOnlyRewardCfg, OBS_DIM, COMMAND_DIM)
        return SimpleNamespace(Task=ContactOnlyTask, MDP=ContactOnlyMDP, Policy=ContactOnlyPolicy,
                               RewardCfg=ContactOnlyRewardCfg, obs_dim=OBS_DIM, command_dim=COMMAND_DIM,
                               contact_offset=1, format=format_name)
    raise ValueError("Unsupported tracker checkpoint format: "+str(format_name))
