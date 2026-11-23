"""Explicit checkpoint dispatch; legacy velocity and contact-only tasks stay distinct."""
from types import SimpleNamespace

VELOCITY_FORMAT = "two_foot_oracle_tracker_v1"
CONTACT_FORMAT = "two_foot_contact_only_tracker_v2"
STABLE_CONTACT_FORMAT = "two_foot_stable_contact_tracker_v3"
UPPER_SUPERVISED_FORMAT = "two_foot_upper_supervised_tracker_v4"
GAIT_STYLE_FORMAT = "two_foot_gait_style_tracker_v5"
LEG_COLLISION_FORMAT = "two_foot_leg_collision_tracker_v6"
NEXT_CONTACT_FORMAT = "two_foot_acp_next_contact_v1"
BODY_GOAL_FORMAT = "two_foot_body_goal_tracker_v1"
JOINT_FORMAT = "two_foot_joint_planner_controller_v7"
CONSISTENT_FORMAT = "two_foot_consistent_joint_controller_v8"


WALKING_FORMAT = "two_foot_fixed_controller_planner_v9"


def backend_for_format(format_name):
    if format_name == NEXT_CONTACT_FORMAT:
        from .next_contact import NextContactMDP, NextContactPolicy, NextContactTask, COMMAND_DIM, REFERENCE_TAIL
        from .leg_collision import LegCollisionCfg, LegCollisionSim
        return SimpleNamespace(Task=NextContactTask, MDP=NextContactMDP, Policy=NextContactPolicy,
                               RewardCfg=LegCollisionCfg, Env=LegCollisionSim, obs_dim=69,
                               command_dim=COMMAND_DIM, contact_offset=0, format=format_name,
                               reserve_steps=REFERENCE_TAIL)
    if format_name == BODY_GOAL_FORMAT:
        from .body_goal import BodyGoalMDP, BodyGoalPolicy, BodyGoalTask, COMMAND_DIM
        from .leg_collision import LegCollisionCfg, LegCollisionSim
        return SimpleNamespace(Task=BodyGoalTask, MDP=BodyGoalMDP, Policy=BodyGoalPolicy,
                               RewardCfg=LegCollisionCfg, Env=LegCollisionSim, obs_dim=69,
                               command_dim=COMMAND_DIM, contact_offset=1, format=format_name)
    if format_name == WALKING_FORMAT:
        from .walking_planner import WalkingPlannerPolicy, WalkingPlannerTask
        backend = backend_for_format(CONSISTENT_FORMAT)
        backend.Policy = WalkingPlannerPolicy
        backend.Task = WalkingPlannerTask
        backend.format = WALKING_FORMAT
        return backend
    if format_name == CONSISTENT_FORMAT:
        from .consistent_joint import ConsistentJointPolicy,ConsistentJointTask,ConsistentJointMDP,ConsistentJointCfg
        from .leg_collision import LegCollisionSim
        return SimpleNamespace(Task=ConsistentJointTask,MDP=ConsistentJointMDP,Policy=ConsistentJointPolicy,
                               RewardCfg=ConsistentJointCfg,Env=LegCollisionSim,obs_dim=69,command_dim=4,
                               contact_offset=1,format=format_name,joint=True,consistent=True)
    if format_name == JOINT_FORMAT:
        from .joint import JointPolicy,JointTask,JointMDP,JointCfg
        from .leg_collision import LegCollisionSim
        return SimpleNamespace(Task=JointTask,MDP=JointMDP,Policy=JointPolicy,RewardCfg=JointCfg,
                               Env=LegCollisionSim,obs_dim=69,command_dim=4,contact_offset=1,format=format_name,joint=True)
    if format_name == LEG_COLLISION_FORMAT:
        from .leg_collision import LegCollisionTask,LegCollisionMDP,LegCollisionPolicy,LegCollisionCfg,LegCollisionSim
        return SimpleNamespace(Task=LegCollisionTask,MDP=LegCollisionMDP,Policy=LegCollisionPolicy,
                               RewardCfg=LegCollisionCfg,Env=LegCollisionSim,obs_dim=69,command_dim=13,
                               contact_offset=1,format=format_name)
    if format_name == GAIT_STYLE_FORMAT:
        from .gait_style import GaitStyleTask, GaitStyleMDP, GaitStylePolicy, GaitStyleCfg
        return SimpleNamespace(Task=GaitStyleTask, MDP=GaitStyleMDP, Policy=GaitStylePolicy,
                               RewardCfg=GaitStyleCfg, obs_dim=69, command_dim=13,
                               contact_offset=1, format=format_name)
    if format_name == UPPER_SUPERVISED_FORMAT:
        from .upper_supervised import (UpperSupervisedTask, UpperSupervisedMDP,
                                       UpperSupervisedPolicy, UpperSupervisedCfg)
        return SimpleNamespace(Task=UpperSupervisedTask, MDP=UpperSupervisedMDP, Policy=UpperSupervisedPolicy,
                               RewardCfg=UpperSupervisedCfg, obs_dim=69, command_dim=13,
                               contact_offset=1, format=format_name)
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
