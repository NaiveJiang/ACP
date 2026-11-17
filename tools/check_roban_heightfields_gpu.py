"""Exercise distinct heightfields in batched physics, including reset after capture."""
import os
os.environ["ACP_ROBOT"]="roban"
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.motion import MotionLib
from supermimic.acp.mdp import MDP
from supermimic.acp.env import TrackingSim
from supermimic.acp.robot import default_joint_pos_list,NUM_J

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--motion-dir',required=True);ap.add_argument('--report',required=True)
args=ap.parse_args();torch.set_num_threads(4)
kin=RobotKinematics('cuda:0');motion=MotionLib.from_dir(args.motion_dir,kin,'cuda:0')
mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device='cuda:0'))
sim=TrackingSim(2,motion,mdp,'cuda:0',eval_mode=True,reset_noise=False)
ids=torch.arange(2,device='cuda:0');start=motion.clip_start[:2]
sim.reset(ids,frames=start,variants=torch.zeros(2,device='cuda:0',dtype=torch.long))
assert sim.hf_geom is not None
before=sim.sim.model.geom_dataid[:,sim.hf_geom].tolist();print('heightfield_dataids',before,flush=True)
assert len(set(before))==2
for _ in range(10):sim.step(torch.zeros(2,NUM_J,device='cuda:0'))
assert torch.isfinite(sim.sim.data.qpos).all()
sim.reset(ids,frames=start.flip(0),variants=torch.zeros(2,device='cuda:0',dtype=torch.long))
for _ in range(10):sim.step(torch.zeros(2,NUM_J,device='cuda:0'))
assert torch.isfinite(sim.sim.data.qpos).all()
after=sim.sim.model.geom_dataid[:,sim.hf_geom].tolist();assert after==before[::-1]
print('swapped_heightfield_dataids',after,flush=True)
Path(args.report).write_text(json.dumps(dict(worlds=2,distinct_heightfields=2,swap_after_graph_capture=True,control_steps=20,finite=True),indent=2)+'\n')
