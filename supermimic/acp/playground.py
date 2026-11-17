"""A connected RoBan playground: exact boxes and a shared scanned/physical heightfield."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import mujoco
import numpy as np
import torch

from . import heightfield
from .robot import JOINT_NAMES, get_spec, pd_params


@dataclass
class Station:
    name: str
    description: str
    origin: tuple[float, float]
    color: tuple[float, float, float, float]

    @property
    def start(self):
        return np.array([self.origin[0] + .25, self.origin[1], 0.])

    @property
    def goal(self):
        return np.array([self.origin[0] + 7.15, self.origin[1], 0.])


class Playground:
    bounds = (-2., 36., -3., 25.)
    spacing = .05

    def __init__(self, output: Path, seed: int = 0):
        output.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.stations = []
        self.boxes, self.colors = [], []
        xmin, xmax, ymin, ymax = self.bounds
        self.x = np.linspace(xmin, xmax, round((xmax-xmin)/self.spacing)+1)
        self.y = np.linspace(ymin, ymax, round((ymax-ymin)/self.spacing)+1)
        self.heights = np.zeros((len(self.y), len(self.x)), dtype=np.float32)
        rng = np.random.default_rng(seed)
        entries = [
            ('Flat', 'Open flat lane'), ('Stairs 5cm', '5 cm steps, up and down'),
            ('Stairs 10cm', '10 cm steps, up and down'), ('Stairs 15cm', '15 cm steps, up and down'),
            ('Variable stairs', 'Unequal 6-12 cm rises'), ('Ramp 8deg', 'Gentle ascent and descent'),
            ('Ramp 15deg', 'Steep ascent and descent'), ('Cross slope', 'Side slope with smooth approaches'),
            ('Rough 2cm', 'Low undulating ground'), ('Rough 5cm', 'Higher undulating ground'),
            ('Thresholds', 'Repeated 6 and 10 cm thresholds'), ('Low platforms', '10 / 20 / 30 cm platforms'),
            ('Box 45cm', '45 cm climb-over box'), ('Box 60cm', '60 cm climb-over box'),
            ('Stepping stones', 'Raised stones separated by gaps'), ('Narrow beam', '30 cm wide raised beam'),
            ('Gap 20cm', '20 cm gap between low decks'), ('Slalom boxes', 'Staggered obstacles'),
            ('Mixed course', 'Steps, low box and thresholds'), ('Walled stairs', 'Staircase between side walls'),
        ]
        palette = [(0.43,.67,.72,1.), (.64,.65,.81,1.), (.76,.58,.39,1.), (.46,.69,.50,1.), (.72,.51,.60,1.)]
        for i, (name, description) in enumerate(entries):
            ox, oy = (i % 4) * 9., (i // 4) * 5.5
            station = Station(name, description, (ox, oy), palette[i//4])
            self.stations.append(station)
            def box(x0, x1, y0, y1, top):
                self.boxes.append([ox+(x0+x1)/2, oy+(y0+y1)/2, top, 1., 0.,
                                   (x1-x0)/2, (y1-y0)/2, top/2, 0.])
                self.colors.append(station.color)
            if i in (1, 2, 3, 4, 19):
                rises = [(.05,.10,.15)[i-1]]*4 if i in (1,2,3) else ([.06,.09,.12,.08] if i==4 else [.08]*4)
                for k, top in enumerate(np.cumsum(rises)):
                    box(1.25+k*.38, 5.6-k*.38, -.85, .85, float(top))
                if i==19:
                    for a,b in [(-1.05,-.9),(.9,1.05)]:box(1.,5.9,a,b,.75)
            elif i in (5,6,7,8,9):
                ix=np.where((self.x>=ox+1.)&(self.x<=ox+6.))[0]
                iy=np.where((self.y>=oy-1.25)&(self.y<=oy+1.25))[0]
                xx,yy=np.meshgrid(self.x[ix]-ox,self.y[iy]-oy)
                edge=np.clip((1.25-np.abs(yy))/.25,0,1)
                along=np.clip(np.minimum(xx-1,6-xx)/.5,0,1)
                if i in (5,6):
                    h=math.tan(math.radians(8 if i==5 else 15))*np.clip(np.minimum(xx-1,6-xx),0,2.)*edge
                elif i==7:
                    h=(yy+1.25)*.14*edge*along
                else:
                    amplitude=.02 if i==8 else .05
                    phase=rng.uniform(-math.pi,math.pi)
                    h=amplitude*(1+.5*np.sin(xx*8+phase)+.5*np.sin(yy*10+xx*3))*edge*along
                self.heights[np.ix_(iy,ix)]=h
            elif i==10:
                for k in range(6):box(1.4+k*.7,1.55+k*.7,-1.,1.,.06 if k%2==0 else .10)
            elif i==11:
                for k in range(3):box(1.4+k*1.45,2.4+k*1.45,-1.,1.,.1*(k+1))
            elif i in (12,13):box(2.45,3.85,-1.05,1.05,.45 if i==12 else .60)
            elif i in (14,15,16):
                # Low approach/departure steps make these lanes accessible from the plaza.
                box(.85,1.15,-.65,.65,.10);box(1.15,1.5,-.65,.65,.20)
                box(5.75,6.05,-.65,.65,.20);box(6.05,6.35,-.65,.65,.10)
                if i==14:
                    for k in range(6):
                        cy=.08*(-1)**k;box(1.5+k*.7,2.0+k*.7,cy-.27,cy+.27,.25)
                elif i==15:box(1.5,5.75,-.15,.15,.25)
                else:
                    box(1.5,3.5,-.65,.65,.25);box(3.7,5.75,-.65,.65,.25)
            elif i==17:
                for k in range(4):
                    cy=.6*(-1)**k;box(1.3+k*1.15,1.8+k*1.15,cy-.45,cy+.45,.3)
            elif i==18:
                box(1.2,2.9,-.95,.95,.08);box(1.6,2.5,-.95,.95,.16)
                box(3.5,4.4,-.85,.85,.25);box(5.1,5.3,-.9,.9,.08)
        self.boxes=np.asarray(self.boxes,dtype=np.float32).reshape(-1,9)
        self.heightfield_path=output/f'arena_v1_seed{seed}_heightfield.npz'
        np.savez_compressed(self.heightfield_path,heights=self.heights,
                            origin_xy=np.array([xmin,ymin]),spacing=np.array([self.spacing,self.spacing]))
        self.hf_index=heightfield.register(self.heightfield_path)
        self.terrain=np.concatenate([heightfield.token(self.hf_index)[None],self.boxes]).astype(np.float32)
        self._tensors={}
        self.manifest_path=output/f'arena_v1_seed{seed}.json'
        self.manifest_path.write_text(json.dumps(dict(bounds=self.bounds,seed=seed,heightfield=str(self.heightfield_path),
            stations=[dict(id=i+1,name=s.name,description=s.description,origin=s.origin,
                           start=s.start.tolist(),goal=s.goal.tolist()) for i,s in enumerate(self.stations)],
            boxes=self.boxes.tolist()),indent=2)+'\n')

    def tensor(self, device, xy=None):
        key=str(device)
        if key not in self._tensors:self._tensors[key]=torch.as_tensor(self.terrain,device=device).unsqueeze(0)
        if xy is None:return self._tensors[key]
        # Include all geometry within both scan footprints, with ample margin.
        near=(np.abs(self.boxes[:,0]-xy[0])<=self.boxes[:,5]+2.) & (np.abs(self.boxes[:,1]-xy[1])<=self.boxes[:,6]+2.)
        ids=np.r_[0,np.flatnonzero(near)+1]
        return self._tensors[key][:,ids]

    def height(self, xy):
        xy=np.asarray(xy,dtype=np.float32)
        shape=xy.shape[:-1];points=xy.reshape(-1,2)
        u=(points[:,0]-self.x[0])/self.spacing;v=(points[:,1]-self.y[0])/self.spacing
        ix=np.clip(np.floor(u).astype(int),0,len(self.x)-2);iy=np.clip(np.floor(v).astype(int),0,len(self.y)-2)
        tx,ty=u-ix,v-iy
        h00,h10=self.heights[iy,ix],self.heights[iy,ix+1]
        h01,h11=self.heights[iy+1,ix],self.heights[iy+1,ix+1]
        height=np.where(tx>=ty,h00+(h10-h00)*tx+(h11-h10)*ty,h00+(h11-h01)*tx+(h01-h00)*ty)
        height=np.where((u>=0)&(u<=len(self.x)-1)&(v>=0)&(v<=len(self.y)-1),height,0.)
        inside=(np.abs(points[:,None,0]-self.boxes[:,0])<=self.boxes[:,5]) & (np.abs(points[:,None,1]-self.boxes[:,1])<=self.boxes[:,6])
        return np.maximum(height,np.where(inside,self.boxes[:,2],0.).max(axis=1)).reshape(shape)

    def build_model(self):
        spec=get_spec()
        spec.option.timestep=.005
        spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.solver=mujoco.mjtSolver.mjSOL_NEWTON
        spec.option.cone=mujoco.mjtCone.mjCONE_PYRAMIDAL
        spec.option.iterations=100;spec.option.ls_iterations=50;spec.option.tolerance=1e-6
        spec.option.ccd_iterations=35
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                geom.friction[:]=[1.,.005,.0001]
        spec.worldbody.add_geom(name='playground_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.1],
            pos=[0,0,-.1],rgba=[.21,.25,.29,1],friction=[1,.005,.0001],group=2)
        hf=heightfield.add_geom(spec,self.hf_index,'playground_heightfield')
        hf.rgba[:]=[.47,.56,.46,1]
        hf.friction[:]=[1,.005,.0001]
        hf.group=2
        for i,(row,color) in enumerate(zip(self.boxes,self.colors)):
            cx,cy,top,c,s,hx,hy,hz=row[:8]
            spec.worldbody.add_geom(name=f'playground_box_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[cx,cy,top-hz],size=[hx,hy,hz],rgba=color,friction=[1,.005,.0001],group=2)
        # Painted lane boundaries are purely visual and never appear in scans or collisions.
        for i,station in enumerate(self.stations):
            ox,oy=station.origin
            for side in [-1,1]:
                spec.worldbody.add_geom(name=f'lane_{i}_{side}',type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[ox+3.6,oy+side*1.7,.003],size=[3.7,.018,.002],rgba=station.color,
                    contype=0,conaffinity=0)
        spec.worldbody.add_light(name='playground_sun',pos=[8,8,20],dir=[-.3,-.3,-1],
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,castshadow=False,
                                 diffuse=[.55,.55,.55],ambient=[.1,.1,.1])
        for joint,limit in zip(JOINT_NAMES,pd_params()[2]):
            spec.add_actuator(name=f'play_{joint}',target=joint,trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gaintype=mujoco.mjtGain.mjGAIN_FIXED,gainprm=[1,0,0,0,0,0,0,0,0,0],
                biastype=mujoco.mjtBias.mjBIAS_NONE,dyntype=mujoco.mjtDyn.mjDYN_NONE,
                gear=[1,0,0,0,0,0],ctrllimited=True,ctrlrange=[-limit,limit],forcelimited=True,forcerange=[-limit,limit])
        model=spec.compile()
        model.vis.headlight.ambient[:]=.2
        model.vis.headlight.diffuse[:]=.25
        model.vis.map.zfar=100
        return model
