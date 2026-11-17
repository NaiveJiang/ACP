"""Export source non-stair rollouts with explicit finite heightfields or exact boxes.

Run in .venv_roban_collect (same MuJoCo version as the saved source model).
Continuous surfaces are rasterized at 50 mm. Vertical cliffs occupy at most one
grid cell; smooth-surface errors and the cliff band are recorded independently.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from export_roban_acp import terrain_boxes, heights


GRID = .05


def ray_heights(m,d,xy):
    out=np.empty(len(xy));gid=np.zeros(1,np.int32)
    for i,(x,y) in enumerate(xy):
        distance=mujoco.mj_ray(m,d,np.array([x,y,30.]),np.array([0.,0.,-1.]),None,1,-1,gid)
        if distance<0:
            # Exact floating-point tile seams can miss both adjacent faces.
            # Only bridge a sub-micrometre crack when all four diagonal probes hit.
            near=[mujoco.mj_ray(m,d,np.array([x+dx,y+dy,30.]),np.array([0.,0.,-1.]),None,1,-1,gid)
                  for dx,dy in ((1e-6,1e-6),(1e-6,-1e-6),(-1e-6,1e-6),(-1e-6,-1e-6))]
            if any(v<0 for v in near):raise ValueError("uncovered source terrain in scan envelope")
            distance=min(near)
        out[i]=30-distance
    return out


def interpolate(h, origin, step, xy):
    u,v=((xy-origin)/step).T
    ix=np.clip(np.floor(u).astype(int),0,h.shape[1]-2);iy=np.clip(np.floor(v).astype(int),0,h.shape[0]-2)
    tx,ty=u-ix,v-iy;a,b,c,e=h[iy,ix],h[iy,ix+1],h[iy+1,ix],h[iy+1,ix+1]
    return np.where(tx>=ty,a+(b-a)*tx+(e-b)*ty,a+(e-c)*tx+(c-a)*ty)


def export(run, folder, dest, model, data, boxes, gids, cache):
    meta=json.loads((folder/'episode.json').read_text())
    first=meta.get('first_done')
    if meta['incomplete'] or not first or first['terminated'] or not meta['finite']:
        return []
    if meta.get('max_route_progress',0)<.5 or meta['max_lateral_deviation']>.5:
        return []
    raw=np.load(folder/'states.npz');allq=raw['qpos'].astype(float)
    # Long, stable traversals may be split into overlapping training windows.
    # A timeout is recorded honestly; full source-goal completion is not inferred.
    starts=[0] if len(allq)<=901 else [0,len(allq)-901]
    results=[]
    for begin in starts:
        q=allq[begin:begin+901].copy()
        if len(q)<301:continue
        lo=np.floor((q[:,:2].min(0)-1.9)/GRID)*GRID
        hi=np.ceil((q[:,:2].max(0)+1.9)/GRID)*GRID
        key=tuple(np.r_[lo,hi])
        continuous='slope' in meta['terrain_type'] or meta['terrain_type']=='random_rough'
        if continuous:
            if key not in cache:
                xx,yy=np.arange(lo[0],hi[0]+GRID/2,GRID),np.arange(lo[1],hi[1]+GRID/2,GRID)
                xy=np.stack(np.meshgrid(xx,yy),-1).reshape(-1,2)
                h=ray_heights(model,data,xy).reshape(len(yy),len(xx))
                # Evaluate cell centers; separate discontinuities from smooth facets.
                mids=np.stack(np.meshgrid((xx[:-1]+xx[1:])/2,(yy[:-1]+yy[1:])/2),-1).reshape(-1,2)
                actual=ray_heights(model,data,mids)
                predicted=interpolate(h,lo,GRID,mids)
                corners=np.stack((h[:-1,:-1],h[:-1,1:],h[1:,:-1],h[1:,1:]))
                cliff=np.ptp(corners,axis=0).ravel()>.15
                error=np.abs(actual-predicted)
                smooth=float(error[~cliff].max()) if (~cliff).any() else 0.
                if smooth>.011:raise ValueError(f"smooth heightfield error {smooth}")
                cache[key]=(h,dict(smooth_cell_center_max_error_m=smooth,all_cell_center_max_error_m=float(error.max()),
                                   cliff_cells=int(cliff.sum()),grid_spacing_m=GRID,cliff_transition_width_m=GRID))
            h,quality=cache[key];floor=float(h.min());b=np.empty((0,10));selected_ids=[]
        else:
            yaw=2*np.arctan2(boxes[:,6],boxes[:,3]);c,s=np.abs(np.cos(yaw)),np.abs(np.sin(yaw))
            half=np.stack((c*boxes[:,7]+s*boxes[:,8],s*boxes[:,7]+c*boxes[:,8]),-1)/2
            mask=((boxes[:,:2]+half>=lo)&(boxes[:,:2]-half<=hi)).all(-1)
            b=boxes[mask].copy();selected_ids=gids[mask].tolist()
            if not len(b):raise ValueError('no boxes under source trajectory')
            floor=float(np.min(b[:,2]+b[:,9]/2))
            xx,yy=np.arange(lo[0],hi[0]+.025,.05),np.arange(lo[1],hi[1]+.025,.05)
            xy=np.stack(np.meshgrid(xx,yy),-1).reshape(-1,2)
            exact=ray_heights(model,data,xy);estimated=heights(xy,b,floor=floor)
            errors=np.abs(exact-estimated)
            for i in np.flatnonzero(errors>.001):
                near=xy[i]+np.array([[1e-6,1e-6],[1e-6,-1e-6],[-1e-6,1e-6],[-1e-6,-1e-6]])
                errors[i]=min(errors[i],float(np.min(np.abs(ray_heights(model,data,near)-estimated[i]))))
            error=float(errors.max())
            if error>.001:raise ValueError(f"box/source scan mismatch {error}")
            quality=dict(surface_check_max_error_m=error,boundary_tolerance_m=1e-6)
        translation=np.r_[-q[0,:2],-floor];q[:,:3]+=translation
        q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
        for i in range(1,len(q)):
            if q[i,3:7]@q[i-1,3:7]<0:q[i,3:7]*=-1
        category=meta['terrain_type'];clip=f"{category}/{run.name}_{folder.name}_s{begin:04d}"
        prefix=dest/clip;prefix.parent.mkdir(parents=True,exist_ok=True)
        motion=Path(str(prefix)+'_motion.npz');terrain=Path(str(prefix)+'_terrain.npy')
        np.savez_compressed(motion,joint_pos=q.astype(np.float32),joint_names=raw['joint_names'],fps=np.float32(50),
                            robot_id=np.array('roban'),source_episode=np.array(str(folder.resolve())))
        if continuous:
            hf=Path(str(prefix)+'_heightfield.npz')
            np.savez_compressed(hf,heights=(h+translation[2]).astype(np.float32),origin_xy=lo+translation[:2],spacing=np.array([GRID,GRID]))
        else:
            b[:,:3]+=translation;b=b[b[:,2]+b[:,9]/2>1e-6]
        np.save(terrain,b[:,None].astype(np.float32))
        record=dict(episode=str(folder.resolve()),accepted=True,clip=clip,frames=len(q),source_frames=[begin,begin+len(q)],
            source_to_acp_translation=translation.tolist(),box_count=len(b),source_geom_ids=selected_ids,
            split_group=f"{run.name}/tile_{meta['terrain_row']}_{meta['terrain_col']}",difficulty=meta.get('difficulty'),
            source_course_completed='course_complete' in first['terms'],status='requires_contact_validation',
            terrain_representation='heightfield_v1' if continuous else 'yaw_boxes_v1',**quality)
        for path,key in [(motion,'motion_sha256'),(terrain,'terrain_sha256')]+([(hf,'heightfield_sha256')] if continuous else []):
            record[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        results.append(record)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--per-category',type=int,default=40);ap.add_argument('--categories',nargs='+')
    args=ap.parse_args();run=args.run.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_binary_path(str(run/'source_model.mjb'));d=mujoco.MjData(m);d.qpos[2]=-100;mujoco.mj_forward(m,d)
    with np.load(run/'terrain_snapshot.npz') as z:
        snap=dict(z);mask=(snap['types']==mujoco.mjtGeom.mjGEOM_BOX)&(np.abs(snap['quat'][:,1:3])<1e-6).all(-1)
        for name in ('geom_id','names','types','pos','quat','size','dataid'):snap[name]=snap[name][mask]
        boxes,gids=terrain_boxes(snap)
    groups=defaultdict(lambda:defaultdict(list))
    for folder in sorted((run/'raw').iterdir()):
        meta=json.loads((folder/'episode.json').read_text())
        if args.categories and meta['terrain_type'] not in args.categories:continue
        if not meta['incomplete'] and meta['first_done'] and not meta['first_done']['terminated']:
            groups[meta['terrain_type']][(meta['terrain_row'],meta['terrain_col'])].append(folder)
    records=[];cache={}
    for category,tiles in groups.items():
        count=0;pending=list(tiles.values())
        while any(pending) and count<args.per_category:
            for tile in pending:
                if not tile or count>=args.per_category:continue
                folder=tile.pop(0)
                try:
                    rows=export(run,folder,out,m,d,boxes,gids,cache);records.extend(rows);count+=len(rows)
                except ValueError as exc:
                    # A later window can fail after an earlier window was written.
                    # Keep no unmanifested partial episode in the candidate set.
                    for partial in (out/category).glob(f"{run.name}_{folder.name}_s*"):
                        partial.unlink()
                    records.append(dict(episode=str(folder),accepted=False,reason=str(exc)))
                (out/'export_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
        print('EXPORTED',category,count,flush=True)
    print(Counter(r.get('reason') for r in records if not r['accepted']))


if __name__=='__main__':main()
