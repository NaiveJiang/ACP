"""Render synchronized reference-command / planner-command physical rollouts.

Renders all category medians, selected without outcome filtering, followed by
up to two additional planner-only failures. Uses exact recorded post-step
states; a terminated robot is frozen while the comparison continues.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

os.environ.setdefault('ACP_ROBOT','roban')
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from supermimic.acp import heightfield
from supermimic.acp.render import scene_xml,mat_to_quat,TARGET_RGBA
from supermimic.acp.robot import JOINT_NAMES

CATEGORIES={
    'flat':'平地前行','flat_backward':'后退','flat_left':'左侧行','flat_right':'右侧行',
    'flat_turn_left':'前进左转','flat_turn_right':'前进右转',
    'pyramid_stairs':'普通凸形楼梯','pyramid_stairs_narrow':'窄凸形楼梯',
    'pyramid_inv_stairs':'普通凹形楼梯','pyramid_inv_stairs_narrow':'窄凹形楼梯',
    'variable_height_stairs_up':'变高上楼梯','variable_height_stairs_down':'变高下楼梯',
    'hf_pyramid_slope':'宽凸坡','hf_pyramid_slope_inv':'宽凹坡',
    'hf_pyramid_slope_narrow':'窄凸坡','hf_pyramid_slope_inv_narrow':'窄凹坡',
    'random_rough':'随机起伏','threshold':'连续门槛','narrow_beams':'窄梁','random_spread_boxes':'散布箱体',
}


def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--only',nargs='+',help='Render selected categories only, for preview')
    args=ap.parse_args();src=args.directory.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    ref=load(src/'nominal_reference_seed0.npz');plan=load(src/'nominal_planner_seed0.npz')
    dr=json.loads((src/'nominal_reference_seed0.json').read_text());dp=json.loads((src/'nominal_planner_seed0.json').read_text())
    assert dr['checkpoint_sha256']==dp['checkpoint_sha256']
    assert dr['protocol']['eta_obs']==0 and dp['protocol']['eta_obs']==1
    assert all(e['valid_physics'] for d in [dr,dp] for e in d['episodes'])
    assert np.array_equal(ref['names'],plan['names']) and np.array_equal(ref['boxes'],plan['boxes'])
    assert np.array_equal(ref['heightfield_paths'],plan['heightfield_paths'])
    assert list(ref['joint_names'])==JOINT_NAMES
    heightfield.initialize(ref['heightfield_paths'].tolist())
    names=ref['names'].tolist();lookup={n:i for i,n in enumerate(names)}
    byname=[{e['clip']:e for e in d['episodes']} for d in [dr,dp]]
    selected=[]
    for category in (args.only or CATEGORIES):
        candidates=[i for i,n in enumerate(names) if n.split('/')[0]==category]
        if category=='flat':candidates=[i for i in candidates if not np.any(ref['boxes'][i,:,5]>0)]
        candidates.sort(key=lambda i:(int(ref['clip_len'][i]),names[i]))
        i=candidates[len(candidates)//2]
        selected.append({'clip':names[i],'category':category,'label':CATEGORIES[category],
                         'selection':'category median reference duration; flat excludes obstacles'})
    if not args.only:
        chosen={r['clip'] for r in selected}
        failures=sorted((e for e in dp['episodes'] if e['failed'] and byname[0][e['clip']]['completed'] and e['clip'] not in chosen),
                        key=lambda e:(e['reference_progress'],e['clip']))
        seen=set()
        for e in failures:
            if e['category'] in seen:continue
            selected.append({'clip':e['clip'],'category':e['category'],'label':CATEGORIES[e['category']]+'（失败补充）',
                             'selection':'additional planner-only failure; earliest progress, distinct categories'})
            seen.add(e['category'])
            if len(seen)==2:break
    (out/'selection.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False)+'\n')
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title=ImageFont.truetype(font,25);small=ImageFont.truetype(font,18);medium=ImageFont.truetype(font,21)
    W,H,FPS=960,720,25;nq=7+len(JOINT_NAMES)
    index=[];offset=0.
    for ordinal,selection in enumerate(selected):
        name=selection['clip'];c=lookup[name];category=selection['category'];episodes=[r[name] for r in byname]
        ends=[e['steps'] for e in episodes];last=max(ends);reference_steps=int(ref['clip_len'][c])-1
        times=[min(2*j,last) for j in range(math.ceil(last/2)+1)]
        model=mujoco.MjModel.from_xml_string(scene_xml(ref['boxes'][c]))
        model.vis.global_.offwidth,model.vis.global_.offheight=W,H
        model.vis.headlight.ambient[:]=.45;model.vis.headlight.diffuse[:]=.6
        model.geom_size[model.geom('floor').id]=(20.,20.,.05)
        grid=model.mat('grid').id;model.mat_rgba[grid,:3]=.55;model.mat_specular[grid]=0.;model.mat_reflectance[grid]=0.
        data=mujoco.MjData(model);renderer=mujoco.Renderer(model,H,W);option=mujoco.MjvOption();option.geomgroup[3:]=0
        camera=mujoco.MjvCamera();camera.elevation=-22;start=int(ref['clip_start'][c])
        R=ref['ref_root_R'][start];camera.azimuth=np.degrees(np.arctan2(R[1,0],R[0,0]))+120
        dest=out/f'{ordinal:02d}_{category}.mp4'
        encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W*2}x{H}',
                                  '-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',
                                  '-crf','20','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
        try:
            for frame,t in enumerate(times):
                states=[]
                for d,end in zip([ref,plan],ends):
                    if t==0:states.append((d['root_p'][0,c],d['root_quat'][0,c],d['q'][0,c]))
                    else:
                        j=min(t,end)-1;states.append((d['post_root_p'][j,c],d['post_root_quat'][j,c],d['post_q'][j,c]))
                fr=start+min(t,reference_steps);rp=ref['ref_root_p'][fr]
                positions=np.stack([rp,states[0][0],states[1][0]])
                camera.lookat[:]=positions.mean(0)+np.array([0.,0.,-.12])
                spread=np.max(np.linalg.norm(positions[:,:2]-camera.lookat[:2],axis=1))
                camera.distance=max(3.6,3.+2.*spread)
                panels=[]
                for side,(d,e,state,end) in enumerate(zip([ref,plan],episodes,states,ends)):
                    data.qpos[:3],data.qpos[3:7],data.qpos[7:nq]=state
                    data.qpos[nq:nq+3]=rp;data.qpos[nq+3:nq+7]=mat_to_quat(ref['ref_root_R'][fr]);data.qpos[nq+7:]=ref['ref_q'][fr]
                    mujoco.mj_forward(model,data);renderer.update_scene(data,camera,scene_option=option)
                    scene=renderer.scene;command_t=min(t,end-1)
                    for b in np.flatnonzero(d['tgt_valid'][command_t,c]):
                        if scene.ngeom>=scene.maxgeom:break
                        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.025,0,0]),
                                           d['tgt_w'][command_t,c,b].astype(float),np.eye(3).ravel(),np.array(TARGET_RGBA[b],dtype=np.float32))
                        scene.ngeom+=1
                    im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                    draw.rectangle((0,0,W,112),fill=(19,26,38))
                    mode='参考目标驱动 · η=0' if side==0 else 'Planner 目标驱动 · η=1'
                    draw.text((16,8),mode+'  |  第 14500 次权重',font=title,fill='white')
                    draw.text((16,43),selection['label']+'  |  '+name.split('/')[-1],font=small,fill=(200,211,228))
                    if t>=end:status='失败终止 · 后续定格' if e['failed'] else '运行至参考末尾'
                    else:status='正在执行'
                    color=(255,153,125) if t>=end and e['failed'] else (132,232,167)
                    draw.text((16,77),f'时间 {t/50:.2f} / {reference_steps/50:.2f} 秒  |  {status}',font=medium,fill=color)
                    draw.rectangle((0,H-88,W,H),fill=(19,26,38))
                    draw.text((16,H-81),'实体：物理闭环输出   蓝色半透明：参考动作',font=small,fill='white')
                    draw.text((16,H-54),'红/蓝点：本组左右脚目标   |   同轨迹、同视角、原速',font=small,fill='white')
                    draw.text((16,H-28),'训练参考集 · 无噪声 · 根目标仍来自参考 · 完成不代表落点精确',font=small,fill=(181,194,213))
                    panels.append(im)
                combined=Image.new('RGB',(W*2,H));combined.paste(panels[0],(0,0));combined.paste(panels[1],(W,0))
                ImageDraw.Draw(combined).line((W,0,W,H),fill=(240,240,240),width=2)
                encoder.stdin.write(combined.tobytes())
                if frame==len(times)//2:combined.save(dest.with_suffix('.jpg'))
            combined.save(out/f'{dest.stem}_end.jpg')
            for _ in range(FPS):encoder.stdin.write(combined.tobytes())
        finally:
            encoder.stdin.close();code=encoder.wait();renderer.close()
        if code:raise RuntimeError(f'ffmpeg exit {code}')
        item={**selection,'file':dest.name,'reference_completed':episodes[0]['completed'],'planner_completed':episodes[1]['completed'],
              'reference_steps':reference_steps,'reference_driver_steps':ends[0],'planner_driver_steps':ends[1],
              'frames':len(times)+FPS,'duration_s':(len(times)+FPS)/FPS,'reel_start_s':offset}
        index.append(item);offset+=item['duration_s'];print(json.dumps(item,ensure_ascii=False),flush=True)
        (out/'progress.json').write_text(json.dumps({'completed_clips':len(index),'total_clips':len(selected)},indent=2)+'\n')
    listing=out/'concat.txt';listing.write_text(''.join(f"file '{r['file']}'\n" for r in index))
    reel=out/'roban_stage2_iter014500_closed_loop_comparison.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(reel)],check=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report={'checkpoint':dr['checkpoint'],'checkpoint_sha256':dr['checkpoint_sha256'],'reference_trajectory_sha256':sha(src/'nominal_reference_seed0.npz'),
            'planner_trajectory_sha256':sha(src/'nominal_planner_seed0.npz'),'resolution':[W*2,H],'fps':FPS,'duration_s':offset,
            'exact_recorded_terminal_states':True,'actual_closed_loop_simulation':True,'same_camera_per_frame':True,
            'noise':[],'seed':0,'contact_markers':'commanded terrain targets, not contact forces','clips':index,'reel':str(reel)}
    (out/'index.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print('WROTE',reel,flush=True)


if __name__=='__main__':main()
