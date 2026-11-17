"""Render synchronized, recorded pre-instability/latest Stage 2 rollouts."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
os.environ.setdefault('ACP_ROBOT', 'roban')
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from supermimic.acp import heightfield
from supermimic.acp.render import scene_xml, TARGET_RGBA
from supermimic.acp.robot import JOINT_NAMES
from render_roban_closed_loop_comparison import CATEGORIES

CATEGORIES = {**CATEGORIES, 'box_climb': '翻箱'}


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def selections(src):
    m = json.loads((src / 'manifest.json').read_text()); result = []
    for split in ('train', 'holdout'):
        records = json.loads((Path(m['datasets'][split]['directory']) / 'manifest.json').read_text())
        order = list(CATEGORIES) if split == 'train' else ['box_climb']
        for category in order:
            candidates = [r for r in records if r['clip'].split('/')[0] == category
                          and (category != 'flat' or r['box_count'] == 0)]
            candidates.sort(key=lambda r: (r['frames'], r['clip']))
            item = candidates[len(candidates) // 2]
            result.append(dict(clip=item['clip'], category=category, split=split, mode='nominal',
                               label=CATEGORIES[category] + ('（独立 holdout）' if split == 'holdout' else ''),
                               selection='Median reference duration within category, independent of outcome; flat excludes obstacles.'))
    for item in list(result):
        if item['category'] in ('flat_backward', 'pyramid_stairs_narrow', 'hf_pyramid_slope_narrow', 'box_climb'):
            result.append({**item, 'mode': 'training_randomization', 'label': item['label'] + ' · 随机扰动'})
    # Explicitly labeled supplements show genuine regressions, without selecting them as representative rates.
    for split in ('train', 'holdout'):
        pre = json.loads((src / f'pre_{split}_nominal_planner_seed0/results.json').read_text())
        after = json.loads((src / f'latest_{split}_nominal_planner_seed0/results.json').read_text())
        latest = {e['clip']: e for e in after['episodes']}
        already = {x['clip'] for x in result if x['mode'] == 'nominal' and x['split'] == split}
        candidates = [e for e in pre['episodes'] if e['valid_physics'] and latest[e['clip']]['valid_physics']
                      and e['completed'] and not latest[e['clip']]['completed'] and e['clip'] not in already]
        candidates.sort(key=lambda e: (0 if e['clip'].startswith('box_climb/') else 1,
                                       latest[e['clip']]['terminal_phase'], e['clip']))
        seen = set()
        for e in candidates:
            cat = e['clip'].split('/')[0]
            if cat in seen:
                continue
            result.append(dict(clip=e['clip'], category=cat, split=split, mode='nominal',
                               label=CATEGORIES[cat] + (' holdout' if split == 'holdout' else '') + ' · 退化补充',
                               selection='Supplement: pre model completes, latest fails; prioritize boxes then earliest failure, distinct categories.'))
            seen.add(cat)
            if len(seen) == (2 if split == 'train' else 1):
                break
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('directory', type=Path)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--limit', type=int)
    ap.add_argument('--skip-unready', action='store_true')
    args = ap.parse_args(); src = args.directory.resolve(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / 'manifest.json').read_text()); chosen = selections(src)
    if args.limit:
        chosen = chosen[:args.limit]
    (out / 'selection.json').write_text(json.dumps(chosen, indent=2, ensure_ascii=False) + '\n')
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title, text, small = [ImageFont.truetype(font, n) for n in (25, 21, 18)]
    W, H, FPS, nq = 960, 720, 25, 7 + len(JOINT_NAMES)
    cached = {}; index = []; pending = []
    render_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for ordinal, selection in enumerate(chosen):
        mode, split, name = selection['mode'], selection['split'], selection['clip']
        dest = out / f'{ordinal:02d}_{split}_{selection["category"]}_{mode}.mp4'
        metadata = dest.with_suffix('.json')
        if metadata.exists() and dest.exists():
            saved = json.loads(metadata.read_text())
            if saved.get('render_source_sha256') == render_sha and saved['clip'] == name:
                index.append(saved)
                continue
        if args.skip_unready and any(not (src / f'{key}_{split}_{mode}_planner_seed0' / 'results.json').exists() for key in ('pre', 'latest')):
            pending.append(selection)
            continue
        pair = []
        for model_key in ('pre', 'latest'):
            case = f'{model_key}_{split}_{mode}_planner_seed0'
            if case not in cached:
                cached[case] = (load(src / case / (mode + '.npz')),
                                json.loads((src / case / 'results.json').read_text()))
            d, report = cached[case]; c = d['names'].tolist().index(name)
            e = next(e for e in report['episodes'] if e['clip'] == name)
            assert e['valid_physics'], ('Cannot render an invalid physics run as a policy outcome', case, name)
            assert report['eta_obs'] == 1 and report['checkpoint_sha256'] == meta['models'][model_key]['sha256']
            assert d['eta_obs'].item() == 1 and list(d['joint_names']) == JOINT_NAMES
            steps = int(d['alive'][:, c].sum()); start = int(d['start_frames'][c])
            assert steps == round(e['elapsed_s'] / .02)
            assert np.array_equal(d['post_f'][:steps, c], np.arange(start + 1, start + steps + 1))
            pair.append(dict(data=d, report=report, c=c, episode=e, steps=steps, start=start, model_key=model_key))
        a, b = pair
        assert a['report']['initial_condition_sha256'] == b['report']['initial_condition_sha256']
        for key in ('boxes', 'initial_root_p', 'initial_root_quat', 'initial_q'):
            assert np.array_equal(a['data'][key][a['c']], b['data'][key][b['c']]), key
        heightfield.initialize(a['data']['heightfield_paths'].tolist())
        model = mujoco.MjModel.from_xml_string(scene_xml(a['data']['boxes'][a['c']]))
        model.vis.global_.offwidth, model.vis.global_.offheight = W, H
        model.vis.headlight.ambient[:] = .45; model.vis.headlight.diffuse[:] = .6
        model.geom_size[model.geom('floor').id] = (20., 20., .05)
        grid = model.mat('grid').id
        model.mat_rgba[grid, :3] = .55
        model.mat_specular[grid] = 0.; model.mat_reflectance[grid] = 0.
        data = mujoco.MjData(model); renderer = mujoco.Renderer(model, H, W)
        option = mujoco.MjvOption(); option.geomgroup[3:] = 0
        camera = mujoco.MjvCamera(); camera.elevation = -22
        d, start = a['data'], a['start']; ref_steps = int(d['clip_len'][a['c']]) - 1
        travel = d['ref_root_p'][start + ref_steps] - d['ref_root_p'][start]
        camera.azimuth = np.degrees(np.arctan2(travel[1], travel[0])) + 110
        last = max(e['steps'] for e in pair); times = [min(2*j, last) for j in range(math.ceil(last/2) + 1)]
        encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                    '-s', f'{W*2}x{H}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
                                    '-threads', '2', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-crf', '20',
                                    '-movflags', '+faststart', str(dest)], stdin=subprocess.PIPE)
        try:
            for frame, t in enumerate(times):
                states = []
                for e in pair:
                    d, c = e['data'], e['c']; at = min(t, e['steps'])
                    states.append(tuple(d[k][c] for k in ('initial_root_p', 'initial_root_quat', 'initial_q')) if at == 0
                                  else tuple(d[k][at-1, c] for k in ('post_root_p', 'post_root_quat', 'post_q')))
                ref_now = a['data']['ref_root_p'][start + min(t, ref_steps)]
                positions = np.stack([ref_now, states[0][0], states[1][0]])
                camera.lookat[:] = positions.mean(0) + np.array([0., 0., -.12])
                spread = np.max(np.linalg.norm(positions[:, :2] - camera.lookat[:2], axis=1))
                camera.distance = max(3.6, 3. + 2. * spread)
                panels = []
                for e, state in zip(pair, states):
                    d, c, ep = e['data'], e['c'], e['episode']; at = min(t, e['steps']); fr = e['start'] + at
                    rp = d['ref_root_p'][fr]
                    data.qpos[:3], data.qpos[3:7], data.qpos[7:nq] = state
                    data.qpos[nq:nq+3], data.qpos[nq+3:nq+7], data.qpos[nq+7:] = rp, d['ref_root_quat'][fr], d['ref_q'][fr]
                    mujoco.mj_forward(model, data); renderer.update_scene(data, camera, scene_option=option)
                    scene = renderer.scene; command_t = max(0, at-1)
                    for body in np.flatnonzero(d['tgt_valid'][command_t, c]):
                        if scene.ngeom >= scene.maxgeom:
                            break
                        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                           np.array([.025, 0., 0.]), d['tgt_w'][command_t, c, body].astype(float),
                                           np.eye(3).ravel(), np.array(TARGET_RGBA[body], dtype=np.float32))
                        scene.ngeom += 1
                    im = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(im)
                    draw.rectangle((0, 0, W, 125), fill=(19, 26, 38))
                    checkpoint = meta['models'][e['model_key']]['iteration']
                    label = '失稳前' if e['model_key'] == 'pre' else '最新'
                    draw.text((16, 8), f'{label} · 第 {checkpoint:,} 次  |  Planner 闭环', font=title, fill='white')
                    draw.text((16, 45), selection['label'], font=text, fill=(200, 211, 228))
                    draw.text((16, 78), name.split('/')[-1], font=small, fill=(200, 211, 228))
                    draw.rectangle((0, H-116, W, H), fill=(19, 26, 38))
                    terminal = t >= e['steps']
                    status = ('失败终止 · 保留末帧' if ep['failed'] else '运行至参考末尾') if terminal else '正在执行'
                    color = (255, 153, 125) if terminal and ep['failed'] else (132, 232, 167)
                    xy = np.linalg.norm(state[0][:2] - rp[:2]) * 100
                    draw.text((16, H-110), f'{at/50:.2f} / {ref_steps/50:.2f} 秒  |  {status}  |  XY 误差 {xy:.1f} cm', font=small, fill=color)
                    draw.text((16, H-80), '实体：真实物理输出   蓝色半透明：参考动作   彩色点：输入接触目标', font=small, fill='white')
                    condition = '无扰动' if mode == 'nominal' else '训练随机化（含 reset 噪声）· seed 0'
                    draw.text((16, H-53), condition + '   |   相同初态、同视角、原速', font=small, fill=(200, 211, 228))
                    draw.text((16, H-27), '接触目标 100% 来自 planner；根目标仍来自参考；完成不等于精准到达', font=small, fill=(180, 193, 211))
                    panels.append(im)
                combined = Image.new('RGB', (W*2, H)); combined.paste(panels[0], (0, 0)); combined.paste(panels[1], (W, 0))
                ImageDraw.Draw(combined).line((W, 0, W, H), fill='white', width=2)
                encoder.stdin.write(combined.tobytes())
                if frame == len(times)//2:
                    combined.save(dest.with_suffix('.jpg'))
            combined.save(dest.with_name(dest.stem + '_end.jpg'))
            for _ in range(FPS):
                encoder.stdin.write(combined.tobytes())
        finally:
            encoder.stdin.close(); code = encoder.wait(); renderer.close()
        if code:
            raise RuntimeError(f'ffmpeg exit {code}')
        item = {**selection, 'file': dest.name, 'frames': len(times)+FPS, 'duration_s': (len(times)+FPS)/FPS,
                'render_source_sha256': render_sha,
                'pre_completed': a['episode']['completed'], 'latest_completed': b['episode']['completed'],
                'pre_steps': a['steps'], 'latest_steps': b['steps'], 'reference_steps': ref_steps,
                'pre_terminal_xy_m': a['episode']['terminal_root_xy_error_m'],
                'latest_terminal_xy_m': b['episode']['terminal_root_xy_error_m'],
                'trajectory_sha256': {e['model_key']: hashlib.sha256((src / f'{e["model_key"]}_{split}_{mode}_planner_seed0' / (mode+'.npz')).read_bytes()).hexdigest() for e in pair}}
        index.append(item)
        metadata.write_text(json.dumps(item, indent=2, ensure_ascii=False)+'\n')
        (out / 'progress.json').write_text(json.dumps(dict(completed=len(index), total=len(chosen), latest=item), ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(item, ensure_ascii=False), flush=True)
    reels = {}
    for name, selected in [('all', index), ('nominal', [x for x in index if x['mode'] == 'nominal']),
                           ('randomized', [x for x in index if x['mode'] != 'nominal']),
                           ('boxes', [x for x in index if x['category'] == 'box_climb']),
                           ('regressions', [x for x in index if x['pre_completed'] and not x['latest_completed']])]:
        if not selected:
            continue
        listing = out / (name+'_concat.txt'); listing.write_text(''.join(f"file '{x['file']}'\n" for x in selected))
        dest = out / (name+'.mp4')
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
                        '-c', 'copy', '-movflags', '+faststart', str(dest)], check=True)
        offset = 0.; chapters = []
        for x in selected:
            chapters.append(dict(file=x['file'], label=x['label'], start_s=offset, duration_s=x['duration_s']))
            offset += x['duration_s']
        reels[name] = dict(file=dest.name, duration_s=offset, chapters=chapters)
    report = dict(models=meta['models'], clips=index, reels=reels, pending=pending, fps=FPS, resolution=[W*2, H],
                  actual_closed_loop_states=True, exact_terminal_states=True, same_camera_per_frame=True,
                  terminal_state_and_reference_frozen_after_termination=True,
                  contact_markers='Exact commanded terrain targets including command noise; not contact forces.')
    (out / 'index.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
