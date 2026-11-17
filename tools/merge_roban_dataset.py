"""Atomically extend the RoBan release with validated, balanced new categories."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reports',nargs='+',type=Path);ap.add_argument('--out',type=Path,default=Path('data/roban_motion'))
    ap.add_argument('--per-category',type=int,default=10);ap.add_argument('--direction-run',default='directions_03')
    ap.add_argument('--categories',nargs='+');ap.add_argument('--replace-categories',action='store_true')
    args=ap.parse_args();dest=args.out.resolve();contacts=Path(str(dest)+'_contacts')
    old=json.loads((dest/'manifest.json').read_text());previous=json.loads((dest/'release.json').read_text())
    baseline=len(old)-previous.get('added',0)
    if args.replace_categories:
        if not args.categories:raise ValueError('Explicit categories required for replacement')
        old=[x for x in old if x['clip'].split('/')[0] not in args.categories]
    old_categories={x['clip'].split('/')[0] for x in old}
    groups=defaultdict(lambda:defaultdict(list));seen=set()
    for report_path in args.reports:
        report=json.loads(report_path.read_text());root=Path(report['motion_dir']).resolve()
        exports={r['clip']:r for r in json.loads((root/'export_manifest.json').read_text()) if r['accepted']}
        for quality in report['clips']:
            if not quality['accepted']:continue
            name=quality['clip'];category=name.split('/')[0]
            if name not in exports:continue
            e=exports[name]
            if category=='random_rough' and e.get('difficulty',0)<=0:continue
            if category in old_categories or name in seen:continue
            if category.startswith('flat_') and Path(e['episode']).parent.parent.name!=args.direction_run:continue
            seen.add(name)
            paths=[root/(name+s) for s in ('_motion.npz','_terrain.npy')]
            keys=['motion_sha256','terrain_sha256']
            if 'heightfield_sha256' in e:paths.append(root/(name+'_heightfield.npz'));keys.append('heightfield_sha256')
            for path,key in zip(paths,keys):
                if sha(path)!=e[key]:raise ValueError(f'Changed after export: {path}')
            label=Path(str(root)+'_contacts')/(name+'.npz')
            if any(p.stat().st_mtime>report_path.stat().st_mtime for p in paths+[label]):raise ValueError('Changed after validation')
            row=dict(e,quality=quality,contacts_sha256=sha(label),validation_report=str(report_path.resolve()))
            groups[category][e['split_group']].append((row,paths,label))
    expected={'flat_backward','flat_left','flat_right','flat_turn_left','flat_turn_right','hf_pyramid_slope','hf_pyramid_slope_inv',
              'hf_pyramid_slope_narrow','hf_pyramid_slope_inv_narrow','random_rough','threshold','narrow_beams','random_spread_boxes'}
    if args.categories:expected=set(args.categories)
    available={c:sum(map(len,groups[c].values())) for c in sorted(expected)}
    if any(n<args.per_category for n in available.values()):raise ValueError(f'Insufficient verified clips: {available}')
    selected=[]
    for category in sorted(expected):
        queues=[sorted(items,key=lambda e:e[0]['clip']) for _,items in sorted(groups[category].items())]
        picked=[];used_episodes=set();deferred=[]
        while any(queues):
            for queue in queues:
                if queue:
                    item=queue.pop(0)
                    if item[0]['episode'] in used_episodes:deferred.append(item)
                    else:picked.append(item);used_episodes.add(item[0]['episode'])
        selected.extend((picked+deferred)[:args.per_category])
    pending=Path(tempfile.mkdtemp(prefix='roban_release_',dir=dest.parent.parent/'artifacts/roban'))
    dm,dc=pending/'motion',pending/'contacts';shutil.copytree(dest,dm);shutil.copytree(contacts,dc)
    if args.replace_categories:
        for category in expected:
            for base in (dm,dc):
                if (base/category).exists():shutil.rmtree(base/category)
    manifest=list(old)
    for row,paths,label in selected:
        name=row['clip'];(dm/name).parent.mkdir(parents=True,exist_ok=True);(dc/name).parent.mkdir(parents=True,exist_ok=True)
        for path in paths:shutil.copy2(path,dm/name.split('/')[0]/path.name)
        shutil.copy2(label,dc/(name+'.npz'))
        source=json.loads((Path(row['episode'])/'episode.json').read_text())
        row.update(status='geometry_contact_validated_reference',split='not_formal_holdout',
                   measured_body_velocity=source.get('measured_body_velocity'),heading_change_rad=source.get('heading_change_rad'))
        manifest.append(row)
    (dm/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary=dict(release='roban_v2',clips=len(manifest),added=len(manifest)-baseline,per_category=dict(Counter(x['clip'].split('/')[0] for x in manifest)),
        frames=sum(x['frames'] for x in manifest),seconds=sum((x['frames']-1)/50 for x in manifest),
        split_groups=len({x['split_group'] for x in manifest}),available_new=available,source_policy_references=True,
        trained_acp_planner_rollouts=False,terrain_schemas=['yaw_boxes_v1','heightfield_v1'],internal_terrain_fields=9)
    (dm/'release.json').write_text(json.dumps(summary,indent=2)+'\n')
    (dm/'dataset_stats.json').write_text(json.dumps(summary,indent=2)+'\n')
    history=dest.parent.parent/'artifacts/roban/releases';history.mkdir(exist_ok=True)
    for name in ('manifest.json','release.json','dataset_stats.json'):
        if not (history/('v1_'+name)).exists():shutil.copy2(dest/name,history/('v1_'+name))
    backup=pending/'old_motion';backup_contacts=pending/'old_contacts'
    dest.rename(backup);contacts.rename(backup_contacts)
    try:
        dm.rename(dest);dc.rename(contacts)
    except Exception:
        if dest.exists():shutil.rmtree(dest)
        if contacts.exists():shutil.rmtree(contacts)
        backup.rename(dest);backup_contacts.rename(contacts);raise
    shutil.rmtree(pending)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
