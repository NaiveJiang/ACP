"""Render fixed examples and build categorized reels from comprehensive recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path);ap.add_argument('--skip-unready',action='store_true');args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.directory.resolve();meta=json.loads((out/'selection.json').read_text());selection=json.loads((out/'video_selection.json').read_text());base=Path(meta['video_dir']);base.mkdir(exist_ok=True,parents=True);videos=[];pending=[]
    for item in selection['clips']:
        src=out/('holdout_randomized_seed0' if item.get('split')=='holdout' else 'train_randomized_seed0')
        if not (src/'results.json').exists():
            if args.skip_unready:pending.append(item['id']);continue
            raise FileNotFoundError(src/'results.json')
        dest=base/item['id'];dest.mkdir(exist_ok=True)
        if not (dest/'index.json').exists():
            with (dest/'render.log').open('w') as log:
                subprocess.run([sys.executable,'tools/render_roban_box_suffix.py',str(src),'--out',str(dest),'--clip',item['clip'],
                    '--phases','0','--mode','training_randomization','--speed','1','--label',item['label'],'--contact-intent','--follow-camera'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        index=json.loads((dest/'index.json').read_text());assert index['checkpoint_sha256']==meta['checkpoint_sha256']
        index['selection']=item.get('selection',selection['selection']);(dest/'index.json').write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n')
        video=dest/index['videos'][0]['file'];verify=dest/'verification.json'
        if not verify.exists():
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate:format=duration','-of','json',str(video)]))
            assert int(probe['streams'][0]['nb_read_frames'])==index['videos'][0]['frames']
            subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
            verify.write_text(json.dumps(dict(ffprobe=probe,full_decode_passed=True,sha256=hashlib.sha256(video.read_bytes()).hexdigest()),indent=2)+'\n')
        e=index['episodes'][0];videos.append(dict(**item,video=str(video),completed=e['completed'],valid_physics=e['valid_physics'],elapsed_s=e['elapsed_s'],duration_s=index['videos'][0]['duration_s']))
        print('VIDEO',item['id'],e['completed'],flush=True)
    groups=dict(flat_moves=['flat','flat_backward','flat_left','flat_right','flat_turn_left','flat_turn_right'],stairs=['pyramid_stairs','pyramid_stairs_narrow','pyramid_inv_stairs','pyramid_inv_stairs_narrow','variable_height_stairs_up','variable_height_stairs_down'],slopes=['hf_pyramid_slope','hf_pyramid_slope_narrow','hf_pyramid_slope_inv','hf_pyramid_slope_inv_narrow'],obstacles=['narrow_beams','random_rough','random_spread_boxes','threshold'],boxes=['box_climb','holdout_box'],failures=[v['id'] for v in videos if not v['completed']])
    reels={};byid={v['id']:v for v in videos}
    for group,ids in groups.items():
        chosen=[byid[i] for i in ids if i in byid]
        if not chosen:continue
        listing=base/(group+'_concat.txt');listing.write_text(''.join(f"file '{v['video']}'\n" for v in chosen));dest=base/(group+'.mp4')
        subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(dest)],check=True)
        subprocess.run(['ffmpeg','-v','error','-i',str(dest),'-f','null','-'],check=True)
        cursor=0.;chapters=[]
        for v in chosen:chapters.append(dict(id=v['id'],label=v['label'],start_s=cursor,duration_s=v['duration_s'],completed=v['completed']));cursor+=v['duration_s']
        reels[group]=dict(video=str(dest),duration_s=cursor,chapters=chapters,full_decode_passed=True)
    result=dict(checkpoint=meta['checkpoint'],checkpoint_sha256=meta['checkpoint_sha256'],selection=selection['selection'],videos=videos,reels=reels,pending=pending)
    (base/'index.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print('DONE',len(videos),'pending',pending,flush=True)


if __name__=='__main__':main()
