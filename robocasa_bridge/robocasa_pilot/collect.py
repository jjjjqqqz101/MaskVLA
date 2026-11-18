"""Resumable quota-based full physical pick-place collection, scoped pilot_v1."""
import argparse,hashlib,json,time,traceback,fcntl
from pathlib import Path
from .engine import PilotEngine
from .recorder import DATA,OUTPUT,write_json
from .protocol import PROJECT, protocol_dict
from .selection import OBJECT_NAMES,selection_requests,resolved_selection

def refresh_manifest():
    DATA.mkdir(parents=True,exist_ok=True)
    with (DATA/'.manifest.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _refresh_manifest_locked()

def _refresh_manifest_locked():
    episodes=[];splits={k:[] for k in ('train','validation','test')}
    for d in sorted((DATA/'episodes').glob('rc_*')):
        summary=d/'summary.json';meta=d/'metadata.json'
        if not summary.exists() or not meta.exists():continue
        s=json.loads(summary.read_text());m=json.loads(meta.read_text())
        if not s.get('training_ready'):continue
        raw=str(d/'trajectory.h5');split=m['split']
        if split not in splits:continue
        row=dict(path=raw,split=split,object_id=m['object_name'],seed=m['seed'],episode_id=d.name,
            steps=s['steps'],scene_family=f"{m['object_name']}:{m['seed']}",initial_rgb_sha256=m['initial_rgb_sha256'])
        episodes.append(row);splits[split].append(raw)
    manifest=dict(schema='robocasa.pilot.dataset.v1',protocol=protocol_dict(),episodes=episodes,splits=splits,
        note='single-target no-distractor initial pilot; validation/test seeds disjoint; no old data modified',
        updated_at=time.time())
    write_json(DATA/'manifest.json',manifest)
    return manifest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--objects',nargs='+',default=list(OBJECT_NAMES),choices=OBJECT_NAMES)
    ap.add_argument('--train',type=int,default=10);ap.add_argument('--validation',type=int,default=2);ap.add_argument('--test',type=int,default=2)
    ap.add_argument('--max-attempts',type=int,default=12);args=ap.parse_args()
    frozen=dict(protocol=protocol_dict(),objects=resolved_selection(),scope=str(PROJECT),
        initial_pilot=dict(distractor_count=0,xy_range_m=.035,yaw_range_rad=.65,tray_counter_offset_m=.14),
        qc=dict(initial_target_pixels_min=20,lift_min_m=.06,bilateral_grasp_required=True,
            stable_placement_steps=10,max_linear_velocity_m_s=.035,release_gripper=-1,real_terminal_hold_steps=20))
    fp=DATA/'frozen_protocol.json'
    if fp.exists():
        if json.loads(fp.read_text())!=frozen:raise ValueError('Frozen pilot protocol differs')
    else:write_json(fp,frozen)
    manifest=refresh_manifest()
    for name in args.objects:
        index=list(OBJECT_NAMES).index(name);e=PilotEngine();e.create(selection_requests([name])[0])
        try:
            for si,(split,quota) in enumerate((('train',args.train),('validation',args.validation),('test',args.test))):
                existing=[x for x in manifest['episodes'] if x['object_id']==name and x['split']==split]
                while len(existing)<quota:
                    n=len(existing);done=False
                    for attempt in range(args.max_attempts):
                        seed=710000+index*10000+si*2000+n*20+attempt
                        eid=f'rc_pilot_{name}_{split}_{seed}'
                        if (DATA/'episodes'/eid).exists():continue
                        t=time.time();error=None
                        try:
                            obs=e.reset_trial(seed)
                            if int(obs['gt_target_mask_offline'].sum())<20:raise RuntimeError('Initial target too occluded')
                            e.start_pilot_record(eid,dict(object_name=name,split=split,seed=seed,candidate=attempt%4,
                                scene_family=f'{name}:{seed}',pilot_protocol_file=str(fp)))
                            e.expert(attempt%4)
                        except Exception as exc:
                            error=f'{type(exc).__name__}: {exc}'
                        summary=e.finish_pilot_record(error) if e.record else dict(training_ready=False,error=error,steps=e.frame_id)
                        row=dict(summary,object=name,split=split,quota_index=n,attempt=attempt,seed=seed,eid=eid,
                            seconds=time.time()-t)
                        with (OUTPUT/'attempts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                        print(json.dumps(row),flush=True)
                        if summary['training_ready']:
                            manifest=refresh_manifest();existing=[x for x in manifest['episodes'] if x['object_id']==name and x['split']==split]
                            done=True;break
                    if not done:raise RuntimeError(f'Could not collect {name} {split} quota index {n}; inspect attempts')
        finally:e.close()
    manifest=refresh_manifest();print(json.dumps(dict(status='complete',counts={k:len(v) for k,v in manifest['splits'].items()})),flush=True)
if __name__=='__main__':main()
