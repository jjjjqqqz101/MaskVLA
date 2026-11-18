"""Autonomous policy rollouts. Expert methods are never invoked here."""
import argparse,base64,json,time,urllib.request
import numpy as np
from .engine import PilotEngine
from .recorder import DATA,OUTPUT,write_json
from .selection import OBJECT_NAMES,selection_requests
from .protocol import ACTION_CONTRACT

def infer(endpoint,obs):
    body={k:base64.b64encode(np.ascontiguousarray(obs[k]).tobytes()).decode() for k in ('base_rgb','wrist_rgb')}
    body['target_mask']=base64.b64encode(np.ascontiguousarray(obs['gt_target_mask_offline'],dtype=np.uint8).tobytes()).decode()
    body['robot_state']=obs['robot_state'].tolist()
    req=urllib.request.Request(endpoint+'/infer',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r:response=json.load(r)
    actions=np.asarray(response['actions'],np.float32)
    if actions.ndim!=2 or actions.shape[1]!=7 or not np.isfinite(actions).all():raise ValueError('Invalid learned-policy output')
    return np.clip(actions,-1,1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',choices=('act','pi05'),required=True)
    ap.add_argument('--endpoint');ap.add_argument('--objects',nargs='+',choices=OBJECT_NAMES,default=list(OBJECT_NAMES))
    ap.add_argument('--trials',type=int,default=2);ap.add_argument('--seed-base',type=int,default=910000);ap.add_argument('--max-steps',type=int,default=600)
    ap.add_argument('--execute',type=int,default=1);ap.add_argument('--tag',default='eval');args=ap.parse_args()
    endpoint=args.endpoint or ('http://127.0.0.1:18861' if args.policy=='act' else 'http://127.0.0.1:18862')
    if endpoint not in ('http://127.0.0.1:18861','http://127.0.0.1:18862'):raise ValueError('Only pilot loopback model endpoints allowed')
    with urllib.request.urlopen(endpoint+'/health',timeout=30) as r:health=json.load(r)
    horizon=int(health.get('horizon',health.get('action_horizon',0)))
    if not 1<=args.execute<=horizon:raise ValueError('execute must be between one and the policy horizon')
    if args.max_steps<1 or args.trials<1:raise ValueError('Positive rollout limits required')
    rows=[]
    for name in args.objects:
        i=list(OBJECT_NAMES).index(name);e=PilotEngine();e.create(selection_requests([name])[0])
        try:
            for trial in range(args.trials):
                seed=args.seed_base+i*1000+trial;obs=e.reset_trial(seed)
                eid=f'rc_policy_{args.policy}_{args.tag}_{name}_{seed}_{int(time.time())}'
                e.start_pilot_record(eid,dict(object_name=name,split='evaluation',seed=seed,source='learned_policy_'+args.policy,
                    policy=args.policy,policy_health=health,execute_steps=args.execute,max_steps=args.max_steps,seed_base=args.seed_base,mask_source='sim_gt_oracle_for_policy_evaluation',
                    no_expert_or_planner=True,scene_family=f'{name}:{seed}'))
                t=time.time();error=None;queries=0
                try:
                    while e.frame_id<args.max_steps:
                        start=time.time();actions=infer(endpoint,e.observation());queries+=1
                        e.stage='learned_policy_full_pick_place'
                        for action in actions[:args.execute]:
                            grip=1. if action[6]>0 else -1.
                            native,arm=e.native_action(action[:3]*.05,action[3:6]*.5,grip)
                            e.gripper=grip;e.step(native,arm)
                            if e.scoring()['simulator_success'] or e.frame_id>=args.max_steps:break
                        if e.scoring()['simulator_success']:break
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                summary=e.finish_pilot_record(error)
                row=dict(summary,policy=args.policy,object=name,trial=trial,seed=seed,queries=queries,execute_steps=args.execute,max_steps=args.max_steps,seed_base=args.seed_base,
                    seconds=time.time()-t,no_expert_or_planner=True,mask_source='sim_gt_oracle_for_policy_evaluation')
                rows.append(row);write_json(OUTPUT/f'{args.policy}_{args.tag}_results.json',dict(health=health,rollouts=rows,
                    success_count=sum(x['simulator_success'] for x in rows),total=len(rows)))
                print(json.dumps(row),flush=True)
        finally:e.close()
if __name__=='__main__':main()
