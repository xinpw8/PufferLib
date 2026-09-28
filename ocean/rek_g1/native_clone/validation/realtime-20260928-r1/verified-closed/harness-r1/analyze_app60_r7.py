from pathlib import Path
import collections,hashlib,json,math,statistics
ROOT=Path('/home/spark-advantage/rek-training/rek-playback-speed-20260928-r2')
run=ROOT/'app60-run-r7';out=ROOT/'app60-result-r7';dest=ROOT/'app60-analysis-r7'
dest.mkdir(exist_ok=False)
summary=json.loads((out/'summary.json').read_text());samples=json.loads((out/'samples.json').read_text());inputs=json.loads((out/'input-requests.json').read_text());images=json.loads((out/'image-receipts.json').read_text())
def dist(v):
    a=sorted(v)
    return dict(n=len(a),mean=statistics.mean(a),median=statistics.median(a),p90=a[int((len(a)-1)*.9)],p99=a[int((len(a)-1)*.99)],min=a[0],max=a[-1]) if a else None
events=collections.Counter();steps=[];renders=[];native=collections.Counter();frames=[];workers={};ticks=[];qpos_good=0;closed=[]
with (run/'session.jsonl').open() as f:
    for line in f:
        r=json.loads(line);events[r['kind']]+=1
        if r['kind']=='worker_created':workers[r['worker']]={'role':r['role'],'pid':r['pid']}
        if r['kind']=='worker_reply' and r['op']=='step':
            steps.append((int(r['monotonicNs']),r['durationMs']));state=r['result']['state'];ticks.append(state['tick'])
            assert len(state['qpos'])==72 and len(state['qvel'])==70 and all(math.isfinite(x) for x in state['qpos']+state['qvel']);qpos_good+=1
            for e in r['result'].get('commandEvents',[]):
                if e.get('attempted'):native['attempted']+=1
                if e.get('accepted'):native['accepted']+=1
                if e.get('rejected'):native['rejected']+=1;native['rejected_reason_'+str(e.get('reason'))]+=1
                if e.get('cancelled'):native['cancelled']+=1
        if r['kind']=='rendered_frame':
            p=run/r['path'];data=p.read_bytes();assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'];frames.append(r);renders.append(r['durationMs'])
        if r['kind']=='worker_exit':closed.append(r)
intervals=[(steps[i][0]-steps[i-1][0])/1e6 for i in range(1,len(steps))]
overhead=[intervals[i-1]-steps[i][1] for i in range(1,len(steps))]
windows=[];base=samples[0]['monotonic_ns']
boundaries=[samples[0]]
for sec in [10,20,30,40,50]:boundaries.append(min(samples,key=lambda s:abs((s['monotonic_ns']-base)/1e9-sec)))
boundaries.append({'monotonic_ns':inputs[0]['sent_monotonic_ns']+int(summary['elapsed_seconds']*1e9),'state':summary['final']})
for a,b in zip(boundaries,boundaries[1:]):
    x=a['state']['pace'];y=b['state']['pace'];wall=y['activeWallMs']-x['activeWallMs'];n=y['activeIntervals']-x['activeIntervals']
    windows.append(dict(start_seconds=(a['monotonic_ns']-base)/1e9,end_seconds=(b['monotonic_ns']-base)/1e9,intervals=n,active_wall_ms=wall,ratio=20*n/wall))
result=dict(success=summary['success'],whole_ratio=summary['pace']['realTimeRatio'],steady_after5s=summary['steady_after5s'],windows=windows,step_rpc_ms=dist([s[1] for s in steps]),step_reply_interval_ms=dist(intervals),reply_interval_minus_current_rpc_ms=dist(overhead),render_ms=dist(renders),event_counts=events,native_command_events=native,ticks_contiguous=ticks==list(range(1,len(ticks)+1)),validated_full_states=qpos_good,frames_verified=len(frames),frame_bytes=sum(f['bytes'] for f in frames),frame_tick_range=[frames[0]['tick'],frames[-1]['tick']],input_held_counts=collections.Counter('+'.join(i['payload']['held']) for i in inputs),move_categories=collections.Counter(i['payload']['move'] for i in inputs if i['payload']['move'] is not None),worker_identities=workers,worker_exit_events=closed,guard_failure=summary['guard_failure'],final={k:summary['final'].get(k) for k in ['tick','score','falls','wins','round','phase','fightResult','fightWinner','ties','redos','unclassified']},health=summary['closed_recorder_health'],caveat='Reply interval minus current RPC includes timer scheduling and JavaScript recording/reporting; it is not solely sleep time.')
result.update(outer_wall_ratio=summary['final_tick']*.02/summary['elapsed_seconds'],outer_elapsed_seconds=summary['elapsed_seconds'],scheduler=summary['final']['pace'].get('scheduler'))
(dest/'ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
