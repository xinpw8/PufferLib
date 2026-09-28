from pathlib import Path
import collections,json,statistics
R=Path('/home/spark-advantage/rek-training/rek-playback-speed-20260928-r2');run=R/'app60-run-r2'
pending={};steps=[];frames=[]
def distribution(a):
 a=sorted(a);return dict(n=len(a),mean=statistics.mean(a),median=statistics.median(a),p90=a[int((len(a)-1)*.9)],p99=a[int((len(a)-1)*.99)],minimum=a[0],maximum=a[-1])
for line in (run/'session.jsonl').open():
 r=json.loads(line);t=int(r['monotonicNs'])/1e6
 if r['kind']=='worker_request':pending[(r['worker'],r['op'])]=t
 if r['kind']=='worker_reply' and r['op']=='step':steps.append(dict(start=pending[(r['worker'],r['op'])],end=t,rpc=r['durationMs']))
 if r['kind']=='rendered_frame':frames.append(t)
s=steps[1:]
gaps=[steps[i]['start']-steps[i-1]['end'] for i in range(1,len(steps))]
raw=[v['rpc'] for v in s];capped=[max(20,v) for v in raw]
result=dict(steady_excludes_first_cold_tick=True,rpc_ms=distribution(raw),reply_to_next_request_ms=distribution(gaps),rpc_above20ms=sum(v>20 for v in raw),rpc_at_or_below20ms=sum(v<=20 for v in raw),mean_positive_rpc_overrun_ms=statistics.mean(max(0,v-20) for v in raw),idealized_phase_reset_cost_ms=statistics.mean(capped),idealized_phase_reset_ratio=20/statistics.mean(capped),actual_mean_start_interval_ms=statistics.mean(steps[i]['start']-steps[i-1]['start'] for i in range(1,len(steps))),recorded_frame_interval_ms=distribution([b-a for a,b in zip(frames,frames[1:])]),caveat='max(20,RPC) is a lower-bound timing model for r6 per-tick phase reset. Measured gap includes trace/report work and scheduling. No separate CPU profiler was inserted.')
(R/'app60-analysis-r2'/'SCHEDULER.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
