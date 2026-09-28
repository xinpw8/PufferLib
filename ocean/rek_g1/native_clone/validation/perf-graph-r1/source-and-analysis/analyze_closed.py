from pathlib import Path
import json,hashlib,struct,difflib,collections
root=Path(__file__).parent
result=root.parent/'remote-result/perf-graph-comparison-r1'
out=root/'closed-analysis-r1';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def replies(name):return [r['reply'] for r in map(json.loads,(result/name/'protocol.jsonl').read_text().splitlines()) if 'reply'in r and 'id'in r['reply']]
a,b=replies('eager'),replies('graph');assert len(a)==len(b)==365
equal=lambda x,y:struct.pack('<d',float(x))==struct.pack('<d',float(y))
fields=['qpos','qvel','raw','mask','score','falls','wins','completedPoints','actions','rewards']
array_stats={k:{'differing_replies':0,'differing_values':0,'max_absolute_error':0,'first':None} for k in fields}
scalar_stats=collections.Counter();event_differences=0;round_discrete_differences=0
for index,(x,y) in enumerate(zip(a,b)):
    sx,sy=x['state'],y['state']
    for key in fields:
        assert len(sx[key])==len(sy[key])
        pairs=[(i,v,w)for i,(v,w)in enumerate(zip(sx[key],sy[key]))if not equal(v,w)]
        if pairs:
            r=array_stats[key];r['differing_replies']+=1;r['differing_values']+=len(pairs)
            i,v,w=pairs[0]
            if r['first'] is None:r['first']={'request_index':index,'tick':sx['tick'],'index':i,'eager':v,'graph':w,'absolute_error':abs(v-w)}
            for i,v,w in pairs:
                error=abs(v-w)
                if error>r['max_absolute_error']:r['max_absolute_error']=error;r['maximum_at']={'request_index':index,'tick':sx['tick'],'index':i,'eager':v,'graph':w}
    for key in ['tick','completedRounds','terminal','winner','roundResult','roundNumber','failureBits','phase','fightResult','fightWinner','ties','redos','unclassified','timeRemaining']:
        if sx[key]!=sy[key]:scalar_stats[key]+=1
    if x.get('commandEvents',[])!=y.get('commandEvents',[]):event_differences+=1
    keys=['tick','score','falls','wins','roundResult','winner','fightResult','fightWinner','roundNumber']
    rx=[{k:r[k]for k in keys}for r in x.get('rounds',[])];ry=[{k:r[k]for k in keys}for r in y.get('rounds',[])]
    if rx!=ry:round_discrete_differences+=1
summary=json.loads((result/'SUMMARY.json').read_text())
original= (root/'compare_workers.executed-r1.py').read_bytes();assert hashlib.sha256(original).hexdigest()=='c5272ed6ed02368731e7095054ff0c6f9ca42f5c6af2bb4cd41091f328c300cb'
report={'schema':'rek.native_clone.graph_failure_analysis.v1','decision':'Keep cuda_graph_step disabled; no extra GPU run required for current clone release.',
 'source_summary_sha256':sha(result/'SUMMARY.json'),'executed_driver_sha256':sha(root/'compare_workers.executed-r1.py'),'corrected_future_driver_sha256':sha(root/'compare_workers.py'),
 'correction':'The executed SUMMARY scope string incorrectly said all values compared exactly. Its ok=false, mismatch count356 and strict assertion were correct. Original outputs remain unchanged; corrected future driver uses conditional outcome wording.',
 'replies_compared':365,'differing_replies':summary['mismatch_replies'],'initial_and_reset_before_first_step_exact':a[:3]==b[:3],
 'array_comparisons':array_stats,'scalar_differing_reply_counts':dict(scalar_stats),'command_event_differing_replies':event_differences,'round_discrete_differing_replies':round_discrete_differences,
 'eager_terminals':summary['eager_terminals'],'graph_terminals':summary['graph_terminals'],'eager_benchmark':summary['eager_benchmark'][0],'graph_benchmark':summary['graph_benchmark'][0],'speed_ratio':summary['benchmark_speed_ratio'],
 'interpretation':'The first differences occur in the first executed control tick, after matching initial/cold-reset snapshots. A single pair of fresh processes does not distinguish graph scheduling effects from ordinary parallel-physics nondeterminism. No tolerance was relaxed; no attribution to a specific kernel is established. The11.38% whole-worker speed increase does not justify enabling a feature that fails the specified strict equivalence gate.',
 'no_new_gpu_runs':True,'units_note':'qpos/raw aggregate maxima combine heterogeneous components; their maxima are diagnostic numeric differences, not position distances.'}
(out/'ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n')
first=array_stats['qpos']['first']
text=f'''# CUDA graph comparison: disabled

The strict comparison failed in {summary['mismatch_replies']} of365 replies. Initial and cold-reset snapshots before the first step matched. The first difference was qpos[{first['index']}] at tick1: {first['absolute_error']:.9g} in that component. Other first-tick components also differ; this was not an initial-state advancement.

Both workers exited0 and produced8 short-round terminals. Discrete terminal records differed in {round_discrete_differences} replies; command-event records differed in {event_differences} replies. Full per-field comparisons are in ANALYSIS.json.

The512-step whole-worker benchmark took {summary['eager_benchmark'][0]['wall_seconds']:.6f}s eager and {summary['graph_benchmark'][0]['wall_seconds']:.6f}s captured, {summary['benchmark_speed_ratio']:.4f}× speed. This includes upload, status, snapshot and protocol overhead, with no rendering.

The original summary's scope sentence incorrectly claimed exact agreement despite its correct failure flag and counts. The original trace/report are preserved. The future driver now reports pass/fail wording conditionally, covered by a regression test.

Keep graph mode disabled. This single comparison does not identify whether the differences arise from capture or existing parallel numerical nondeterminism. No extra GPU run or tolerance relaxation is required for the current release. Match-lifecycle work remains independent.
'''
(out/'ANALYSIS.md').write_text(text)
# Normalize line endings only for the human-readable diff; source snapshots stay intact.
(root/'worker.diff').write_text(''.join(difflib.unified_diff((root/'eval_worker.original.cpp').read_text().splitlines(True),(root/'eval_worker.cpp').read_text().splitlines(True),fromfile='original/eval_worker.cpp',tofile='candidate/eval_worker.cpp')),newline='\n')
print(json.dumps({'analysis_sha256':sha(out/'ANALYSIS.json'),'first_qpos':first,'scalar_diff':dict(scalar_stats),'event_diff':event_differences,'round_diff':round_discrete_differences,'corrected_driver_sha256':sha(root/'compare_workers.py')},indent=2))
