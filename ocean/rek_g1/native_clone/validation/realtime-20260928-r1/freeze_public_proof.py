"""Freeze selected closed synthetic evidence. No model/runtime execution."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PENDING = ROOT / 'public-proof-pending-r1'
FINAL = ROOT / 'public-proof-final-r1'
COMPACT = Path(r'C:\rekagent\work\rek-playback-speed-20260928-r2\compact-results-r1')
APP = Path(r'C:\rekagent\work\rek-playback-app-20260928-r7')
REPO = Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile\ocean\rek_g1\native_clone\validation\realtime-20260928-r1')
NAS = Path(r'\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-28\rek-playback-speed-r1\realtime-proof-r1')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def copy(source, relative):
    destination = FINAL / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as handle:
        handle.write(source.read_bytes())

def write(relative, value):
    path = FINAL / relative
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(value)

assert sha(COMPACT/'MANIFEST.json') == '5aebab4ecee4583fd7e8bc1be23145e2fbdc13a29287b317e317cb830b96e611'
cm = json.loads((COMPACT/'MANIFEST.json').read_text())
for row in cm['files']:
    source = COMPACT / row['path']
    assert sha(source) == row['sha256'] and source.stat().st_size == row['bytes']
    assert source.name not in ('session.jsonl', 'samples.json', 'input-requests.json')
    assert source.suffix != '.onnx'
assert {p.relative_to(COMPACT).as_posix() for p in COMPACT.rglob('*') if p.is_file()} == {r['path'] for r in cm['files']} | {'MANIFEST.json'}
assert sha(APP/'app/SOURCE-MANIFEST.json') == '6be9fa74673eca108da8c07cd414d27d55b6cc9f625db1ab70e2f5c3e6f02cca'
FINAL.mkdir()
for source in sorted(PENDING.rglob('*')):
    if source.is_file() and source.name not in ('README-PENDING.md', 'HISTORY-PENDING.json', 'STAGING-MANIFEST.json'):
        copy(source, source.relative_to(PENDING))
for source in sorted(COMPACT.rglob('*')):
    if source.is_file():
        copy(source, Path('verified-closed') / source.relative_to(COMPACT))
for name in ('APP-R7-REVIEW.json', 'CROSSBATCH-INDEPENDENT-REVIEW.json'):
    copy(ROOT/name, name)
for name in ('R6-SCHEDULER.json', 'R7-REVIEW-AND-MECHANISM.json', 'STAGED.json', 'app-r7.diff'):
    copy(APP/name, Path('app-source-review') / name)
for name in ('paced_loop.cjs', 'paced_loop.test.cjs', 'server.cjs'):
    copy(APP/'app/league'/name, Path('app-source-review') / name)
for name in ('tests-r7.json', 'tests-r7.txt'):
    copy(APP/'app/validation'/name, Path('app-source-review') / name)

selected = json.loads((COMPACT/'app60-result-r7-affinity/summary.json').read_text())
analysis = json.loads((COMPACT/'app60-analysis-r7-affinity/ANALYSIS.json').read_text())
prior = json.loads((COMPACT/'app60-result-r7/summary.json').read_text())
worker = json.loads((COMPACT/'app60-run-r7-affinity/worker.json').read_text())
identity = json.loads((COMPACT/'app60-run-r7-affinity/identity.json').read_text())
assert selected['success'] and selected['guard_failure'] is None and analysis['success']
assert selected['final_tick'] == analysis['validated_full_states'] == 2991
assert selected['closed_recorder_health']['frames'] == analysis['frames_verified'] == 1207
assert selected['input_packets'] == 3000 and selected['image_gets'] == 1200
assert analysis['ticks_contiguous'] and all(row['code'] == 0 for row in analysis['worker_exit_events'])
assert worker['arenas'] == 1 and worker['cuda_graph_step'] is False
assert identity['env']['REK_PHYSICS_BACKEND'] == 'mujoco_cuda'
assert identity['env']['REK_ALLOW_CPU_EVALUATION'] == '0'
assert identity['execution']['controlStepSeconds'] == 0.02
assert selected['pace']['scheduler']['rebases'] == 1
assert all(row['allowed_cpus'] == [5,6,7,8,9,15,16,17,18,19] for row in selected['owned_affinity'])
steady = selected['steady_after5s']
assert abs(steady['active_intervals'] * 20 / steady['active_wall_ms'] - steady['real_time_ratio']) < 1e-12
outer = selected['final_tick'] * 0.02 / selected['elapsed_seconds']
assert abs(outer - analysis['outer_wall_ratio']) < 1e-12
result = {
    'schema': 'rek.realtime_final_proof.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
    'selected': 'r7_one_arena_batch2_gpu_cpu_affinity_5-9_15-19',
    'native_sha256': 'ef519db4c8b3b3a6696ebc8dfd7b686bffc789a7b91f1ef8d60dfab3494e0975',
    'app_manifest_sha256': '6be9fa74673eca108da8c07cd414d27d55b6cc9f625db1ab70e2f5c3e6f02cca',
    'compact_closed_manifest_sha256': sha(COMPACT/'MANIFEST.json'),
    'description': 'Approximately1x steady-state playback; whole cold-start trial remains below literal1x.',
    'control_step_seconds': 0.02, 'physics_step_seconds': 0.002, 'physics_substeps': 10,
    'graph_enabled': False, 'new_physics_math_or_solver_change': False,
    'cold_start_inclusive': {'actual_steps': selected['final_tick'], 'outer_wall_seconds': selected['elapsed_seconds'],
        'outer_simulated_seconds_per_wall_second': outer,
        'app_active_interval_ratio': selected['pace']['realTimeRatio'],
        'app_active_intervals': selected['pace']['activeIntervals'],
        'app_active_wall_ms': selected['pace']['activeWallMs']},
    'steady_after_first_5_seconds': {**steady, 'control_steps_per_second': steady['real_time_ratio'] * 50},
    'scheduler': selected['pace']['scheduler'],
    'inputs': selected['input_packets'], 'png_gets': selected['image_gets'], 'recorded_frames': analysis['frames_verified'],
    'ticks_contiguous': True, 'errors': 0, 'guard_failure': None,
    'comparison_unpinned_r7': {'overall_app_ratio': prior['pace']['realTimeRatio'], 'after5s_ratio': prior['steady_after5s']['real_time_ratio']},
    'prior_r6_app_ratio': 0.9558830809073361,
    'operational_gate': {'source': 'Parent fixed acceptance gate', 'minimum_ratio': 0.99, 'passed': True},
    'literal_entire_trial_ratio_at_least_one': False,
    'limits': ['One closed60s synthetic interactive workload; not a guarantee for every contact load or future run.',
               'Full cold-start wall ratio and app active-interval ratio use different stated denominators.',
               'Batch2 decoder is numerically compatible on tested inputs, not bit-identical to batch8.',
               'Original Unity/official-server physical parity and fighting ability remain unproven.',
               'Strict-reference graph binary is a staged fallback; no graph experiment or deployment occurred.'],
}
write('RESULTS.json', json.dumps(result, indent=2)+'\n')
write('RESULTS.md', f'''# Real-time playback result

The selected one-arena GPU app sustained **0.999985x after the first5s**, with50Hz synthetic controls,20Hz PNG requests and recording active. This is approximately1x steady-state playback. The complete cold-start trial remained below literal1x.

| Measurement | Result |
|---|---:|
| First5s excluded | 2,750 control intervals /55.000839s |
| Steady control rate | {steady['real_time_ratio']*50:.6f} steps/s |
| Steady simulation/wall ratio | {steady['real_time_ratio']:.9f}x |
| Entire trial, actual steps /outer wall | 2,991 /{selected['elapsed_seconds']:.6f}s |
| Entire-trial simulation/outer-wall ratio | {outer:.9f}x |
| App active-interval ratio, including startup | {selected['pace']['realTimeRatio']:.9f}x |
| Recorded frames /PNG requests /input packets | 1,207 /1,200 /3,000 |

All2,991 returned states had contiguous ticks. Both owned workers exited normally, with no recorder, renderer, worker or guard failures. The scheduler reported one wall-debt rebase of162.620ms and a maximum task duration of181.915ms. The first10s window was0.9793x; the later10s windows ranged0.9971–1.0043x. Startup and short scheduling variation remain visible in the record.

The prior r6 full-app run was0.955883x. Absolute20ms deadlines improved the unrestricted r7 run to0.993805x overall and0.996341x after5s. The selected run additionally confined its own Node, simulation and renderer processes to CPU cores5–9 and15–19. This measured comparison is one run of each configuration, not a statistical performance guarantee.

The final runtime uses one physical arena, two controller rows, the existing batch2 weights, separate rendering and the r7 scheduler. Control dt remains20ms, physics dt2ms with ten substeps, and solver/motor weights are unchanged. CUDA graph mode is off. Static tensors match the batch8 export; all4,096 sampled tokens matched exactly, while connected decoder outputs differed by at most9.54e-6. That supports bounded numerical compatibility, with the failed exact comparison preserved under `batch2`.

The four-arena phase profile identified physics step/refresh as14.917ms of20.350ms mean runtime. Measurement/referee/reset used2.764ms and preparation1.118ms. Host and CUDA timing intervals overlap and must not be added. The r6 timing analysis also found1,132 of2,867 steady RPCs above20ms: completion-relative scheduling retained those delays instead of recovering during shorter steps. The included SCHEDULER.json states the limits of this causal timing model.

Native executable SHA: `ef519db4c8b3b3a6696ebc8dfd7b686bffc789a7b91f1ef8d60dfab3494e0975`.
App source manifest SHA: `6be9fa74673eca108da8c07cd414d27d55b6cc9f625db1ab70e2f5c3e6f02cca`.

The parent acceptance gate of at least0.99x passed. This report does not claim a literal whole-trial result of at least1.000x, full Unity/server physics parity, improved fighting ability or a universal50Hz guarantee. No model weights or raw human-session traces are included. The linked strict-reference alternative remains an unexecuted fallback.
''')
write('README.md', '''# Closed real-time validation

Read RESULTS.md for the selected result and its cold-start limitation. RESULTS.json preserves the exact denominators. `verified-closed` contains hash-verified compact source results and configuration, including earlier diagnostic outcomes. `batch2` preserves static model-coefficient and direct native inference tests, including the exact-match failure. `app-source-review` and the review receipts bind the scheduler source and CPU checks. `phase-profile-source` contains diagnostic instrumentation only. `strict-reference-preparation` was linked but never executed on the GPU.

The production app and deployment artifacts are published separately. This directory contains evidence and small test tools. It does not contain model weights or raw human motion/input traces.
''')
copy(Path(__file__), 'freeze_public_proof.py')
records = [{'path':p.relative_to(FINAL).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(FINAL.rglob('*')) if p.is_file()]
write('MANIFEST.json', json.dumps({'schema':'rek.realtime_public_proof_manifest.v1','files':records},indent=2)+'\n')
for destination in (REPO, NAS):
    destination.mkdir()
    for source in sorted(FINAL.rglob('*')):
        if not source.is_file(): continue
        target = destination/source.relative_to(FINAL)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as handle: handle.write(source.read_bytes())
        assert sha(target) == sha(source)
receipt = {'local':str(FINAL),'repo':str(REPO),'nas':str(NAS),'files':len(records)+1,
    'manifest_sha256':sha(FINAL/'MANIFEST.json'),'results_json_sha256':sha(FINAL/'RESULTS.json'),
    'results_md_sha256':sha(FINAL/'RESULTS.md'),'all_destinations_readback_equal':True}
(ROOT/'FINAL-PROOF-PUBLICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
