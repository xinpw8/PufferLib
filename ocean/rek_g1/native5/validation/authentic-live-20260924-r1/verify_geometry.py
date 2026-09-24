import collections
import importlib.util
import json
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location('analysis', Path(__file__).with_name('analyze_completed.py'))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
for attempt in ['authentic-s801-retry9','authentic-s802-retry2']:
    trial = a.BASE / attempt / 'trial'
    states, state_receipt = a.load(trial/'relay.stdout.jsonl', lambda x:x.get('event')=='g1_policy_state')
    sources = {r['observation_sequence']:r for r in states}
    encoded, encoded_receipt = a.load(trial/'encoder.stdout.jsonl', lambda x:x.get('event')=='policy_observation' and x.get('ready') is True)
    workers, worker_receipt = a.load(trial/'worker.stdout.jsonl', lambda x:x.get('type')=='action')
    comparisons = []
    for row in encoded:
        request = row['worker_request']
        source = sources[request['seq']]
        geom = a.geometry(source)
        obs = request['observation']
        error = geom['own_bearing_deg']-180*obs[87]
        wrapped_error = abs(math.degrees(math.atan2(math.sin(math.radians(error)), math.cos(math.radians(error)))))
        comparisons.append((wrapped_error, abs(geom['root_gap_xz_m']-obs[86])))
    worker_attacks = [r for r in workers if r['action'] >= 16]
    action_counts = collections.Counter(r['action'] for r in worker_attacks)
    local = states[0]['local_slot']
    result = {'attempt':attempt,'compared_ready_observations':len(comparisons),
        'max_absolute_wrapped_bearing_difference_deg':max(x[0] for x in comparisons),
        'max_root_gap_difference_m':max(x[1] for x in comparisons),
        'comparison':'Offline projected root-local +X geometry versus recorded encoder worker_request.observation[87]*180 and observation[86]',
        'native_forward_proof':'ocean/rek_g1/native5/validation/observable-balance-heading-20260921.md',
        'worker_attack_proposals':len(worker_attacks),'worker_attack_proposals_by_action':dict(action_counts),
        'worker_action16_proposals':action_counts[16],
        'left_side_kick_action16_native6_proposals':action_counts[16],
        'left_front_kick_action17_native7_proposals':action_counts[17],
        'left_jab_action21_native1_proposals':action_counts[21],
        'source_samples':len(states),
        'local_runner_move_index_available_samples':sum(r['fighters'][local]['runner']['current_move_index'] is not None for r in states),
        'local_runner_motion_name_available_samples':sum(r['fighters'][local]['runner']['current_motion_name'] is not None for r in states),
        'native_action_busy_available_samples':sum(r['input']['action_busy'] is not None for r in states),
        'server_start_or_move_execution_identified':False,
        'inputs':[state_receipt,encoded_receipt,worker_receipt]}
    assert result['max_absolute_wrapped_bearing_difference_deg'] < 1e-7 and result['max_root_gap_difference_m'] < 1e-7
    with (a.OUT/(attempt+'.geometry-verification.json')).open('x') as h:
        json.dump(result,h,indent=2);h.write('\n')
    print(json.dumps(result))
