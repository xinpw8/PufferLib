"""Offline, CPU-only analysis of two completed policy logs. No client connections."""
import base64
import bisect
import collections
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

BASE = Path('/home/spark-advantage/rek-training/authentic-ppo-live-20260924-r1')
OUT = BASE / 'analysis-s801r9-s802r2-20260924-r1'
MOVES = [6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16]
CALL_NAMES = ['Slip', 'SlipEStop', 'Knockdown', 'BeatCount', 'Knockout', 'DoubleKnockdown', 'DoubleKnockout']

def describe(values):
    a = sorted(v for v in values if v is not None and math.isfinite(v))
    return {'n': len(a), 'min': min(a) if a else None, 'median': statistics.median(a) if a else None,
            'mean': statistics.mean(a) if a else None, 'p95': a[math.floor((len(a)-1)*.95)] if a else None,
            'max': max(a) if a else None}

def geometry(s):
    local = s['local_slot']
    a, b = s['fighters'][local], s['fighters'][local ^ 1]
    dx = b['root_position_xyz'][0] - a['root_position_xyz'][0]
    dz = b['root_position_xyz'][2] - a['root_position_xyz'][2]
    bearings = []
    for side, f in enumerate([a, b]):
        q = f['root_rotation_xyzw']
        norm = math.sqrt(sum(v*v for v in q))
        x, y, z, w = [v / norm for v in q]
        fx, fz = 1-2*(y*y+z*z), 2*(x*z-w*y)
        bearing = None
        if math.hypot(fx, fz) > 1e-12 and math.hypot(dx, dz) > 0:
            angle = math.atan2(dz if side == 0 else -dz, dx if side == 0 else -dx) - math.atan2(fz, fx)
            bearing = math.degrees(math.atan2(math.sin(angle), math.cos(angle)))
        bearings.append(bearing)
    return {'root_gap_xz_m': math.hypot(dx, dz), 'own_bearing_deg': bearings[0], 'opponent_bearing_deg': bearings[1]}

def geometry_summary(rows):
    out = {key: describe([r[key] for r in rows]) for key in ['root_gap_xz_m', 'own_bearing_deg', 'opponent_bearing_deg']}
    out['abs_own_bearing_deg'] = describe([abs(r['own_bearing_deg']) for r in rows if r['own_bearing_deg'] is not None])
    for threshold in [15, 30, 45, 90]:
        out['own_abs_bearing_over_' + str(threshold) + '_count'] = sum(r['own_bearing_deg'] is not None and abs(r['own_bearing_deg']) > threshold for r in rows)
    for threshold in [.5, .75, 1, 1.25, 1.5]:
        out['root_gap_over_' + str(threshold) + '_m_count'] = sum(r['root_gap_xz_m'] > threshold for r in rows)
    return out

def load(file, keep):
    before = file.stat()
    digest = hashlib.sha256()
    rows = []
    with file.open('rb') as h:
        for line_number, line in enumerate(h, 1):
            digest.update(line)
            if not line.strip():
                continue
            obj = json.loads(line)
            if keep(obj):
                obj['_line'] = line_number
                rows.append(obj)
    after = file.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    return rows, {'file': str(file), 'bytes': after.st_size, 'sha256': digest.hexdigest()}

def validate_referee(s):
    r = s['referee']
    if not r['available']:
        return None
    body = base64.b64decode(r['wire_body_base64'], validate=True)
    assert len(body) == 33 and hashlib.sha256(body).hexdigest() == r['wire_body_sha256']
    assert r['observation_hooks_verified'] and r['reason'] == 'received_snapshot_applied_and_bound'
    assert r['authority_scope'] == 'server_authored_packet_observed_on_client_not_server_current_state'
    for key, offset in [('count_mask',25), ('count_seconds',26), ('call_sequence',27), ('packet_phase',0), ('packet_round_number',1), ('packet_round_result',13)]:
        assert r[key] == body[offset]
    assert r['slot0_count_active'] == bool(body[25]&1) and r['slot1_count_active'] == bool(body[25]&2)
    assert r['packet_round_active'] == bool(body[2]) and r['packet_round_knockout_occurred'] == bool(body[12])
    age = (s['clock']['qpc_ticks'] - r['receipt_qpc_ticks']) / r['receipt_qpc_frequency_hz']
    assert r['receipt_qpc_frequency_hz'] == s['clock']['qpc_frequency_hz'] and 0 <= age <= .5
    assert abs(age-r['receipt_age_seconds']) < 1e-9
    if body[27]:
        assert r['call_available'] and r['call_type'] == body[28] and r['call_name'] == CALL_NAMES[body[28]]
        assert r['call_faller'] == struct.unpack_from('b', body, 29)[0] and r['call_points'] == body[30]
    else:
        assert not r['call_available']
    return {'counters': list(struct.unpack_from('<hh', body, 8)), 'age': age}

def analyze(attempt):
    trial = BASE / attempt / 'trial'
    summary = json.loads((trial / 'summary.json').read_text())
    rows, provenance = load(trial / 'relay.stdout.jsonl', lambda r: r.get('event', '').startswith('g1_policy_'))
    all_states = [r for r in rows if r['event'] == 'g1_policy_state']
    round_id = next(r['round_identity_sha256'] for r in all_states if r['stream_active'])
    states = [r for r in all_states if r['round_identity_sha256'] == round_id]
    states.sort(key=lambda r: r['clock']['qpc_ticks'])
    by_seq = {r['observation_sequence']: r for r in states}
    assert len(by_seq) == len(states)
    local = states[0]['local_slot']
    assert all(s['local_slot'] == local for s in states)
    hz = states[0]['clock']['qpc_frequency_hz']
    origin = states[0]['clock']['qpc_ticks']
    ticks = [s['clock']['qpc_ticks'] for s in states]
    t = lambda s: (s['clock']['qpc_ticks'] - origin) / hz
    calls, receipts, score_events, count_episodes = {}, {}, [], []
    active_counts = {}
    previous_receipt = None
    previous_score = states[0]['round']['clean_hits']
    previous_state = states[0]
    for s in states:
        r = s['referee']
        checked = validate_referee(s)
        if checked is not None:
            receipt_id = (r['lifecycle'], r['receipt_sequence'])
            if receipt_id not in receipts:
                receipts[receipt_id] = True
                fresh = None
                if r['call_available']:
                    call_id = (r['lifecycle'], r['call_observation_sequence'])
                    if call_id not in calls:
                        fresh = {'call_id': list(call_id), 'sequence': r['call_sequence'], 'name': r['call_name'], 'faller': r['call_faller'],
                                 'points': r['call_points'], 'censored': r['call_history_censored'], 'first_transition': r['call_sequence_transition'],
                                 'left_censored_in_analysis': previous_receipt is None or r['call_sequence_transition'] == 'repeated_latched_call',
                                 'receipt_elapsed_s': (r['receipt_qpc_ticks']-origin)/hz, 'source_elapsed_s': t(s),
                                 'source_line': s['_line'], 'counters': checked['counters'], 'count_mask': r['count_mask']}
                        calls[call_id] = fresh
                for slot in [0, 1]:
                    on = bool(r['count_mask'] & (1 << slot))
                    elapsed = (r['receipt_qpc_ticks']-origin)/hz
                    if on and slot not in active_counts:
                        active_counts[slot] = {'slot': slot, 'start_receipt_s': elapsed, 'start_source_line': s['_line'], 'count_seconds_seen': [],
                                               'left_censored': previous_receipt is None, 'onset_call': fresh['name'] if fresh else None}
                    if on:
                        episode = active_counts[slot]
                        if r['count_seconds'] not in episode['count_seconds_seen']:
                            episode['count_seconds_seen'].append(r['count_seconds'])
                    elif slot in active_counts:
                        episode = active_counts.pop(slot)
                        episode.update(end_receipt_s=elapsed, duration_s=elapsed-episode['start_receipt_s'], right_censored=False,
                                       resolution_call=fresh['name'] if fresh else None,
                                       explicit_countout=bool(fresh and not fresh['censored'] and not fresh['left_censored_in_analysis'] and
                                           (fresh['name'] == 'DoubleKnockout' or fresh['name'] == 'Knockout' and fresh['faller'] == slot)))
                        count_episodes.append(episode)
                previous_receipt = r
        score = s['round']['clean_hits']
        for slot in [0, 1]:
            delta = score[slot] - previous_score[slot]
            assert delta >= 0
            if delta:
                score_events.append({'slot': slot, 'delta': delta, 'new_points': score[slot], 'elapsed_s': t(s), 'observation_sequence': s['observation_sequence'],
                                     'source_line': s['_line'], 'observation_gap_s': t(s)-t(previous_state),
                                     'current_call_name': r.get('call_name'), 'current_call_faller': r.get('call_faller'), 'current_call_points': r.get('call_points'),
                                     'current_call_id': [r.get('lifecycle'),r.get('call_observation_sequence')], 'current_count_mask': r.get('count_mask'),
                                     **geometry(s)})
        previous_score, previous_state = score, s
    for episode in active_counts.values():
        episode.update(end_receipt_s=t(states[-1]), duration_s=t(states[-1])-episode['start_receipt_s'], right_censored=True, explicit_countout=False)
        count_episodes.append(episode)
    for event in score_events:
        event['near_new_explicit_five_point_calls'] = [c['call_id'] for c in calls.values() if event['delta'] == 5 and c['points'] == 5
            and c['name'] in ['Knockout','DoubleKnockout'] and not c['censored'] and not c['left_censored_in_analysis']
            and abs(event['elapsed_s'] - c['receipt_elapsed_s']) <= .25 and c['counters'][event['slot']] == event['new_points']
            and (c['name'] == 'DoubleKnockout' or c['faller'] != event['slot'])]
    attacks = []
    acks = [r for r in rows if r['event'] == 'g1_policy_action' and r['round_identity_sha256'] == round_id]
    ack_ids = {r['request_id']: r for r in acks}
    for d in [r for r in rows if r['event'] == 'g1_policy_dispatch' and r['round_identity_sha256'] == round_id]:
        source = by_seq[d['observation_sequence']]
        ack = ack_ids[d['request_id']]
        assert d['send_method_returned'] and ack['applied'] and MOVES[ack['action']-16] == d['move_index']
        i = bisect.bisect_right(ticks, d['clock']['qpc_ticks'])-1
        preceding = states[i]
        attacks.append({'request_id': d['request_id'], 'action': ack['action'], 'move_index': d['move_index'], 'elapsed_s': t(d),
                        'source_observation_sequence': source['observation_sequence'], 'source_age_s': t(d)-t(source), 'source_geometry': geometry(source),
                        'preceding_pose_age_s': t(d)-t(preceding), 'geometry': geometry(preceding),
                        'own_count_flag': preceding['referee'].get('slot'+str(local)+'_count_active'),
                        'opponent_count_flag': preceding['referee'].get('slot'+str(local^1)+'_count_active'),
                        'actual_server_attack_start': 'unknown', 'server_acceptance': d['server_acceptance']})
    exposure = {'known_referee_s':0., 'unknown_referee_s':0., 'own_count_s':0., 'opponent_count_s':0., 'both_count_s':0.}
    for s, nxt in zip(states, states[1:]):
        if not s['round']['active']:
            continue
        dt = t(nxt)-t(s)
        r = s['referee']
        if not r['available']:
            exposure['unknown_referee_s'] += dt
            continue
        exposure['known_referee_s'] += dt
        own, opp = r['slot'+str(local)+'_count_active'], r['slot'+str(local^1)+'_count_active']
        exposure['own_count_s'] += dt*own
        exposure['opponent_count_s'] += dt*opp
        exposure['both_count_s'] += dt*(own and opp)
    result = {'attempt':attempt, 'checkpoint_sha256':summary['checkpoint_sha256'], 'round_identity_sha256':round_id, 'local_slot':local,
        'final_round':summary['final_round'], 'first_round':states[0]['round'], 'last_round':states[-1]['round'],
        'sources':len(states), 'active_sources':sum(s['round']['active'] for s in states), 'source_span_s':t(states[-1]),
        'source_intervals_s':describe([t(b)-t(a) for a,b in zip(states,states[1:])]), 'source_hz':(len(states)-1)/t(states[-1]),
        'policy_applied':sum(r['applied'] for r in acks), 'policy_applied_s':sum(r['applied'] for r in acks)/t(states[-1]),
        'wire_validated_available_sources':sum(s['referee']['available'] for s in states), 'unique_received_referee_snapshots':len(receipts),
        'point_counter_delta_counts_by_slot':[{str(v):sum(e['slot']==slot and e['delta']==v for e in score_events) for v in [1,2,5]} for slot in [0,1]],
        'point_counter_delta_sum_by_slot':[sum(e['delta'] for e in score_events if e['slot']==slot) for slot in [0,1]],
        'other_point_deltas':[e for e in score_events if e['delta'] not in [1,2,5]],
        'new_referee_call_counts':dict(collections.Counter(c['name']+':faller'+str(c['faller']) for c in calls.values() if not c['censored'] and not c['left_censored_in_analysis'])),
        'referee_calls':list(calls.values()), 'count_episodes':count_episodes, 'count_flag_exposure_left_hold_estimate':exposure,
        'visual_falling_samples_by_slot':[sum(s['fighters'][slot]['falling'] for s in states) for slot in [0,1]],
        'visual_fallen_samples_by_slot':[sum(s['fighters'][slot]['fallen'] for s in states) for slot in [0,1]],
        'attack_requests':sum(r['action']>=16 for r in acks), 'locally_applied_attack_requests':sum(r['action']>=16 and r['applied'] for r in acks),
        'attack_dispatches':len(attacks), 'attack_mix_by_action':dict(collections.Counter(a['action'] for a in attacks)),
        'attack_mix_by_native_move':dict(collections.Counter(a['move_index'] for a in attacks)),
        'dispatch_geometry':geometry_summary([a['geometry'] for a in attacks]),
        'dispatch_pose_age_s':describe([a['preceding_pose_age_s'] for a in attacks]),
        'attacks_during_own_count':sum(a['own_count_flag'] is True for a in attacks), 'attacks_during_opponent_count':sum(a['opponent_count_flag'] is True for a in attacks),
        'geometry_by_native_move':{str(move):geometry_summary([a['geometry'] for a in attacks if a['move_index']==move]) for move in sorted(set(a['move_index'] for a in attacks))},
        'policy_action_mix':dict(collections.Counter(r['action'] for r in acks if r['applied'])), 'inputs':[provenance]}
    assert result['point_counter_delta_sum_by_slot'] == [summary['final_round']['clean_hits'][i]-states[0]['round']['clean_hits'][i] for i in [0,1]]
    with (OUT/(attempt+'.summary.json')).open('x') as h:
        json.dump(result,h,indent=2); h.write('\n')
    for name,data in [('attacks',attacks),('score-deltas',score_events)]:
        with (OUT/(attempt+'.'+name+'.jsonl')).open('x') as h:
            for row in data: h.write(json.dumps(row)+'\n')
    return result

assert abs(geometry({'local_slot':0,'fighters':[{'root_position_xyz':[0,0,0],'root_rotation_xyzw':[0,0,0,1]},
    {'root_position_xyz':[1,0,0],'root_rotation_xyzw':[0,1,0,0]}]})['own_bearing_deg']) < 1e-9
if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    for name in ['authentic-s801-retry9','authentic-s802-retry2']:
        report=analyze(name)
        print(json.dumps({k:v for k,v in report.items() if k not in ['geometry_by_native_move','first_round','last_round']}))
