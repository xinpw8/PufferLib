"""Is there actually an evidence package here yet?

A directory of scripts is not evidence. This checks which artifacts exist, that
each is well formed, and — the part that matters — that they all describe the
same build. Traces recorded against one client build and a survey taken from
another cannot be reasoned about together, and nothing else notices.

    python check_artifacts.py --dir evidence_out/

Exits non-zero until the package is complete, so it can gate a review the way
the parity test gates a clone.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace import Trace
from static_survey import derive_fixed_timestep
from controller_path import validate_report as validate_controller_path

# Stages, in the order the sequence in README.md establishes them. Each depends
# on the one before, so the first gap is the thing to go and do.
STAGES = ('build pinned', 'authority determined', 'statics surveyed',
          'traces recorded', 'envelope established', 'parity tested')
COMMAND_SEQUENCE_SCHEMA = 'rek.client_fixed.command_schedule.v2'


def _load_json(path):
    try:
        return json.loads(path.read_text()), None
    except Exception as e:
        return None, str(e)


def check(directory: Path):
    results = []
    fingerprints = {}

    def record(name, state, detail=''):
        results.append({'artifact': name, 'state': state, 'detail': detail})

    # 1. inventory
    inv_path = directory / 'inventory.json'
    inv = None
    if not inv_path.exists():
        record('inventory.json', 'MISSING',
               'run inventory.py against the install')
    else:
        inv, err = _load_json(inv_path)
        if err:
            record('inventory.json', 'INVALID', err)
            inv = None
        else:
            roots = inv.get('merkle_roots') or {}
            if set(roots) != {'manifest', 'immutable', 'behavioural'}:
                record('inventory.json', 'INVALID',
                       'predates the Merkle roots; re-run inventory.py')
            elif inv.get('build_fingerprint') != roots.get('immutable'):
                record('inventory.json', 'INVALID',
                       'build_fingerprint is not the immutable root')
            elif inv.get('errors'):
                # A fingerprint computed while some files were locked is not an
                # identity: the same install scanned again cleanly will disagree.
                record('inventory.json', 'INCOMPLETE',
                       f'{len(inv["errors"])} file(s) unreadable (e.g. '
                       f'{inv["errors"][0]["path"]}); close the game and re-run, '
                       f'or this fingerprint will not reproduce')
                fingerprints['inventory.json'] = inv['build_fingerprint']
            else:
                fingerprints['inventory.json'] = inv['build_fingerprint']
                buildid = (inv.get('steam') or {}).get('buildid')
                record('inventory.json', 'PRESENT',
                       f'{inv["file_count"]} files, buildid '
                       f'{buildid or "UNKNOWN — no appmanifest found"}')

    # 2. authority
    auth_files = sorted(directory.glob('authority*.json'))
    if not auth_files:
        record('authority test', 'MISSING',
               'run authority_test.py — this decides everything downstream')
    else:
        conclusive = []
        for f in auth_files:
            data, err = _load_json(f)
            if err:
                record(f.name, 'INVALID', err)
                continue
            v = data.get('verdict') or {}
            verdict = v.get('verdict', 'absent')
            if verdict in ('local_authority', 'remote_authority',
                           'local_prediction_remote_correction'):
                conclusive.append((f.name, verdict))
                record(f.name, 'PRESENT', verdict)
            else:
                record(f.name, 'INCONCLUSIVE',
                       f'{verdict}: {v.get("because", "")[:120]}')
        # Always emit the aggregate, so the stage machine has one record to
        # read rather than having to interpret however many run files exist.
        if conclusive:
            record('authority test', 'PRESENT',
                   '; '.join(f'{n}: {v}' for n, v in conclusive))
        else:
            record('authority test', 'MISSING',
                   'runs exist but none reached a conclusive verdict')

    # 3. static survey
    ss_path = directory / 'static_survey.json'
    if not ss_path.exists():
        record('static_survey.json', 'MISSING', 'run static_survey.py')
    else:
        ss, err = _load_json(ss_path)
        if err:
            record('static_survey.json', 'INVALID', err)
        else:
            fingerprints['static_survey.json'] = ss.get('build_fingerprint')
            tm = (ss.get('settings') or {}).get('TimeManager')
            ft = ss.get('fixed_timestep') or {}
            if not ft.get('source') and tm:
                # A survey written before the derivation existed still recorded
                # the raw TimeManager fields, and turning those into Hz is
                # arithmetic over data already in hand. Deriving it here beats
                # sending someone back to the install to recompute it.
                ft = derive_fixed_timestep(tm.get('values') or {})
                if ft.get('source'):
                    ft = dict(ft, source=f'{ft["source"]}, derived at check time')
            if ft.get('source'):
                tick = f'{ft["hz"]:.4g} Hz'
            elif tm:
                # Present but not convertible is not the same as measured, and
                # the tick rate is the unit everything else is expressed in.
                tick = 'PRESENT BUT NOT DERIVABLE'
            else:
                tick = 'NOT FOUND'
            record('static_survey.json', 'PRESENT',
                   f'{len(ss.get("bodies", []))} body/joint components, '
                   f'{len(ss.get("model_assets", []))} model assets, '
                   f'tick rate {tick}')

    # 4c. Bounded native controller semantics. Presence means that exact client
    # method extents and serialized values were pinned. It does not mean those
    # methods execute in the server-authoritative private-AI transition path.
    controller_path = directory / 'controller_path.json'
    if not controller_path.exists():
        record('controller_path.json', 'MISSING',
               'run controller_path.py over the pinned native and ISIL inputs')
    else:
        controller, err = _load_json(controller_path)
        if err:
            record('controller_path.json', 'INVALID', err)
        else:
            errors = validate_controller_path(controller)
            sources = controller.get('sources') or {}
            source_files = (
                ('inventory', directory / 'inventory.json'),
                ('il2cpp_recovery', directory / 'il2cpp_recovery.json'),
                ('asset_probe', directory / str(
                    (sources.get('asset_probe') or {}).get('filename', ''))),
            )
            for source_name, source_path in source_files:
                expected = (sources.get(source_name) or {}).get('sha256')
                if not source_path.is_file():
                    errors.append(f'{source_name} source artifact is missing')
                    continue
                actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
                if actual != expected:
                    errors.append(f'{source_name} source artifact hash mismatch')
            if errors:
                record('controller_path.json', 'INVALID', '; '.join(errors[:4]))
            else:
                fingerprints['controller_path.json'] = controller.get(
                    'build_fingerprint')
                methods = controller.get('native_methods') or []
                moves = (((controller.get('serialized_t800') or {})
                          .get('robot_config') or {}).get('move_map') or [])
                non_null_moves = sum(
                    ((row.get('pointer') or {}).get('path_id') or 0) != 0
                    for row in moves)
                resolved_objects = sum(
                    isinstance(row.get('referenced_object'), dict)
                    and row['referenced_object'].get('state') != 'unknown'
                    for row in moves)
                record('controller_path.json', 'PRESENT',
                       f'{len(methods)} native methods, {non_null_moves}/'
                       f'{len(moves)} non-null T800 move pointers, '
                       f'{resolved_objects} referenced objects resolved; '
                       f'private-AI activation UNKNOWN')

    # 5. traces
    traces = sorted(directory.glob('*.trace'))
    rek_traces, clone_traces, bad = [], [], []
    for t in traces:
        try:
            tr = Trace.load(t)
        except Exception as e:
            bad.append((t.name, str(e)))
            continue
        fingerprints[t.name] = tr.build_fingerprint
        (rek_traces if tr.source == 'rek' else clone_traces).append((t.name, tr))
    for name, err in bad:
        record(name, 'INVALID', err)

    if not rek_traces:
        record('REK traces', 'MISSING',
               'the recorder has to exist and be run first')
    elif len(rek_traces) < 3:
        record('REK traces', 'INCOMPLETE',
               f'{len(rek_traces)} run(s); at least 3 repeats of one experiment '
               'are needed before a quantile envelope means anything')
    else:
        command_sequences = [tr.header.get('command_sequence_sha256')
                             for _, tr in rek_traces]
        command_schemas = [tr.header.get('command_sequence_schema')
                           for _, tr in rek_traces]
        if not all(command_sequences):
            record('REK traces', 'INCOMPLETE',
                   'every run must identify its measured command sequence')
        elif (not all(command_schemas)
              or set(command_schemas) != {COMMAND_SEQUENCE_SCHEMA}):
            record('REK traces', 'INCOMPLETE',
                   f'runs must use command sequence schema '
                   f'{COMMAND_SEQUENCE_SCHEMA}')
        elif len(set(command_sequences)) != 1:
            record('REK traces', 'INCOMPLETE',
                   'runs used different command sequences; repeat one experiment')
        else:
            fighter_pairings = [tr.header.get('fighter_pairing')
                                for _, tr in rek_traces]
            if not all(isinstance(pairing, dict) for pairing in fighter_pairings):
                record('REK traces', 'INCOMPLETE',
                       'every run must identify its measured fighter pairing')
            elif len({json.dumps(pairing, sort_keys=True, separators=(',', ':'))
                      for pairing in fighter_pairings}) != 1:
                record('REK traces', 'INCOMPLETE',
                       'runs used different fighter pairings; repeat one experiment')
            else:
                sample_phases = [
                    tr.header.get('command_sample_phase_substeps')
                    for _, tr in rek_traces]
                if any(phase is None for phase in sample_phases):
                    record('REK traces', 'INCOMPLETE',
                           'every run must identify its measured fixed-substep '
                           'sample phase')
                elif len(set(sample_phases)) != 1:
                    record('REK traces', 'INCOMPLETE',
                           'runs sampled different fixed-substep phases; '
                           'repeat one experiment at one phase')
                else:
                    record('REK traces', 'PRESENT',
                           f'{len(rek_traces)} runs at fixed-substep phase '
                           f'{sample_phases[0]:+d}')

    # A server-authoritative trace has to pin the server too, and every REK
    # channel has to say where it was read from.
    for name, tr in rek_traces:
        uncited = [c for c in tr.header.get('channels', [])
                   if c not in tr.provenance]
        if uncited:
            record(name, 'INVALID',
                   f'{len(uncited)} channel(s) with no provenance: {uncited[:5]}')
        elif tr.authority == 'server' and not tr.server.get('session_id'):
            record(name, 'INVALID', 'server-authoritative with no server identity')
        elif tr.authority == 'unknown':
            record(name, 'INCOMPLETE',
                   'authority not declared; set local or server once the '
                   'authority test has answered it')

    # 5. envelope
    env_path = directory / 'envelope.json'
    if not env_path.exists():
        record('envelope.json', 'MISSING', 'differ.py baseline over the REK runs')
    else:
        env, err = _load_json(env_path)
        if err:
            record('envelope.json', 'INVALID', err)
        elif env.get('command_sequence_schema') != COMMAND_SEQUENCE_SCHEMA:
            record('envelope.json', 'INVALID',
                   f'envelope predates command sequence schema '
                   f'{COMMAND_SEQUENCE_SCHEMA}; rebuild it from valid repeats')
        elif not env.get('command_sequence_sha256'):
            record('envelope.json', 'INVALID',
                   'envelope has no measured command sequence identity')
        else:
            fingerprints['envelope.json'] = env.get('build_fingerprint')
            record('envelope.json', 'PRESENT',
                   f'{env.get("runs")} runs, accept at '
                   f'{env.get("accept_at", "max (old schema)")}, '
                   f'{len(env.get("channels", {}))} channels')

    # 6. parity
    par_path = directory / 'parity_report.json'
    if not par_path.exists():
        record('parity_report.json', 'MISSING',
               'nothing has been held to the envelope yet')
    else:
        par, err = _load_json(par_path)
        if err:
            record('parity_report.json', 'INVALID', err)
        else:
            record('parity_report.json',
                   'PRESENT' if par.get('passed') else 'FAILING',
                   f'{par.get("clone")}, mode {par.get("mode")}, '
                   f'{len(par.get("failures", []))} failure(s)')

    # Cross-check: one build, or none of this composes.
    distinct = {fp for fp in fingerprints.values() if fp}
    if len(distinct) > 1:
        by_fp = {}
        for name, fp in fingerprints.items():
            if fp:
                by_fp.setdefault(fp, []).append(name)
        record('build agreement', 'INCONSISTENT',
               'artifacts describe %d different builds: %s' % (
                   len(distinct),
                   '; '.join(f'{fp[:12]}... <- {", ".join(sorted(names))}'
                             for fp, names in by_fp.items())))
    elif distinct:
        record('build agreement', 'PRESENT',
               f'all {len(fingerprints)} artifact(s) cite {list(distinct)[0][:16]}...')

    return results


def stage_of(results):
    state = {r['artifact']: r['state'] for r in results}
    ok = lambda k: state.get(k) == 'PRESENT'
    if any(r['state'] == 'INCONSISTENT' for r in results):
        return 'inconsistent — artifacts describe different builds'
    if not ok('inventory.json'):
        return 'no evidence'
    if not ok('authority test'):
        return STAGES[0]
    if not ok('static_survey.json'):
        return STAGES[1]
    if not ok('controller_path.json'):
        return STAGES[2]
    if not ok('REK traces'):
        return STAGES[2]
    if not ok('envelope.json'):
        return STAGES[3]
    if not ok('parity_report.json'):
        return STAGES[4]
    return STAGES[5]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default='.', help='directory holding the artifacts')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    directory = Path(args.dir)
    if not directory.is_dir():
        sys.exit(f'{directory} is not a directory')
    results = check(directory)
    stage = stage_of(results)
    complete = stage == STAGES[-1]

    if args.json:
        print(json.dumps({'stage': stage, 'complete': complete,
                          'results': results}, indent=1))
    else:
        width = max(len(r['artifact']) for r in results)
        for r in results:
            print(f'  {r["state"]:<12} {r["artifact"]:<{width}}  {r["detail"]}')
        print(f'\nstage: {stage}')
        if not complete:
            nxt = next((r for r in results
                        if r['state'] in ('MISSING', 'INVALID', 'INCOMPLETE',
                                          'INCONSISTENT')), None)
            if nxt:
                print(f'next : {nxt["artifact"]} — {nxt["detail"]}')
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
