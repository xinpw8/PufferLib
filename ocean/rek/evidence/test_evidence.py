"""Checks for the evidence package. No game install and no UnityPy needed.

These cover the machinery that decides whether a parity claim is trustworthy:
that a build fingerprint is stable against noise and sensitive to real change,
that a trace round-trips, and that the differ actually fails a clone which is
outside REK's own run-to-run spread. They do not, and cannot, say anything
about whether a clone is faithful — only REK can answer that.

    python ocean/rek/evidence/test_evidence.py
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inventory = _load('inventory')
static_survey = _load('static_survey')
trace_mod = _load('trace')
differ = _load('differ')
authority = _load('authority_test')
check_artifacts = _load('check_artifacts')

checks = []


def check(fn):
    checks.append(fn)
    return fn


def fake_install(tmp: Path) -> Path:
    """A minimal Unity IL2CPP Windows build under a Steam library."""
    game = tmp / 'SteamLibrary' / 'steamapps' / 'common' / 'REK'
    (game / 'REK_Data' / 'il2cpp_data' / 'Metadata').mkdir(parents=True)
    (game / 'REK_Data' / 'Plugins' / 'x86_64').mkdir(parents=True)
    (game / 'REK_Data' / 'StreamingAssets').mkdir(parents=True)
    (game / 'logs').mkdir()
    (game / 'REK.exe').write_bytes(b'MZ...')
    (game / 'UnityPlayer.dll').write_bytes(b'unity')
    (game / 'GameAssembly.dll').write_bytes(b'il2cpp')
    (game / 'REK_Data' / 'il2cpp_data' / 'Metadata' / 'global-metadata.dat').write_bytes(b'meta')
    (game / 'REK_Data' / 'globalgamemanagers').write_bytes(b'settings')
    (game / 'REK_Data' / 'sharedassets0.assets').write_bytes(b'assets')
    (game / 'REK_Data' / 'Plugins' / 'x86_64' / 'physx.dll').write_bytes(b'plugin')
    (game / 'REK_Data' / 'StreamingAssets' / 'balance.onnx').write_bytes(b'weights')
    (game / 'logs' / 'output.txt').write_text('noise')
    (tmp / 'SteamLibrary' / 'steamapps' / 'appmanifest_4582660.acf').write_text(
        '"AppState"\n{\n\t"appid"\t\t"4582660"\n\t"name"\t\t"REK"\n'
        '\t"installdir"\t\t"REK"\n\t"buildid"\t\t"19284412"\n}\n')
    return game


@check
def inventory_finds_the_build_identity():
    with tempfile.TemporaryDirectory() as d:
        game = fake_install(Path(d))
        inv = inventory.scan(game)
        assert inv['steam']['buildid'] == '19284412', inv['steam']
        assert inv['steam']['appid'] == '4582660'
        kinds = {f['path']: f['kind'] for f in inv['files']}
        assert kinds['GameAssembly.dll'] == 'il2cpp_code'
        assert kinds['REK_Data/il2cpp_data/Metadata/global-metadata.dat'] == 'il2cpp_metadata'
        assert kinds['UnityPlayer.dll'] == 'unity_runtime'
        assert kinds['REK_Data/globalgamemanagers'] == 'unity_settings'
        assert kinds['REK_Data/Plugins/x86_64/physx.dll'] == 'native_plugin'
        assert kinds['REK_Data/StreamingAssets/balance.onnx'] == 'model_asset'
        assert kinds['REK.exe'] == 'executable'
        # Volatile files are recorded — the manifest is complete — but they are
        # classified so the immutable root can exclude them. Dropping them
        # entirely would leave the record incomplete; counting them toward the
        # identity would make it useless.
        assert kinds['logs/output.txt'] == 'volatile'
        assert inv['file_count'] == len(kinds)
        assert inv['immutable_file_count'] == inv['file_count'] - 1
        roots = inv['merkle_roots']
        assert set(roots) == {'manifest', 'immutable', 'behavioural'}
        assert inv['build_fingerprint'] == roots['immutable']
        assert len({roots['manifest'], roots['immutable']}) == 2


@check
def fingerprint_ignores_noise_and_catches_updates():
    with tempfile.TemporaryDirectory() as d:
        game = fake_install(Path(d))
        first = inventory.scan(game)
        fp = first['build_fingerprint']

        (game / 'logs' / 'output.txt').write_text('a different log entirely')
        churned = inventory.scan(game)
        assert churned['build_fingerprint'] == fp
        # ...but the complete manifest still notices, which is the point of
        # keeping both roots.
        assert churned['merkle_roots']['manifest'] != first['merkle_roots']['manifest']

        (game / 'GameAssembly.dll').write_bytes(b'il2cpp v2')
        assert inventory.scan(game)['build_fingerprint'] != fp


@check
def the_identity_covers_files_no_category_list_anticipated():
    # The failure the immutable root exists to prevent: a behaviour-bearing file
    # in a bucket nobody thought to enumerate. An Addressables bundle dropped in
    # after the fact must move the identity.
    with tempfile.TemporaryDirectory() as d:
        game = fake_install(Path(d))
        before = inventory.scan(game)
        (game / 'REK_Data' / 'StreamingAssets' / 'aa').mkdir(parents=True)
        (game / 'REK_Data' / 'StreamingAssets' / 'aa' / 'catalog.json').write_text('{}')
        after = inventory.scan(game)
        assert after['build_fingerprint'] != before['build_fingerprint']
        kinds = {f['path']: f['kind'] for f in after['files']}
        assert kinds['REK_Data/StreamingAssets/aa/catalog.json'] == 'addressables_catalog'


@check
def a_merkle_root_is_order_independent_and_content_sensitive():
    pairs = [('b/x', 'h2'), ('a/y', 'h1'), ('c/z', 'h3')]
    assert inventory.merkle_root(pairs) == inventory.merkle_root(reversed(pairs))
    assert inventory.merkle_root(pairs) != inventory.merkle_root(
        [('b/x', 'h2'), ('a/y', 'h1'), ('c/z', 'h4')])
    # Path is bound into the leaf, so moving a file changes the root even though
    # its content did not.
    assert inventory.merkle_root(pairs) != inventory.merkle_root(
        [('b/x', 'h2'), ('a/MOVED', 'h1'), ('c/z', 'h3')])
    assert inventory.merkle_root([]) == inventory.merkle_root([])


@check
def scalars_flattens_a_typetree():
    flat = static_survey.scalars(
        {'m_Gravity': {'x': 0.0, 'y': -9.81, 'z': 0.0},
         'm_DefaultSolverIterations': 6,
         'm_Nested': [{'a': 1}, {'a': 2}]})
    assert flat['m_Gravity.y'] == -9.81, flat
    assert flat['m_DefaultSolverIterations'] == 6
    assert flat['m_Nested[1].a'] == 2, flat


def _write(path, source, fp='fp0', jitter=0.0, hit_tick=10, channels=None,
           n=40, extra_channel=None):
    channels = channels or ['root_x', 'root_yaw']
    if extra_channel:
        channels = channels + [extra_channel]
    with trace_mod.TraceWriter(path, channels, fp, source, experiment='e1', seed=7) as w:
        for t in range(n):
            vals = {'root_x': 0.05 * t + jitter * t,
                    'root_yaw': 0.01 * t}
            if extra_channel:
                vals[extra_channel] = 0.0
            w.append(t, vals)
        w.event(hit_tick, 'hit', by=0)
        w.event(n - 1, 'round_end', winner=0)


@check
def a_trace_round_trips():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'a.trace'
        _write(p, 'rek')
        t = trace_mod.Trace.load(p)
        assert t.source == 'rek' and t.build_fingerprint == 'fp0'
        assert len(t) == 40 and t.ticks[:3] == [0, 1, 2]
        assert abs(t.channels['root_x'][10] - 0.5) < 1e-12
        assert t.header['seed'] == 7
        assert [e['kind'] for e in t.events] == ['hit', 'round_end']


@check
def a_trace_must_name_its_build_and_source():
    with tempfile.TemporaryDirectory() as d:
        for kwargs in ({'build_fingerprint': '', 'source': 'rek'},
                       {'build_fingerprint': 'fp0', 'source': 'whatever'}):
            try:
                trace_mod.TraceWriter(Path(d) / 'x.trace', ['a'], **kwargs)
            except ValueError:
                continue
            raise AssertionError(f'accepted a trace with {kwargs}')


@check
def a_server_authoritative_trace_must_identify_the_server():
    # The client hash pins the client. If the simulation is remote, the server
    # can be updated independently, so two traces from one client build may come
    # from different authoritative simulators.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        try:
            trace_mod.TraceWriter(d / 'x.trace', ['a'], 'fp0', 'rek',
                                  authority='server')
        except ValueError as e:
            assert 'does not' in str(e) and 'session_id' in str(e), e
        else:
            raise AssertionError('accepted a server trace with no server identity')

        with trace_mod.TraceWriter(
                d / 'ok.trace', ['a'], 'fp0', 'rek', authority='server',
                server={'endpoint': 'eu-1.example:7777', 'session_id': 'abc123',
                        'protocol_version': 7, 'server_version': None}) as w:
            w.append(0, {'a': 1.0})
        t = trace_mod.Trace.load(d / 'ok.trace')
        assert t.authority == 'server' and t.server['session_id'] == 'abc123'

        # A local trace needs none of that, and defaults to declaring nothing.
        with trace_mod.TraceWriter(d / 'loc.trace', ['a'], 'fp0', 'rek',
                                   authority='local') as w:
            w.append(0, {'a': 1.0})
        assert trace_mod.Trace.load(d / 'loc.trace').authority == 'local'
        assert trace_mod.Trace.load(d / 'ok.trace').header['channels'] == ['a']


@check
def frames_must_be_complete_and_ordered():
    with tempfile.TemporaryDirectory() as d:
        w = trace_mod.TraceWriter(Path(d) / 'x.trace', ['a', 'b'], 'fp0', 'rek')
        try:
            w.append(0, {'a': 1.0})
        except ValueError:
            pass
        else:
            raise AssertionError('accepted a frame missing a channel')
        w.append(5, {'a': 1.0, 'b': 2.0})
        try:
            w.append(5, {'a': 1.0, 'b': 2.0})
        except ValueError:
            pass
        else:
            raise AssertionError('accepted a non-increasing tick')
        w.close()


def _write_series(path, source, series, events=(), fp='fp0', channel='root_x'):
    """A trace with an explicit per-tick series, for shaping distributions."""
    with trace_mod.TraceWriter(path, [channel], fp, source, experiment='e1') as w:
        for tick, value in series:
            w.append(tick, {channel: value})
        for tick, kind in events:
            w.event(tick, kind)


def _run(args, cwd):
    return subprocess.run([sys.executable, str(HERE / 'differ.py')] + args,
                          capture_output=True, text=True, cwd=cwd)


@check
def baseline_measures_reks_spread_and_compare_gates_on_it():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # Three REK runs of the same experiment, differing slightly.
        _write(d / 'r0.trace', 'rek', jitter=0.000)
        _write(d / 'r1.trace', 'rek', jitter=0.001)
        _write(d / 'r2.trace', 'rek', jitter=0.002, hit_tick=11)
        p = _run(['baseline', 'r0.trace', 'r1.trace', 'r2.trace', '--out', 'env.json'], d)
        assert p.returncode == 0, p.stderr
        env = json.loads((d / 'env.json').read_text())
        # Widest pairwise spread on root_x is 0.002 * 39.
        assert abs(env['channels']['root_x']['max'] - 0.002 * 39) < 1e-9, env['channels']
        assert env['events']['hit']['max_tick_jitter'] == 1, env['events']

        # A clone inside that spread passes.
        _write(d / 'good.trace', 'clone:reference', jitter=0.001, hit_tick=11)
        p = _run(['compare', 'r0.trace', 'good.trace', '--envelope', 'env.json',
                  '--report', 'good.json'], d)
        assert p.returncode == 0, p.stdout + p.stderr
        assert 'PARITY PASS' in p.stdout

        # One outside it fails, and says where it first left.
        _write(d / 'bad.trace', 'clone:reference', jitter=0.02, hit_tick=25)
        p = _run(['compare', 'r0.trace', 'bad.trace', '--envelope', 'env.json',
                  '--report', 'bad.json'], d)
        assert p.returncode == 1, p.stdout
        assert 'PARITY FAIL' in p.stdout
        report = json.loads((d / 'bad.json').read_text())
        assert 'root_x' in report['failures'] and 'event:hit' in report['failures']
        assert report['first_divergent_tick'] is not None


@check
def a_deterministic_oracle_demands_exact_equality():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name in ('r0.trace', 'r1.trace'):
            _write(d / name, 'rek', jitter=0.0)
        p = _run(['baseline', 'r0.trace', 'r1.trace', '--out', 'env.json'], d)
        assert 'deterministically' in p.stdout, p.stdout
        env = json.loads((d / 'env.json').read_text())
        assert env['channels']['root_x']['max'] == 0.0

        # With a zero envelope even a tiny error is a failure, which is the
        # stronger result and must not be softened.
        _write(d / 'near.trace', 'clone:reference', jitter=1e-9)
        p = _run(['compare', 'r0.trace', 'near.trace', '--envelope', 'env.json',
                  '--report', 'r.json'], d)
        assert p.returncode == 1, p.stdout


@check
def a_clone_that_omits_state_cannot_pass():
    # The failure mode that matters: a reduced-order model simply not recording
    # the joint state it never had. Silence must not read as agreement.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write(d / 'r0.trace', 'rek', extra_channel='joint_knee_l')
        _write(d / 'r1.trace', 'rek', extra_channel='joint_knee_l')
        _run(['baseline', 'r0.trace', 'r1.trace', '--out', 'env.json'], d)
        _write(d / 'thin.trace', 'clone:proxy')          # no joint channel
        p = _run(['compare', 'r0.trace', 'thin.trace', '--envelope', 'env.json',
                  '--report', 'r.json'], d)
        assert p.returncode == 1, p.stdout
        report = json.loads((d / 'r.json').read_text())
        assert report['channels_missing_from_clone'] == ['joint_knee_l'], report
        assert not report['passed']


@check
def traces_from_different_builds_cannot_be_mixed():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write(d / 'r0.trace', 'rek', fp='fp_old')
        _write(d / 'r1.trace', 'rek', fp='fp_new')
        p = _run(['baseline', 'r0.trace', 'r1.trace', '--out', 'env.json'], d)
        assert p.returncode != 0 and 'different builds' in p.stdout + p.stderr


@check
def a_clone_trace_cannot_masquerade_as_the_oracle():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write(d / 'c0.trace', 'clone:proxy')
        _write(d / 'c1.trace', 'clone:proxy')
        p = _run(['baseline', 'c0.trace', 'c1.trace', '--out', 'env.json'], d)
        assert p.returncode != 0, p.stdout
        assert 'must be REK' in p.stdout + p.stderr


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 0.5
        return self.t


def authority_run(online_peers, blocked_peers, marks, blocked_established=None):
    """Replay an authority experiment without a game or a network."""
    clock = FakeClock()
    peers = {'baseline_online': online_peers, 'blocked': blocked_peers,
             'restored': online_peers}
    state = {'phase': 'baseline_online'}

    def sample():
        ps = peers[state['phase']]
        est = (blocked_established if state['phase'] == 'blocked'
               and blocked_established is not None else len(ps))
        return ([{'raddr': a, 'status': 'ESTABLISHED' if i < est else 'CLOSE_WAIT'}
                 for i, a in enumerate(ps)], [])

    run = authority.Run(sample, clock)
    for _ in range(4):
        run.observe()
    run.set_phase('blocked')
    state['phase'] = 'blocked'
    for kind in marks:
        run.mark(kind)
    for _ in range(4):
        run.observe()
    run.set_phase('restored')
    state['phase'] = 'restored'
    run.observe()
    return run.report()


@check
def a_block_that_did_not_apply_yields_no_verdict():
    # The failure that would otherwise manufacture evidence: the firewall rule
    # missed the process, the game kept talking, and the marks get read as if
    # the network had been cut.
    report = authority_run(['1.2.3.4:443'], ['1.2.3.4:443'],
                           ['input', 'state-progressed', 'score-ok'])
    v = authority.interpret(report)
    assert v['verdict'] == 'inconclusive', v
    assert not v['block']['effective']
    assert 'still established' in v['block']['reason']


@check
def a_game_with_no_connections_cannot_be_tested_this_way():
    report = authority_run([], [], ['input', 'state-progressed'])
    v = authority.interpret(report)
    assert v['verdict'] == 'inconclusive', v
    assert 'no remote connections even before' in v['block']['reason']


@check
def state_continuing_with_the_network_cut_means_local_authority():
    report = authority_run(['1.2.3.4:443'], [],
                           ['input', 'state-progressed', 'reset-ok', 'score-ok'])
    v = authority.interpret(report)
    assert v['block']['effective'], v['block']
    assert v['verdict'] == 'local_authority', v
    assert 'live matches' in v['limits']


@check
def progress_without_a_completed_interaction_is_only_weak_evidence():
    report = authority_run(['1.2.3.4:443'], [], ['input', 'state-progressed'])
    v = authority.interpret(report)
    assert v['verdict'] == 'local_authority_weak', v
    assert 'reset' in v['because']


@check
def freezing_when_cut_off_means_remote_authority():
    report = authority_run(['1.2.3.4:443'], [], ['input', 'frozen', 'reset-failed'])
    v = authority.interpret(report)
    assert v['verdict'] == 'remote_authority', v


@check
def progress_then_correction_means_local_prediction():
    report = authority_run(['1.2.3.4:443'], [],
                           ['input', 'state-progressed', 'rollback'])
    v = authority.interpret(report)
    assert v['verdict'] == 'local_prediction_remote_correction', v


@check
def an_unmarked_blocked_phase_is_inconclusive():
    report = authority_run(['1.2.3.4:443'], [], [])
    v = authority.interpret(report)
    assert v['verdict'] == 'inconclusive', v
    assert 'nothing was marked' in v['because']


@check
def one_anomalous_rek_run_must_not_widen_the_tolerance():
    # The defect this guards: accepting at the maximum lets a single outlier run
    # buy slack for the clone. Three runs agree closely; a fourth has one large
    # excursion. A clone that is wrong everywhere by less than that excursion
    # must still fail.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        flat = [(t, 0.0) for t in range(200)]
        for name in ('r0.trace', 'r1.trace', 'r2.trace'):
            _write_series(d / name, 'rek', flat)
        spike = [(t, 5.0 if t == 100 else 0.0) for t in range(200)]
        _write_series(d / 'r3.trace', 'rek', spike)

        p = _run(['baseline', 'r0.trace', 'r1.trace', 'r2.trace', 'r3.trace',
                  '--out', 'p99.json'], d)
        assert p.returncode == 0, p.stderr
        env = json.loads((d / 'p99.json').read_text())
        ch = env['channels']['root_x']
        assert ch['max'] == 5.0 and ch['p99'] < 5.0, ch
        assert ch['outlier_ratio'] > 10, ch
        assert 'more than 10x their p99' in p.stdout, p.stdout

        # Wrong by 1.0 everywhere: well inside the max, well outside the p99.
        _write_series(d / 'clone.trace', 'clone:x', [(t, 1.0) for t in range(200)])
        p99_run = _run(['compare', 'r0.trace', 'clone.trace',
                        '--envelope', 'p99.json', '--report', 'a.json'], d)
        assert p99_run.returncode == 1, p99_run.stdout

        _run(['baseline', 'r0.trace', 'r1.trace', 'r2.trace', 'r3.trace',
              '--accept-at', 'max', '--out', 'max.json'], d)
        max_run = _run(['compare', 'r0.trace', 'clone.trace',
                        '--envelope', 'max.json', '--report', 'b.json'], d)
        assert max_run.returncode == 0, max_run.stdout
        # Which is the point: --accept-at max is the permissive setting, and the
        # default must not be it.
        assert json.loads((d / 'p99.json').read_text())['accept_at'] == 'p99'


@check
def a_dropped_event_does_not_cascade_into_timing_errors():
    # Zipping two event lists pairs unrelated occurrences as soon as one is
    # missing, turning one dropped hit into an error on every hit after it.
    m = differ.match_events([10, 20, 30, 40], [10, 30, 40], window=1)
    assert m['matched'] == 3 and m['false_negatives'] == 1, m
    assert m['false_positives'] == 0 and m['max_matched_offset'] == 0, m
    assert m['recall'] == 0.75 and m['precision'] == 1.0, m

    spurious = differ.match_events([10, 20], [10, 15, 20], window=1)
    assert spurious['false_positives'] == 1 and spurious['recall'] == 1.0, spurious
    assert spurious['precision'] < 1.0, spurious

    # Outside the window is not a match at all, however close the count.
    late = differ.match_events([10], [14], window=1)
    assert late['matched'] == 0 and late['precision'] == 0.0, late


@check
def compare_reports_how_much_of_the_run_was_still_informative():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        flat = [(t, 0.0) for t in range(100)]
        for name in ('r0.trace', 'r1.trace', 'r2.trace'):
            _write_series(d / name, 'rek', flat)
        _run(['baseline', 'r0.trace', 'r1.trace', 'r2.trace', '--out', 'env.json'], d)

        # Tracks exactly, then leaves at tick 40 and stays gone.
        drift = [(t, 0.0 if t < 40 else 1.0) for t in range(100)]
        _write_series(d / 'clone.trace', 'clone:x', drift)
        p = _run(['compare', 'r0.trace', 'clone.trace', '--envelope', 'env.json',
                  '--report', 'r.json'], d)
        assert p.returncode == 1, p.stdout
        rep = json.loads((d / 'r.json').read_text())
        assert rep['first_divergent_tick'] == 40, rep['first_divergent_tick']
        assert rep['valid_horizon_ticks'] == 40, rep['valid_horizon_ticks']
        assert 'past the divergence point' in p.stdout, p.stdout


@check
def short_horizon_scores_windows_independently():
    # The point of the mode: one divergence should cost you the window it is in,
    # not every tick after it. Open-loop cannot distinguish those.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        anchors = [(t, 'reset') for t in (0, 50, 100, 150)]
        flat = [(t, 0.0) for t in range(200)]
        for name in ('r0.trace', 'r1.trace', 'r2.trace'):
            _write_series(d / name, 'rek', flat, anchors)
        _run(['baseline', 'r0.trace', 'r1.trace', 'r2.trace', '--out', 'env.json'], d)

        # Wrong only inside the third window.
        drift = [(t, 1.0 if 100 <= t < 130 else 0.0) for t in range(200)]
        _write_series(d / 'clone.trace', 'clone:x', drift, anchors)

        p = _run(['compare', 'r0.trace', 'clone.trace', '--envelope', 'env.json',
                  '--mode', 'short-horizon', '--window', '30',
                  '--report', 'sh.json'], d)
        assert p.returncode == 1, p.stdout
        rep = json.loads((d / 'sh.json').read_text())
        sh = rep['short_horizon']
        assert sh['windows_tested'] == 4 and sh['windows_passed'] == 3, sh
        assert sh['pass_fraction'] == 0.75, sh
        failed = [w['anchor_tick'] for w in sh['windows'] if not w['passed']]
        assert failed == [100], failed
        assert rep['failures'] == ['window@100'], rep['failures']

        # Same data open-loop: one verdict for the whole run, and the tail after
        # tick 100 carries no information.
        p = _run(['compare', 'r0.trace', 'clone.trace', '--envelope', 'env.json',
                  '--report', 'ol.json'], d)
        assert p.returncode == 1
        ol = json.loads((d / 'ol.json').read_text())
        assert ol['first_divergent_tick'] == 100
        assert ol['short_horizon'] is None


@check
def short_horizon_refuses_to_run_without_anchors():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        flat = [(t, 0.0) for t in range(60)]
        for name in ('r0.trace', 'r1.trace'):
            _write_series(d / name, 'rek', flat)          # no reset/inject events
        _run(['baseline', 'r0.trace', 'r1.trace', '--out', 'env.json'], d)
        _write_series(d / 'clone.trace', 'clone:x', flat)
        p = _run(['compare', 'r0.trace', 'clone.trace', '--envelope', 'env.json',
                  '--mode', 'short-horizon', '--report', 'r.json'], d)
        assert p.returncode != 0, p.stdout
        # Silently falling back to open-loop would be the dangerous behaviour.
        assert 'nothing to start a window from' in p.stdout + p.stderr


@check
def channel_units_are_carried_through():
    assert differ.channel_unit('root.0.pos.x') == 'm'
    assert differ.channel_unit('root.0.angvel.z') == 'rad/s'
    assert differ.channel_unit('joint.0.knee_l.pos') == 'rad'
    assert differ.channel_unit('joint.0.knee_l.vel') == 'rad/s'
    assert differ.channel_unit('contact.0.foot_l.impulse') == 'N*s'
    assert differ.channel_unit('ctrl.0.hidden[3]') == 'dimensionless'


@check
def quantiles_are_interpolated_not_nearest():
    assert differ.quantile([0.0, 1.0], 0.5) == 0.5
    assert differ.quantile([0.0, 10.0], 0.99) == 9.9
    assert differ.quantile([5.0], 0.99) == 5.0
    assert differ.quantile([], 0.5) == 0.0


def _states(results):
    return {r['artifact']: r['state'] for r in results}


@check
def an_empty_directory_is_honestly_reported_as_no_evidence():
    with tempfile.TemporaryDirectory() as d:
        results = check_artifacts.check(Path(d))
        assert check_artifacts.stage_of(results) == 'no evidence'
        assert all(r['state'] == 'MISSING' for r in results), results


@check
def the_gate_advances_one_stage_at_a_time():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        game = fake_install(d / 'install')
        inv = inventory.scan(game)
        (d / 'inventory.json').write_text(json.dumps(inv))
        results = check_artifacts.check(d)
        assert _states(results)['inventory.json'] == 'PRESENT'
        assert check_artifacts.stage_of(results) == 'build pinned'

        # An inconclusive authority run does not advance anything — that is the
        # whole point of refusing a verdict when the block failed.
        (d / 'authority_practice.json').write_text(json.dumps(
            {'verdict': {'verdict': 'inconclusive', 'because': 'block failed'}}))
        results = check_artifacts.check(d)
        assert _states(results)['authority test'] == 'MISSING', _states(results)
        assert check_artifacts.stage_of(results) == 'build pinned'

        (d / 'authority_practice.json').write_text(json.dumps(
            {'verdict': {'verdict': 'local_authority', 'because': 'kept stepping'}}))
        (d / 'static_survey.json').write_text(json.dumps(
            {'build_fingerprint': inv['build_fingerprint'], 'bodies': [],
             'model_assets': [], 'settings': {}}))
        results = check_artifacts.check(d)
        assert check_artifacts.stage_of(results) == 'statics surveyed', results


@check
def artifacts_describing_different_builds_are_refused():
    # Traces from one client build and a survey from another cannot be reasoned
    # about together, and nothing else in the pipeline would notice.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        game = fake_install(d / 'install')
        inv = inventory.scan(game)
        (d / 'inventory.json').write_text(json.dumps(inv))
        (d / 'static_survey.json').write_text(json.dumps(
            {'build_fingerprint': 'a-completely-different-build',
             'bodies': [], 'model_assets': [], 'settings': {}}))
        results = check_artifacts.check(d)
        assert _states(results)['build agreement'] == 'INCONSISTENT', results
        assert 'different builds' in check_artifacts.stage_of(results)


@check
def two_rek_runs_are_not_enough_for_an_envelope():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write(d / 'r0.trace', 'rek')
        _write(d / 'r1.trace', 'rek')
        results = check_artifacts.check(d)
        assert _states(results)['REK traces'] == 'INCOMPLETE', results
        assert 'at least 3' in [r for r in results
                                if r['artifact'] == 'REK traces'][0]['detail']


def main() -> int:
    failed = 0
    for fn in checks:
        try:
            fn()
            print(f'  ok    {fn.__name__.replace("_", " ")}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL  {fn.__name__.replace("_", " ")}: {e}')
    print(f'\n{len(checks) - failed}/{len(checks)} checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
