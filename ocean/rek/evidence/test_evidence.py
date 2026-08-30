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
il2cpp = _load('il2cpp_probe')
collect_mod = _load('collect')

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
def scanning_reports_progress_and_survives_unreadable_files():
    seen = []
    with tempfile.TemporaryDirectory() as d:
        game = fake_install(Path(d))
        (game / 'REK_Data' / 'locked.assets').write_bytes(b'x')

        # A running game holds locks on Windows, so some files will refuse to
        # open. Patched rather than chmod'ed because this suite may run as root,
        # where permissions are advisory.
        real = inventory.sha256

        def refusing(path):
            if path.name == 'locked.assets':
                raise PermissionError(32, 'file in use by another process')
            return real(path)

        inventory.sha256 = refusing
        try:
            inv = inventory.scan(game, progress=lambda n, b: seen.append((n, b)))
        finally:
            inventory.sha256 = real

    assert inv['errors'], 'an unreadable file was silently dropped'
    assert inv['errors'][0]['path'] == 'REK_Data/locked.assets', inv['errors']
    assert 'in use' in inv['errors'][0]['error']

    # It stays in the manifest, marked unreadable. Dropping it would compute the
    # identity over a subset, so the same install scanned again with the game
    # closed would disagree and nothing would say why.
    entry = [f for f in inv['files'] if f['path'].endswith('locked.assets')]
    assert entry and entry[0]['unreadable'] and entry[0]['sha256'] is None, entry

    # And the incomplete read is visible as an identity that will not reproduce.
    with tempfile.TemporaryDirectory() as d2:
        d2 = Path(d2)
        (d2 / 'inventory.json').write_text(json.dumps(inv))
        results = check_artifacts.check(d2)
        assert _states(results)['inventory.json'] == 'INCOMPLETE', results
        assert 'will not reproduce' in [
            r for r in results if r['artifact'] == 'inventory.json'][0]['detail']

    # progress is time-based, so on a tiny fixture it may legitimately never
    # fire; what matters is that passing it does not break the scan.
    assert all(isinstance(n, int) for n, _ in seen)
    # Everything else still hashed.
    assert any(f['path'] == 'GameAssembly.dll' for f in inv['files'])

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


def _prov(channels, source):
    """REK channels must cite a source; a clone's come from its own code."""
    if source != 'rek':
        return None
    return {c: {'kind': 'method', 'ref': f'RobotState.Get_{c}'} for c in channels}


def _write(path, source, fp='fp0', jitter=0.0, hit_tick=10, channels=None,
           n=40, extra_channel=None):
    channels = channels or ['root_x', 'root_yaw']
    if extra_channel:
        channels = channels + [extra_channel]
    with trace_mod.TraceWriter(path, channels, fp, source, experiment='e1', seed=7,
                               provenance=_prov(channels, source)) as w:
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
                trace_mod.TraceWriter(Path(d) / 'x.trace', ['a'],
                                      provenance=_prov(['a'], 'rek'), **kwargs)
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
                                  authority='server',
                                  provenance=_prov(['a'], 'rek'))
        except ValueError as e:
            assert 'does not' in str(e) and 'session_id' in str(e), e
        else:
            raise AssertionError('accepted a server trace with no server identity')

        with trace_mod.TraceWriter(
                d / 'ok.trace', ['a'], 'fp0', 'rek', authority='server',
                provenance=_prov(['a'], 'rek'),
                server={'endpoint': 'eu-1.example:7777', 'session_id': 'abc123',
                        'protocol_version': 7, 'server_version': None}) as w:
            w.append(0, {'a': 1.0})
        t = trace_mod.Trace.load(d / 'ok.trace')
        assert t.authority == 'server' and t.server['session_id'] == 'abc123'

        # A local trace needs none of that, and defaults to declaring nothing.
        with trace_mod.TraceWriter(d / 'loc.trace', ['a'], 'fp0', 'rek',
                                   authority='local',
                                   provenance=_prov(['a'], 'rek')) as w:
            w.append(0, {'a': 1.0})
        assert trace_mod.Trace.load(d / 'loc.trace').authority == 'local'
        assert trace_mod.Trace.load(d / 'ok.trace').header['channels'] == ['a']


@check
def a_rek_channel_must_cite_where_it_came_from():
    # The recorder is written after the survey and the control-path trace name
    # the real fields. This is what stops a channel being invented in between:
    # once it is in the file, a guessed channel is indistinguishable from a
    # measured one.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        try:
            trace_mod.TraceWriter(d / 'x.trace', ['root_x', 'balance'], 'fp0', 'rek')
        except ValueError as e:
            assert 'no provenance' in str(e) and 'balance' in str(e), e
        else:
            raise AssertionError('accepted an uncited REK channel')

        # Half-cited is still refused, and names which half.
        try:
            trace_mod.TraceWriter(
                d / 'x.trace', ['root_x', 'balance'], 'fp0', 'rek',
                provenance={'root_x': {'kind': 'method', 'ref': 'X.GetRoot'}})
        except ValueError as e:
            assert "['balance']" in str(e), e
        else:
            raise AssertionError('accepted a partially cited REK trace')

        # A citation has to be one of the recognised kinds, with a real ref.
        for cite, why in (({'kind': 'seemed obvious', 'ref': 'x'}, 'kind must be'),
                          ({'kind': 'method', 'ref': '  '}, 'ref is empty'),
                          ({'kind': 'method'}, 'ref is empty'),
                          ('RobotState.GetRoot', 'citation must be')):
            try:
                trace_mod.TraceWriter(d / 'x.trace', ['root_x'], 'fp0', 'rek',
                                      provenance={'root_x': cite})
            except ValueError as e:
                assert why in str(e), (cite, e)
            else:
                raise AssertionError(f'accepted citation {cite!r}')

        # Citing something that is not a declared channel is a mistake too.
        try:
            trace_mod.TraceWriter(
                d / 'x.trace', ['root_x'], 'fp0', 'rek',
                provenance={'root_x': {'kind': 'method', 'ref': 'X.GetRoot'},
                            'ghost': {'kind': 'method', 'ref': 'X.Nothing'}})
        except ValueError as e:
            assert 'not declared as a channel' in str(e), e
        else:
            raise AssertionError('accepted provenance for an undeclared channel')

        # Properly cited: written, and the citations survive the round trip.
        with trace_mod.TraceWriter(
                d / 'ok.trace', ['root_x'], 'fp0', 'rek',
                provenance={'root_x': {'kind': 'serialized_field',
                                       'ref': 'ArticulationBody.m_AnchorPosition.x'}}) as w:
            w.append(0, {'root_x': 1.0})
        t = trace_mod.Trace.load(d / 'ok.trace')
        assert t.provenance['root_x']['kind'] == 'serialized_field'
        assert 'AnchorPosition' in t.provenance['root_x']['ref']

        # A clone needs no citation — its channels come from its own source.
        with trace_mod.TraceWriter(d / 'c.trace', ['root_x'], 'fp0',
                                   'clone:x') as w:
            w.append(0, {'root_x': 1.0})
        assert trace_mod.Trace.load(d / 'c.trace').provenance == {}


@check
def frames_must_be_complete_and_ordered():
    with tempfile.TemporaryDirectory() as d:
        w = trace_mod.TraceWriter(Path(d) / 'x.trace', ['a', 'b'], 'fp0', 'rek',
                                  provenance=_prov(['a', 'b'], 'rek'))
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
    with trace_mod.TraceWriter(path, [channel], fp, source, experiment='e1',
                               provenance=_prov([channel], source)) as w:
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
def a_block_command_that_failed_is_decisive_on_its_own():
    # Sockets can look convincing for other reasons. If the command meant to
    # apply the block exited non-zero, the intervention did not happen and no
    # amount of socket evidence should override that.
    report = authority_run(['1.2.3.4:443'], [],
                           ['input', 'state-progressed', 'reset-ok'])
    report['commands'] = [{'t': 2.0, 'phase': 'blocked',
                           'cmd': 'netsh advfirewall ...', 'returncode': 1}]
    v = authority.interpret(report)
    assert v['verdict'] == 'inconclusive', v
    assert 'exited 1' in v['block']['reason'], v['block']

    # Exit 0 leaves the socket evidence to decide.
    report['commands'][0]['returncode'] = 0
    v = authority.interpret(report)
    assert v['verdict'] == 'local_authority', v


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


class LCG:
    """Deterministic pseudo-random, so these checks never flake."""
    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (1103515245 * self.s + 12345) % (1 << 31)
        return self.s / (1 << 31)


def _episodes(directory, prefix, source, n, shift=0.0, hits=lambda r: 3, seed=1):
    rng = LCG(seed)
    paths = []
    for i in range(n):
        r = rng.next()
        path = directory / f'{prefix}{i}.trace'
        value = r + shift
        _write_series(path, source, [(t, value) for t in range(50)],
                      [(t * 10 + 5, 'hit') for t in range(hits(r))])
        paths.append(str(path.name))
    return paths


@check
def distributional_mode_compares_outcomes_not_trajectories():
    # The question that survives chaos: over many episodes, do the two produce
    # the same distribution of outcomes? Trajectory matching cannot ask it.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rek = _episodes(d, 'r', 'rek', 24, seed=7)
        same = _episodes(d, 'c', 'clone:x', 24, seed=99)
        p = _run(['distributional', '--rek', *rek, '--clone', *same,
                  '--report', 'ok.json'], d)
        assert p.returncode == 0, p.stdout
        assert 'DISTRIBUTIONAL PASS' in p.stdout

        # Same shape, shifted. Every trajectory is individually plausible and
        # the distribution is wrong.
        off = _episodes(d, 'b', 'clone:x', 24, shift=2.0, seed=99)
        p = _run(['distributional', '--rek', *rek, '--clone', *off,
                  '--report', 'bad.json'], d)
        assert p.returncode == 1, p.stdout
        rep = json.loads((d / 'bad.json').read_text())
        assert 'final:root_x' in rep['failures'], rep['failures']
        assert rep['statistics']['final:root_x']['ks'] == 1.0


@check
def distributional_mode_refuses_too_few_episodes():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rek = _episodes(d, 'r', 'rek', 3, seed=7)
        clone = _episodes(d, 'c', 'clone:x', 3, seed=8)
        p = _run(['distributional', '--rek', *rek, '--clone', *clone], d)
        assert p.returncode != 0 and 'is not a distribution' in p.stdout + p.stderr

        # And with enough to run but not enough to mean much, it says so.
        rek = _episodes(d, 'q', 'rek', 6, seed=7)
        clone = _episodes(d, 'w', 'clone:x', 6, seed=8)
        p = _run(['distributional', '--rek', *rek, '--clone', *clone,
                  '--report', 'weak.json'], d)
        assert 'not yet contradicted' in p.stdout, p.stdout


@check
def a_clone_that_never_produces_an_event_is_caught():
    # Silence again: a clone that simply never emits a hit must not pass by
    # having no distribution to disagree with.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rek = _episodes(d, 'r', 'rek', 12, seed=7)
        for i in range(12):
            _write_series(d / f'n{i}.trace', 'clone:x',
                          [(t, 0.5) for t in range(50)], [])
        p = _run(['distributional', '--rek', *rek,
                  '--clone', *[f'n{i}.trace' for i in range(12)],
                  '--report', 'silent.json'], d)
        assert p.returncode == 1, p.stdout
        rep = json.loads((d / 'silent.json').read_text())
        assert 'first_tick:hit' in rep['statistics_missing_from_clone'], rep
        assert not rep['passed']


@check
def the_ks_statistic_behaves():
    assert differ.ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0
    assert differ.ks_statistic([1, 2, 3], [10, 11, 12]) == 1.0
    assert 0 < differ.ks_statistic([1, 2, 3, 4], [3, 4, 5, 6]) < 1.0
    assert differ.ks_statistic([], [1]) == 1.0


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


class FakeType:
    def __init__(self, name):
        self.name = name


class FakeAssetsFile:
    unity_version = '2022.3.40f1'


class FakeGameObject:
    def __init__(self, name):
        self.m_Name = name


class FakeRead:
    """What obj.read() returns — only m_GameObject is used, for collider owners."""
    def __init__(self, owner):
        self._owner = owner

    @property
    def m_GameObject(self):
        if self._owner is None:
            raise AttributeError('no GameObject')
        return self

    def read(self):
        return FakeGameObject(self._owner)


class FakeObject:
    def __init__(self, type_name, tree, owner=None):
        self.type = FakeType(type_name)
        self._tree = tree
        self._owner = owner
        self.assets_file = FakeAssetsFile()

    def read_typetree(self):
        return self._tree

    def read(self):
        return FakeRead(self._owner)


class FakeBundle:
    def __init__(self, objects):
        self.objects = objects


def install_fake_unitypy(objects):
    """Inject a stub UnityPy so static_survey's scan loop can be exercised.

    static_survey imports UnityPy inside survey(), so replacing the module here
    is enough. Without this the whole scan loop — the part that runs on the
    machine we cannot reach — would never execute until it ran there.
    """
    import types
    mod = types.ModuleType('UnityPy')
    mod.load = lambda path: FakeBundle(objects)
    sys.modules['UnityPy'] = mod


REK_OBJECTS = [
    FakeObject('TimeManager', {'Fixed Timestep': 0.013888889,
                               'Maximum Allowed Timestep': 0.1,
                               'm_TimeScale': 1.0}),
    FakeObject('PhysicsManager', {'m_Gravity': {'x': 0.0, 'y': -9.81, 'z': 0.0},
                                  'm_DefaultSolverIterations': 12,
                                  'm_DefaultContactOffset': 0.01}),
    FakeObject('Avatar', {'m_Name': 'L100Avatar',
                          'm_TOS': [{'first': 1, 'second': 'mixamorig:Hips'},
                                    {'first': 2, 'second': 'mixamorig:LeftHand'}]}),
    FakeObject('ArticulationBody', {'m_Mass': 3.2, 'm_LinearDamping': 0.05,
                                    'm_XDrive': {'stiffness': 800.0,
                                                 'damping': 40.0}},
               owner='left_shoulder'),
    FakeObject('CapsuleCollider', {'m_Radius': 0.085, 'm_Height': 0.2,
                                   'm_Center': {'x': 0, 'y': 0, 'z': 0}},
               owner='mixamorig:RightHand'),
    FakeObject('AnimationClip', {'m_Name': 'Jab', 'm_SampleRate': 30.0,
                                 'm_MuscleClip': {'m_StopTime': 0.7333},
                                 'm_Events': [{'time': 0.2,
                                               'functionName': 'HitboxOn',
                                               'data': 'RightHand'}]}),
    FakeObject('NNModel', {'m_Name': 'balance_policy', 'm_ModelData': {}}),
    FakeObject('MonoBehaviour', {'m_Name': 'MatchRules', 'roundSeconds': 60,
                                 'downsToLose': 3, 'scoreOnHit': 1}),
]


@check
def the_survey_scan_loop_actually_runs():
    install_fake_unitypy(REK_OBJECTS)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inv = {'build_fingerprint': 'fp0', 'steam': {'buildid': '19284412'},
               'files': [{'path': 'REK_Data/globalgamemanagers',
                          'kind': 'unity_settings', 'sha256': 'x', 'size': 1},
                         {'path': 'REK_Data/StreamingAssets/balance.onnx',
                          'kind': 'model_asset', 'sha256': 'y', 'size': 2},
                         {'path': 'REK_Data/Plugins/x86_64/physx.dll',
                          'kind': 'native_plugin', 'sha256': 'z', 'size': 3}]}
        r = static_survey.survey(d, inv, d / 'survey.json')

    assert r['build_fingerprint'] == 'fp0' and r['steam_buildid'] == '19284412'
    assert r['unity_version'] == '2022.3.40f1'

    # The tick rate, read rather than assumed. This is the value the discarded
    # model guessed at.
    tm = r['settings']['TimeManager']['values']
    assert abs(tm['Fixed Timestep'] - 0.013888889) < 1e-9, tm
    pm = r['settings']['PhysicsManager']['values']
    assert pm['m_Gravity.y'] == -9.81 and pm['m_DefaultSolverIterations'] == 12

    body = r['bodies'][0]
    assert body['type'] == 'ArticulationBody' and body['owner'] == 'left_shoulder'
    assert body['values']['m_XDrive.stiffness'] == 800.0, body['values']

    col = r['colliders'][0]
    assert col['owner'] == 'mixamorig:RightHand' and col['values']['m_Radius'] == 0.085

    clip = r['animation_clips'][0]
    assert clip['name'] == 'Jab' and abs(clip['duration_s'] - 0.7333) < 1e-6
    assert clip['event_count'] == 1
    assert clip['events'][0]['function'] == 'HitboxOn'

    assert r['rig']['library'] == 'Adobe Mixamo', r['rig']

    # Files from the inventory come through without needing a container.
    sources = {m.get('source') for m in r['model_assets']}
    assert sources == {'file', 'unity_object'}, r['model_assets']
    assert len(r['native_code']) == 1

    assert any(h['name'] == 'MatchRules' for h in r['name_hits']), r['name_hits']
    assert r['not_recoverable_statically']


@check
def nothing_static_is_ever_marked_authoritative():
    # The invariant the whole taxonomy exists for. Presence in the build is not
    # participation in the transition function, and no static reading may
    # promote a record to a finding.
    install_fake_unitypy(REK_OBJECTS)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inv = {'build_fingerprint': 'fp0', 'steam': {},
               'files': [{'path': 'a.assets', 'kind': 'asset_container',
                          'sha256': 'x', 'size': 1}]}
        r = static_survey.survey(d, inv, d / 'survey.json')

    roles = []
    for key in ('bodies', 'colliders', 'model_assets', 'animation_clips',
                'name_hits'):
        roles += [rec.get('role') for rec in r[key]]
    roles += [v.get('role') for v in r['settings'].values()]
    roles.append(r['rig'].get('role'))

    assert roles, 'nothing was classified at all'
    assert all(role in static_survey.ROLES for role in roles), roles
    assert 'authoritative' not in roles, roles
    # Clips specifically: leads, and they carry the caution in the data.
    assert all(c['role'] == 'candidate_lead' for c in r['animation_clips'])
    assert all('not about any attack envelope' in c['caution']
               for c in r['animation_clips'])


class ExplodingObject(FakeObject):
    def read_typetree(self):
        raise KeyError('unknown field m_SomethingNew')


@check
def objects_that_cannot_be_read_are_reported_not_swallowed():
    # The dangerous failure: a schema mismatch on the real build silently
    # produces an empty survey, and "absent" is then read as "not present"
    # when it actually meant "could not be read".
    install_fake_unitypy([ExplodingObject('ArticulationBody', {}),
                          ExplodingObject('ArticulationBody', {}),
                          FakeObject('TimeManager', {'Fixed Timestep': 0.02})])
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inv = {'build_fingerprint': 'fp0', 'steam': {},
               'files': [{'path': 'a.assets', 'kind': 'asset_container',
                          'sha256': 'x', 'size': 1}]}
        r = static_survey.survey(d, inv, d / 'survey.json')

    assert r['bodies'] == []
    errs = r.get('read_errors')
    assert errs and errs[0]['type'] == 'ArticulationBody', errs
    assert errs[0]['count'] == 2 and 'KeyError' in errs[0]['example'], errs
    # The object that could be read still is.
    assert r['settings']['TimeManager']['values']['Fixed Timestep'] == 0.02


@check
def a_missing_time_manager_is_reported_absent_not_defaulted():
    install_fake_unitypy([FakeObject('MonoBehaviour', {'m_Name': 'Thing'})])
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inv = {'build_fingerprint': 'fp0', 'steam': {},
               'files': [{'path': 'a.assets', 'kind': 'asset_container',
                          'sha256': 'x', 'size': 1}]}
        r = static_survey.survey(d, inv, d / 'survey.json')
    missing = {a.get('missing') for a in r['absent']}
    assert 'TimeManager' in missing and 'PhysicsManager' in missing, r['absent']
    assert 'TimeManager' not in r['settings']
    # No tick rate is invented to stand in for the one that was not found.
    assert not any('timestep' in str(v).lower() for v in r['settings'].values())


def _il2cpp_install(tmp: Path, sanity=0xFAB11BAF, version=31, sentis=True):
    game = tmp / 'REK'
    (game / 'REK_Data' / 'il2cpp_data' / 'Metadata').mkdir(parents=True)
    (game / 'REK_Data' / 'Plugins' / 'x86_64').mkdir(parents=True)
    meta = game / 'REK_Data' / 'il2cpp_data' / 'Metadata' / 'global-metadata.dat'
    meta.write_bytes(
        sanity.to_bytes(4, 'little') + version.to_bytes(4, 'little', signed=True)
        + b'\x00RobotBalanceController\x00ArticulationBody\x00'
        + b'\x00NetworkTransportTickRate\x00RoundEndScoreboard\x00'
        + b'\x00PlayerInputBuffer\x00\x01\x02\x03')
    body = b'\x00Unity.Sentis.Worker\x00' if sentis else b'\x00PlainOldCode\x00'
    (game / 'GameAssembly.dll').write_bytes(
        b'MZ\x00' + body + b'\x00PhysicsScene.Simulate\x00Mixamo_Rig\x00')
    (game / 'REK_Data' / 'Plugins' / 'x86_64' / 'physx.dll').write_bytes(
        b'\x00PxSolverBody\x00')
    files = [
        {'path': 'REK_Data/il2cpp_data/Metadata/global-metadata.dat',
         'kind': 'il2cpp_metadata', 'size': meta.stat().st_size, 'sha256': 'a'},
        {'path': 'GameAssembly.dll', 'kind': 'il2cpp_code', 'size': 10, 'sha256': 'b'},
        {'path': 'REK_Data/Plugins/x86_64/physx.dll', 'kind': 'native_plugin',
         'size': 10, 'sha256': 'c'},
    ]
    return game, {'build_fingerprint': 'fp0', 'files': files}


@check
def the_il2cpp_probe_reads_the_metadata_header():
    with tempfile.TemporaryDirectory() as d:
        game, inv = _il2cpp_install(Path(d), version=31)
        r = il2cpp.probe(game, inv, Path(d) / 'out.json')
    md = r['metadata']
    assert md['valid'] and md['metadata_version'] == 31, md
    assert md['sanity'] == '0xfab11baf', md


@check
def a_packed_or_non_il2cpp_metadata_file_is_reported_not_assumed():
    with tempfile.TemporaryDirectory() as d:
        game, inv = _il2cpp_install(Path(d), sanity=0xDEADBEEF)
        r = il2cpp.probe(game, inv, Path(d) / 'out.json')
    md = r['metadata']
    assert not md['valid'], md
    assert 'encrypted/packed' in md['why'], md
    # It still scans the binaries: a packed metadata file does not mean there is
    # nothing to learn from the rest.
    assert r['scanned'], r


@check
def the_probe_classifies_names_into_instrumentation_targets():
    with tempfile.TemporaryDirectory() as d:
        game, inv = _il2cpp_install(Path(d))
        r = il2cpp.probe(game, inv, Path(d) / 'out.json')
    b = r['buckets']
    assert 'RobotBalanceController' in b['controller']['names'], b['controller']
    assert 'ArticulationBody' in b['physics']['names'], b['physics']
    assert 'PhysicsScene.Simulate' in b['physics']['names']
    assert 'NetworkTransportTickRate' in b['netcode']['names'], b['netcode']
    assert 'RoundEndScoreboard' in b['match_rules']['names'], b['match_rules']
    assert 'PlayerInputBuffer' in b['input']['names'], b['input']
    assert 'Mixamo_Rig' in b['animation']['names'], b['animation']
    assert all(v['role'] == 'candidate_lead' for v in b.values())
    assert len(r['scanned']) == 3


@check
def the_probe_detects_whether_an_inference_runtime_ships():
    # The strongest single result available without a decompiler: whether a
    # neural controller runs in the client at all.
    with tempfile.TemporaryDirectory() as d:
        game, inv = _il2cpp_install(Path(d), sentis=True)
        r = il2cpp.probe(game, inv, Path(d) / 'out.json')
    assert r['inference_runtime_present'] is True
    assert 'Unity.Sentis.Worker' in r['buckets']['inference_runtime']['names']

    with tempfile.TemporaryDirectory() as d:
        game, inv = _il2cpp_install(Path(d), sentis=False)
        r = il2cpp.probe(game, inv, Path(d) / 'out.json')
    assert r['inference_runtime_present'] is False
    assert r['buckets']['inference_runtime']['count'] == 0


@check
def the_probe_scans_the_whole_binary_not_just_the_front():
    # GameAssembly.dll runs to hundreds of megabytes and the interesting symbols
    # are not at the front. An earlier cap would have looked like a successful
    # probe while missing almost everything.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        big = d / 'big.bin'
        # A symbol far past any plausible cap, and one placed to straddle a
        # chunk boundary exactly.
        chunk = 1 << 16
        filler = b'\x00' * (chunk - 8)
        big.write_bytes(filler + b'Straddling_ControllerName' + filler
                        + b'DeepPolicyNetwork' + b'\x00' * 1000)
        names, truncated = il2cpp.extract_strings(big, chunk=chunk)
        assert 'DeepPolicyNetwork' in names, 'missed a symbol past the first chunk'
        assert 'Straddling_ControllerName' in names, 'symbol split at a boundary'
        assert truncated is False


@check
def files_listed_but_not_on_disk_are_reported():
    with tempfile.TemporaryDirectory() as d:
        game, inv = _il2cpp_install(Path(d))
        inv['files'].append({'path': 'REK_Data/Plugins/x86_64/gone.dll',
                             'kind': 'native_plugin', 'size': 1, 'sha256': 'd'})
        r = il2cpp.probe(game, inv, Path(d) / 'out.json')
    assert any(a.get('missing', '').endswith('gone.dll') for a in r['absent']), r['absent']


def _full_install(tmp: Path):
    """A build with both the Unity side and the IL2CPP side present."""
    game = fake_install(tmp)
    (game / 'GameAssembly.dll').write_bytes(
        b'MZ\x00Unity.Sentis.Worker\x00RobotBalanceController\x00'
        b'PhysicsScene.Simulate\x00')
    meta = game / 'REK_Data' / 'il2cpp_data' / 'Metadata' / 'global-metadata.dat'
    meta.write_bytes((0xFAB11BAF).to_bytes(4, 'little')
                     + (31).to_bytes(4, 'little', signed=True)
                     + b'\x00NetworkTransportTickRate\x00')
    return game


@check
def collect_runs_the_whole_non_interactive_pipeline():
    # The closest thing to a dry run of the one step that has to happen on a
    # machine this repo cannot reach.
    install_fake_unitypy(REK_OBJECTS)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        game = _full_install(d / 'game')
        out = d / 'evidence_out'
        log, inv = collect_mod.collect(game, out)

        assert all(step['ok'] for step in log), log
        assert {s['step'] for s in log} == {'inventory', 'il2cpp_probe',
                                            'static_survey'}

        for name in ('inventory.json', 'il2cpp_probe.json', 'static_survey.json',
                     'collect_log.json'):
            assert (out / name).exists(), name

        # Every artifact cites the same build, which is what makes them
        # composable at all.
        fp = inv['build_fingerprint']
        for name in ('il2cpp_probe.json', 'static_survey.json'):
            assert json.loads((out / name).read_text())['build_fingerprint'] == fp

        results = check_artifacts.check(out)
        state = _states(results)
        assert state['inventory.json'] == 'PRESENT', results
        assert state['static_survey.json'] == 'PRESENT', results
        assert state['build agreement'] == 'PRESENT', results
        # The authority test cannot be automated, so the package correctly
        # stops one stage short.
        assert check_artifacts.stage_of(results) == 'build pinned', results

        probe = json.loads((out / 'il2cpp_probe.json').read_text())
        assert probe['metadata']['metadata_version'] == 31
        assert probe['inference_runtime_present'] is True


@check
def a_partial_collection_is_legible_rather_than_empty():
    # No UnityPy: the asset survey cannot run. The build must still be pinned
    # and the probe must still run, and the failure has to be named.
    sys.modules.pop('UnityPy', None)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        game = _full_install(d / 'game')
        out = d / 'evidence_out'
        log, inv = collect_mod.collect(game, out)

        by_step = {s['step']: s for s in log}
        assert by_step['inventory']['ok'] and by_step['il2cpp_probe']['ok'], log
        assert not by_step['static_survey']['ok'], log
        assert 'UnityPy' in by_step['static_survey']['error'], by_step['static_survey']
        # The build is still pinned and the binaries still probed, so the run is
        # worth keeping rather than starting over.
        assert inv is not None and (out / 'inventory.json').exists()
        assert (out / 'il2cpp_probe.json').exists()
        assert not (out / 'static_survey.json').exists()
        assert json.loads((out / 'collect_log.json').read_text())['steps'] == log


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
def the_gate_rejects_a_rek_trace_with_uncited_channels():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # Written through the writer, then the citation stripped — which is how
        # this would actually happen: an older trace, or one hand-edited.
        _write(d / 'r0.trace', 'rek')
        raw = (d / 'r0.trace').read_bytes()
        raw = raw.replace(b'"provenance": {"root_x"', b'"provenance": {"nope_x"', 1)
        (d / 'r0.trace').write_bytes(raw)
        results = check_artifacts.check(d)
        state = _states(results)
        assert state.get('r0.trace') == 'INVALID', results
        assert 'no provenance' in [r for r in results
                                   if r['artifact'] == 'r0.trace'][0]['detail']


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
        except Exception as e:
            # A crash is a failure, not a reason to abandon the remaining
            # checks and report nothing.
            failed += 1
            print(f'  ERROR {fn.__name__.replace("_", " ")}: '
                  f'{type(e).__name__}: {e}')
    print(f'\n{len(checks) - failed}/{len(checks)} checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
