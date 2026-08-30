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
        inv = inventory.scan(game, include_volatile=False)
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
        # Logs are excluded by default; counting them would make the fingerprint
        # useless as an identity.
        assert 'logs/output.txt' not in kinds


@check
def fingerprint_ignores_noise_and_catches_updates():
    with tempfile.TemporaryDirectory() as d:
        game = fake_install(Path(d))
        first = inventory.scan(game, False)['build_fingerprint']

        (game / 'logs' / 'output.txt').write_text('a different log entirely')
        assert inventory.scan(game, False)['build_fingerprint'] == first

        (game / 'GameAssembly.dll').write_bytes(b'il2cpp v2')
        assert inventory.scan(game, False)['build_fingerprint'] != first


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
