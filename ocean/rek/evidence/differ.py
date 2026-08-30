"""Differential validation: is the clone within REK's own run-to-run variance?

Two commands, in the order they have to be run.

    differ.py baseline rek_a.trace rek_b.trace rek_c.trace --out envelope.json
    differ.py compare rek_a.trace clone.trace --envelope envelope.json

`baseline` measures how far REK differs from *itself* when the same experiment
is repeated: same initial state, same seed, same action sequence, several runs.
That spread is the acceptance envelope. `compare` then holds a clone trace to
it and exits non-zero if it is looser anywhere.

The envelope is a quantile, not a maximum. A single anomalous REK run would
otherwise widen the tolerance enough to admit a materially wrong simulator, so
the default acceptance level is the 99th percentile with the maximum reported
alongside, and a wide gap between them is called out rather than silently
absorbed. Events are scored by precision and recall against a matching window,
not by zipping two lists and hoping they correspond.

Open-loop comparison has a horizon. Contact-rich humanoid dynamics amplify any
difference, so once a trajectory has diverged past the envelope, everything
after that tick is uninformative — agreement there is luck and disagreement
there is double-counting. The report names the first divergent tick and treats
the tail as invalid rather than averaging through it. For validation past that
point use short-horizon transition tests from injected states (--mode
short-horizon), or repeated closed-loop experiments compared distributionally.

This is the part that makes parity a number instead of a claim, and the reason
the oracle has to be REK. A suite of tests written against our own rules can
only confirm that the code does what its author expected, which is exactly how
a plausible but wrong model passes everything.

If REK is deterministic for a given seed the envelope collapses to zero and the
comparison becomes exact equality, which is the stronger result. Nothing here
assumes either way — it measures.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace import Trace


def _overlap(traces):
    """Ticks present in every trace, in order."""
    common = set(traces[0].ticks)
    for t in traces[1:]:
        common &= set(t.ticks)
    return sorted(common)


def _index(trace):
    return {tick: i for i, tick in enumerate(trace.ticks)}


def quantile(sorted_vals, q):
    """Linear-interpolated quantile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarise_errors(values):
    """The distribution, not just the worst point."""
    vals = sorted(values)
    n = len(vals)
    mean = sum(vals) / n if n else 0.0
    return {
        'n': n,
        'median': quantile(vals, 0.5),
        'p95': quantile(vals, 0.95),
        'p99': quantile(vals, 0.99),
        'max': vals[-1] if vals else 0.0,
        'rms': math.sqrt(sum(v * v for v in vals) / n) if n else 0.0,
        'mean': mean,
    }


# Units, so an envelope is read in the quantity it actually measures rather than
# as a bare float. Naming follows the convention in README.md.
def channel_unit(name):
    n = name.lower()
    if n.endswith(('.impulse',)):
        return 'N*s'
    if '.quat' in n:
        return 'quaternion component'
    if 'angvel' in n or (n.startswith('joint.') and n.endswith('.vel')):
        return 'rad/s'
    if n.endswith('.vel') or '.vel.' in n:
        return 'm/s'
    if n.startswith('joint.') or '.euler' in n:
        return 'rad'
    if n.startswith('root.') and ('.pos' in n):
        return 'm'
    if n.startswith('ctrl.'):
        return 'dimensionless'
    if n.startswith(('score.', 'downs.', 'round.')):
        return 'count/state'
    return 'unknown'


def _pairwise_channel_spread(traces, channels, ticks):
    """Per-channel error distribution over every pair of runs."""
    idx = [_index(t) for t in traces]
    out = {}
    for ch in channels:
        diffs = []
        for a in range(len(traces)):
            for b in range(a + 1, len(traces)):
                ca, cb = traces[a].channels[ch], traces[b].channels[ch]
                for tick in ticks:
                    diffs.append(abs(ca[idx[a][tick]] - cb[idx[b][tick]]))
        rec = summarise_errors(diffs)
        rec['unit'] = channel_unit(ch)
        # A max far above the bulk of the distribution means one run misbehaved.
        # Accepting at the max would hand that slack to the clone.
        rec['outlier_ratio'] = (rec['max'] / rec['p99']) if rec['p99'] > 0 else (
            float('inf') if rec['max'] > 0 else 1.0)
        out[ch] = rec
    return out


def _event_spread(traces):
    """How much event timing and counts move between identical REK runs."""
    kinds = {e['kind'] for t in traces for e in t.events}
    out = {}
    for kind in sorted(kinds):
        per_run = [sorted(e['tick'] for e in t.events if e['kind'] == kind)
                   for t in traces]
        counts = {len(p) for p in per_run}
        jitters = []
        for a in range(len(per_run)):
            for b in range(a + 1, len(per_run)):
                for x, y in zip(per_run[a], per_run[b]):
                    jitters.append(abs(x - y))
        dist = summarise_errors(jitters) if jitters else summarise_errors([0.0])
        out[kind] = {
            'count_varies': len(counts) > 1,
            'counts': sorted(counts),
            'tick_jitter': dist,
            # Matching window for compare. p99 rather than max, for the same
            # reason the channel envelope uses p99.
            'match_window': int(math.ceil(dist['p99'])),
            'max_tick_jitter': int(dist['max']),
        }
    return out


def match_events(reference, candidate, window):
    """Greedy nearest-match within `window` ticks, then precision and recall.

    Counting events and zipping them in order silently pairs unrelated
    occurrences as soon as one is missing, which turns a dropped hit into a
    timing error on every hit after it.
    """
    ref = sorted(reference)
    cand = sorted(candidate)
    used = [False] * len(cand)
    matched, offsets = 0, []
    for r in ref:
        best, best_d = None, None
        for i, c in enumerate(cand):
            if used[i]:
                continue
            d = abs(c - r)
            if d <= window and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is not None:
            used[best] = True
            matched += 1
            offsets.append(best_d)
    fp = len(cand) - matched
    fn = len(ref) - matched
    precision = matched / len(cand) if cand else (1.0 if not ref else 0.0)
    recall = matched / len(ref) if ref else (1.0 if not cand else 0.0)
    return {
        'reference_count': len(ref), 'candidate_count': len(cand),
        'matched': matched, 'false_positives': fp, 'false_negatives': fn,
        'precision': precision, 'recall': recall,
        'max_matched_offset': max(offsets) if offsets else 0,
        'window': window,
    }


def baseline(paths, out_path, accept):
    traces = [Trace.load(p) for p in paths]
    if len(traces) < 2:
        sys.exit('baseline needs at least two repeated REK runs to measure spread')
    for t, p in zip(traces, paths):
        if t.source != 'rek':
            sys.exit(f'{p} has source={t.source!r}; the baseline must be REK against itself')
    fps = {t.build_fingerprint for t in traces}
    if len(fps) != 1:
        sys.exit(f'traces span {len(fps)} different builds: {sorted(fps)}')

    channels = sorted(set(traces[0].channels) &
                      set.intersection(*[set(t.channels) for t in traces]))
    ticks = _overlap(traces)
    if not ticks:
        sys.exit('the runs share no ticks')

    env = {
        'schema': 2,
        'build_fingerprint': traces[0].build_fingerprint,
        'runs': len(traces),
        'ticks_compared': len(ticks),
        'accept_at': accept,
        'channels': _pairwise_channel_spread(traces, channels, ticks),
        'events': _event_spread(traces),
    }
    Path(out_path).write_text(json.dumps(env, indent=1))

    exact = [c for c, v in env['channels'].items() if v['max'] == 0.0]
    print(f'{len(traces)} REK runs, {len(ticks)} shared ticks, {len(channels)} channels')
    print(f'acceptance level: {accept}')
    print(f'bit-identical channels: {len(exact)}/{len(channels)}')

    if len(traces) < 3:
        print('\nNOTE: two runs give one difference per tick, so the quantiles '
              'are barely distinguishable from the max. Use at least three, and '
              'preferably more, before treating this envelope as robust.')

    if len(exact) == len(channels) and not any(
            v['count_varies'] or v['max_tick_jitter'] for v in env['events'].values()):
        print('REK replays this experiment deterministically. The clone must match '
              'it exactly — the envelope is zero.')
    else:
        loosest = sorted(env['channels'].items(), key=lambda kv: -kv[1][accept])[:8]
        print('\nwidest run-to-run spread:')
        for ch, v in loosest:
            print(f'  {ch:<34} {accept} {v[accept]:.6g} {v["unit"]:<8} '
                  f'median {v["median"]:.6g}  max {v["max"]:.6g}')
        # One bad run must not become everyone's allowance.
        skewed = [(c, v) for c, v in env['channels'].items()
                  if v['outlier_ratio'] > 10 and v['max'] > 0]
        if skewed:
            print(f'\n{len(skewed)} channel(s) have a max more than 10x their p99 — '
                  f'one REK run behaved unlike the others. Accepting at the max '
                  f'would hand that slack to the clone; investigate before '
                  f'trusting this envelope:')
            for c, v in skewed[:6]:
                print(f'  {c:<34} p99 {v["p99"]:.6g}  max {v["max"]:.6g}')

        for kind, v in sorted(env['events'].items()):
            if v['count_varies']:
                print(f'\nevent {kind!r} does not even occur a consistent number of '
                      f'times across identical REK runs: counts {v["counts"]}. '
                      f'That is a property of the experiment, not of any clone.')
    print(f'\nWrote {out_path}')
    return 0


# Ticks a short-horizon comparison may start from: points where both sides are
# known to be aligned because the state was reset or injected, rather than
# because the trajectory happened not to have diverged yet.
ANCHOR_KINDS = ('inject', 'reset', 'round_start')


def short_horizon(rek, clone, env, window, accept):
    """Windowed transition tests from anchored states.

    Open-loop replay answers "does the whole episode match", which in a
    contact-rich humanoid sim is mostly a question about how fast small
    differences amplify. This answers the more useful one: from a state both
    sides are known to share, do the next `window` ticks agree? Each window is
    scored independently, so one early divergence cannot invalidate the rest of
    the run.
    """
    anchors = sorted({e['tick'] for e in rek.events if e['kind'] in ANCHOR_KINDS})
    if not anchors:
        return None

    shared = sorted(set(rek.channels) & set(clone.channels))
    ri, ci = _index(rek), _index(clone)
    common = set(rek.ticks) & set(clone.ticks)

    windows = []
    for a in anchors:
        ticks = [t for t in range(a, a + window) if t in common]
        if len(ticks) < 2:
            continue
        per_channel, failed = {}, []
        for ch in shared:
            spread = env['channels'].get(ch)
            allowed = spread.get(accept) if spread else None
            diffs = [abs(rek.channels[ch][ri[t]] - clone.channels[ch][ci[t]])
                     for t in ticks]
            rec = summarise_errors(diffs)
            rec['allowed'] = allowed
            rec['within_envelope'] = allowed is not None and rec[accept] <= allowed
            per_channel[ch] = rec
            if not rec['within_envelope']:
                failed.append(ch)
        windows.append({'anchor_tick': a, 'ticks': len(ticks),
                        'channels': per_channel, 'failed_channels': failed,
                        'passed': not failed})

    passed = [w for w in windows if w['passed']]
    return {
        'window_ticks': window,
        'anchors': len(anchors),
        'windows_tested': len(windows),
        'windows_passed': len(passed),
        'pass_fraction': len(passed) / len(windows) if windows else 0.0,
        'windows': windows,
    }


def compare(rek_path, clone_path, envelope_path, report_path, horizon, window):
    rek, clone = Trace.load(rek_path), Trace.load(clone_path)
    if rek.source != 'rek':
        sys.exit(f'{rek_path} has source={rek.source!r}; the reference must be REK')
    if not str(clone.source).startswith('clone:'):
        sys.exit(f'{clone_path} has source={clone.source!r}; expected clone:<name>')
    env = json.loads(Path(envelope_path).read_text())
    if env['build_fingerprint'] != rek.build_fingerprint:
        sys.exit('envelope was measured on a different build than this reference trace')

    shared = sorted(set(rek.channels) & set(clone.channels))
    missing = sorted(set(rek.channels) - set(clone.channels))
    ticks = _overlap([rek, clone])
    ri, ci = _index(rek), _index(clone)

    accept = env.get('accept_at', 'p99')
    channels, failures, first_div = {}, [], None

    for ch in shared:
        spread = env['channels'].get(ch)
        allowed = spread.get(accept) if spread else None
        diffs, worst_tick, worst = [], None, -1.0
        for tick in ticks:
            d = abs(rek.channels[ch][ri[tick]] - clone.channels[ch][ci[tick]])
            diffs.append(d)
            if d > worst:
                worst, worst_tick = d, tick
            if allowed is not None and d > allowed:
                if first_div is None or tick < first_div:
                    first_div = tick
        rec = summarise_errors(diffs)
        rec.update({'worst_tick': worst_tick, 'allowed': allowed,
                    'accept_at': accept, 'unit': channel_unit(ch)})
        # Judged on the same statistic the envelope was cut at, not on the
        # single worst sample, which one unlucky tick would otherwise decide.
        rec['within_envelope'] = allowed is not None and rec[accept] <= allowed
        channels[ch] = rec
        if not rec['within_envelope']:
            failures.append(ch)

    # Events: a hit that lands two ticks late is a different game. Matched
    # nearest-within-window, then scored by precision and recall, so a dropped
    # event does not cascade into a timing error on every event after it.
    events = {}
    kinds = {e['kind'] for e in rek.events} | {e['kind'] for e in clone.events}
    for kind in sorted(kinds):
        r = [e['tick'] for e in rek.events if e['kind'] == kind]
        c = [e['tick'] for e in clone.events if e['kind'] == kind]
        # Named distinctly from the short-horizon window: they are different
        # quantities, and sharing the name silently zeroed the latter.
        match_window = env['events'].get(kind, {}).get('match_window', 0)
        rec = match_events(r, c, match_window)
        rec['agrees'] = (rec['precision'] == 1.0 and rec['recall'] == 1.0)
        events[kind] = rec
        if not rec['agrees']:
            failures.append(f'event:{kind}')

    # Everything after the first divergence is uninformative in open loop:
    # contact-rich dynamics amplify any difference, so later agreement is luck
    # and later disagreement is the same error counted again.
    valid_ticks = ([t for t in ticks if t < first_div] if first_div is not None
                   else ticks)
    hidden = [c for c in rek.channels if '.hidden' in c or '.recurrent' in c]

    windowed = None
    if horizon == 'short-horizon':
        windowed = short_horizon(rek, clone, env, window, accept)
        if windowed is None:
            sys.exit(
                'short-horizon mode needs anchor events in the REK trace — one '
                f'of {ANCHOR_KINDS} — marking states both sides are known to '
                'share. Without them there is nothing to start a window from, '
                'and the comparison would silently be open-loop again.')
        # In this mode the verdict is the windows, not the whole-episode drift.
        failures = [f'window@{w["anchor_tick"]}' for w in windowed['windows']
                    if not w['passed']]
        failures += [f'event:{k}' for k, v in events.items() if not v['agrees']]

    report = {
        'schema': 2,
        'mode': horizon,
        'short_horizon': windowed,
        'build_fingerprint': rek.build_fingerprint,
        'reference': str(rek_path), 'candidate': str(clone_path),
        'clone': clone.source,
        'accept_at': accept,
        'ticks_compared': len(ticks),
        'valid_horizon_ticks': len(valid_ticks),
        'channels_missing_from_clone': missing,
        'first_divergent_tick': first_div,
        'hidden_state_channels': hidden,
        'channels': channels,
        'events': events,
        'failures': failures,
        'passed': not failures and not missing,
    }
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=1))

    print(f'{len(ticks)} shared ticks, {len(shared)} shared channels')
    if missing:
        print(f'NOT RECORDED BY THE CLONE: {missing}')
    for ch, v in sorted(channels.items(), key=lambda kv: -kv[1][accept])[:12]:
        flag = 'ok  ' if v['within_envelope'] else 'OVER'
        allowed = 'n/a' if v['allowed'] is None else f'{v["allowed"]:.6g}'
        print(f'  {flag} {ch:<34} {accept} {v[accept]:.6g} {v["unit"]:<8} '
              f'allowed {allowed}  max {v["max"]:.6g}')
    for kind, v in events.items():
        flag = 'ok  ' if v['agrees'] else 'OVER'
        print(f'  {flag} event:{kind:<28} P {v["precision"]:.2f} R {v["recall"]:.2f}  '
              f'{v["matched"]}/{v["reference_count"]} matched within '
              f'{v["window"]} tick(s), {v["false_positives"]} spurious')

    if hidden and any(h not in clone.channels for h in hidden):
        print(f'\nREK recorded controller state the clone did not: '
              f'{[h for h in hidden if h not in clone.channels]}. Identical '
              f'visible poses can evolve differently when recurrent state or '
              f'skill phase differs, so this has to be captured or reconstructed.')

    if windowed:
        print(f'\nshort-horizon: {windowed["windows_passed"]}/'
              f'{windowed["windows_tested"]} windows of '
              f'{windowed["window_ticks"]} ticks passed, from '
              f'{windowed["anchors"]} anchor(s)')
        for w in windowed['windows']:
            if not w['passed']:
                print(f'  FAIL @{w["anchor_tick"]}: {w["failed_channels"][:6]}')

    if first_div is not None:
        print(f'\nfirst divergence beyond envelope at tick {first_div} — '
              f'{len(valid_ticks)} of {len(ticks)} ticks were still informative.')
        if horizon == 'open-loop' and len(ticks) - len(valid_ticks) > len(ticks) // 2:
            print('Most of this comparison is past the divergence point and means '
                  'nothing. Re-run as --mode short-horizon from injected states, '
                  'or compare repeated closed-loop experiments distributionally.')
    print('\nPARITY PASS' if report['passed'] else '\nPARITY FAIL: ' + ', '.join(failures[:10]))
    return 0 if report['passed'] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('baseline', help="measure REK's own run-to-run spread")
    b.add_argument('traces', nargs='+')
    b.add_argument('--out', default='envelope.json')
    b.add_argument('--accept-at', default='p99', choices=('median', 'p95', 'p99', 'max'),
        help='statistic of REK\'s own spread to accept at (default p99; max lets '
             'a single anomalous run widen the tolerance)')

    c = sub.add_parser('compare', help='hold a clone trace to that envelope')
    c.add_argument('rek_trace')
    c.add_argument('clone_trace')
    c.add_argument('--envelope', default='envelope.json')
    c.add_argument('--report', default='parity_report.json')
    c.add_argument('--mode', default='open-loop', choices=('open-loop', 'short-horizon'),
        help='open-loop replays a whole episode and is only informative up to '
             'the first divergence; short-horizon scores independent windows '
             'starting at each inject/reset/round_start event')
    c.add_argument('--window', type=int, default=30,
        help='ticks per window in short-horizon mode (default 30)')

    args = ap.parse_args()
    if args.cmd == 'baseline':
        return baseline(args.traces, args.out, args.accept_at)
    return compare(args.rek_trace, args.clone_trace, args.envelope, args.report,
                   args.mode, args.window)


if __name__ == '__main__':
    sys.exit(main())
