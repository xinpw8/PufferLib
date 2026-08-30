"""Differential validation: is the clone within REK's own run-to-run variance?

Two commands, in the order they have to be run.

    differ.py baseline rek_a.trace rek_b.trace rek_c.trace --out envelope.json
    differ.py compare rek_a.trace clone.trace --envelope envelope.json

`baseline` measures how far REK differs from *itself* when the same experiment
is repeated: same initial state, same seed, same action sequence, several runs.
That spread is the acceptance envelope. `compare` then holds a clone trace to
it and exits non-zero if it is looser anywhere.

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


def _pairwise_channel_spread(traces, channels, ticks):
    """Max and RMS absolute difference per channel, over every pair of runs."""
    idx = [_index(t) for t in traces]
    out = {}
    for ch in channels:
        worst, sq, n = 0.0, 0.0, 0
        for a in range(len(traces)):
            for b in range(a + 1, len(traces)):
                ca, cb = traces[a].channels[ch], traces[b].channels[ch]
                for tick in ticks:
                    d = abs(ca[idx[a][tick]] - cb[idx[b][tick]])
                    worst = max(worst, d)
                    sq += d * d
                    n += 1
        out[ch] = {'max': worst, 'rms': math.sqrt(sq / n) if n else 0.0}
    return out


def _event_spread(traces):
    """How much event timing and counts move between identical REK runs."""
    kinds = {e['kind'] for t in traces for e in t.events}
    out = {}
    for kind in sorted(kinds):
        per_run = [sorted(e['tick'] for e in t.events if e['kind'] == kind)
                   for t in traces]
        counts = {len(p) for p in per_run}
        jitter = 0
        for a in range(len(per_run)):
            for b in range(a + 1, len(per_run)):
                for x, y in zip(per_run[a], per_run[b]):
                    jitter = max(jitter, abs(x - y))
        out[kind] = {
            'count_varies': len(counts) > 1,
            'counts': sorted(counts),
            'max_tick_jitter': jitter,
        }
    return out


def baseline(paths, out_path):
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
        'schema': 1,
        'build_fingerprint': traces[0].build_fingerprint,
        'runs': len(traces),
        'ticks_compared': len(ticks),
        'channels': _pairwise_channel_spread(traces, channels, ticks),
        'events': _event_spread(traces),
    }
    Path(out_path).write_text(json.dumps(env, indent=1))

    exact = [c for c, v in env['channels'].items() if v['max'] == 0.0]
    print(f'{len(traces)} REK runs, {len(ticks)} shared ticks, {len(channels)} channels')
    print(f'bit-identical channels: {len(exact)}/{len(channels)}')
    if len(exact) == len(channels) and not any(
            v['count_varies'] or v['max_tick_jitter'] for v in env['events'].values()):
        print('REK replays this experiment deterministically. The clone must match '
              'it exactly — the envelope is zero.')
    else:
        loosest = sorted(env['channels'].items(), key=lambda kv: -kv[1]['max'])[:8]
        print('widest run-to-run spread:')
        for ch, v in loosest:
            print(f'  {ch:<40} max {v["max"]:.6g}  rms {v["rms"]:.6g}')
    print(f'\nWrote {out_path}')
    return 0


def compare(rek_path, clone_path, envelope_path, report_path):
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

    channels, failures, first_div = {}, [], None
    for ch in shared:
        allowed = env['channels'].get(ch, {}).get('max')
        worst, sq, at = 0.0, 0.0, None
        for tick in ticks:
            d = abs(rek.channels[ch][ri[tick]] - clone.channels[ch][ci[tick]])
            sq += d * d
            if d > worst:
                worst, at = d, tick
            if allowed is not None and d > allowed and at is not None:
                if first_div is None or tick < first_div:
                    first_div = tick
        rec = {'max': worst, 'rms': math.sqrt(sq / len(ticks)) if ticks else 0.0,
               'worst_tick': at, 'allowed': allowed}
        rec['within_envelope'] = allowed is not None and worst <= allowed
        channels[ch] = rec
        if not rec['within_envelope']:
            failures.append(ch)

    # Events: a hit that lands two ticks late is a different game. Matched by
    # kind and order, with the jitter REK itself showed as the allowance.
    events = {}
    kinds = {e['kind'] for e in rek.events} | {e['kind'] for e in clone.events}
    for kind in sorted(kinds):
        r = sorted(e['tick'] for e in rek.events if e['kind'] == kind)
        c = sorted(e['tick'] for e in clone.events if e['kind'] == kind)
        allowed = env['events'].get(kind, {}).get('max_tick_jitter', 0)
        offsets = [abs(x - y) for x, y in zip(r, c)]
        rec = {
            'rek_count': len(r), 'clone_count': len(c),
            'max_tick_offset': max(offsets) if offsets else 0,
            'allowed_tick_jitter': allowed,
        }
        rec['agrees'] = (len(r) == len(c)
                         and rec['max_tick_offset'] <= allowed)
        events[kind] = rec
        if not rec['agrees']:
            failures.append(f'event:{kind}')

    report = {
        'schema': 1,
        'build_fingerprint': rek.build_fingerprint,
        'reference': str(rek_path), 'candidate': str(clone_path),
        'clone': clone.source,
        'ticks_compared': len(ticks),
        'channels_missing_from_clone': missing,
        'first_divergent_tick': first_div,
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
    for ch, v in sorted(channels.items(), key=lambda kv: -kv[1]['max'])[:12]:
        flag = 'ok ' if v['within_envelope'] else 'OVER'
        allowed = 'n/a' if v['allowed'] is None else f'{v["allowed"]:.6g}'
        print(f'  {flag} {ch:<38} max {v["max"]:.6g}  allowed {allowed}')
    for kind, v in events.items():
        flag = 'ok ' if v['agrees'] else 'OVER'
        print(f'  {flag} event:{kind:<32} {v["rek_count"]} vs {v["clone_count"]}, '
              f'offset {v["max_tick_offset"]} (allowed {v["allowed_tick_jitter"]})')
    if first_div is not None:
        print(f'\nfirst divergence beyond envelope at tick {first_div}')
    print('\nPARITY PASS' if report['passed'] else '\nPARITY FAIL: ' + ', '.join(failures[:10]))
    return 0 if report['passed'] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('baseline', help="measure REK's own run-to-run spread")
    b.add_argument('traces', nargs='+')
    b.add_argument('--out', default='envelope.json')

    c = sub.add_parser('compare', help='hold a clone trace to that envelope')
    c.add_argument('rek_trace')
    c.add_argument('clone_trace')
    c.add_argument('--envelope', default='envelope.json')
    c.add_argument('--report', default='parity_report.json')

    args = ap.parse_args()
    if args.cmd == 'baseline':
        return baseline(args.traces, args.out)
    return compare(args.rek_trace, args.clone_trace, args.envelope, args.report)


if __name__ == '__main__':
    sys.exit(main())
