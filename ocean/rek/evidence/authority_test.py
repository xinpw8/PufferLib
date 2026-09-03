"""Step 2, done properly: where does practice-mode physics actually execute?

net_observe.py samples sockets. That is reconnaissance and cannot decide this on
its own: practice could hold authentication, telemetry, leaderboard or presence
sockets while simulating locally; it could be server-driven over a mostly idle
socket; it could run local prediction corrected by a remote authority; or it
could contact the server only at reset and result submission. Socket presence
distinguishes none of those.

What does distinguish them is intervention. Run practice normally, cut the
game's networking after the arena has loaded, keep issuing inputs, and see
whether state evolution continues.

    python authority_test.py --name REK --out authority_practice.json

The tool never touches your firewall. It prompts you to apply and remove the
block, or runs commands you supply with --block-cmd / --unblock-cmd, and
timestamps everything either way.

While it runs, type marks and press Enter. The vocabulary is fixed so the
interpretation is mechanical rather than a reading of free text:

    input             you issued a movement or attack command
    state-progressed  the robot visibly responded and the world kept stepping
    frozen            the world stopped stepping, or inputs stopped taking effect
    latency-up        it still responds, but visibly later than before
    reset-ok          an arena reset completed
    reset-failed      a reset was attempted and did not complete
    score-ok          a hit registered on the scoreboard
    score-failed      a hit landed visually and did not register
    rollback          state visibly jumped or corrected itself
    note <text>       anything else, free text, not used by the verdict

    block / unblock   apply and lift the network block
    done              finish

The verdict is derived only from marks in the blocked phase, and only after
checking the block actually took effect. If the game kept talking to the network
while "blocked", no conclusion is offered — a failed intervention that reads as
evidence is worse than no evidence.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_observe import sample

PHASES = ('baseline_online', 'blocked', 'restored')

MARKS = ('input', 'state-progressed', 'frozen', 'latency-up', 'reset-ok',
         'reset-failed', 'score-ok', 'score-failed', 'rollback')


class Run:
    """Timeline of samples, phase changes and operator marks.

    Sampling and the clock are injected so the whole run can be replayed in a
    test without a game or a network.
    """

    def __init__(self, sample_fn, now_fn=time.time):
        self._sample = sample_fn
        self._now = now_fn
        self.t0 = now_fn()
        self.phase = PHASES[0]
        self.samples = []
        self.marks = []
        self.commands = []
        self.phase_changes = [{'t': 0.0, 'phase': self.phase}]

    def command(self, cmd, returncode):
        """What the operator's block/unblock command did.

        A block command that failed still moves the phase, and the socket check
        is then the only thing standing between a failed intervention and a
        confident verdict. Recording the exit status puts the more direct
        evidence in the artifact too.
        """
        self.commands.append({'t': self._t(), 'phase': self.phase,
                              'cmd': cmd, 'returncode': returncode})

    def _t(self):
        return round(self._now() - self.t0, 3)

    def set_phase(self, phase):
        if phase not in PHASES:
            raise ValueError(f'unknown phase {phase!r}')
        self.phase = phase
        self.phase_changes.append({'t': self._t(), 'phase': phase})

    def mark(self, kind, text=''):
        self.marks.append({'t': self._t(), 'phase': self.phase,
                           'kind': kind, 'text': text})

    def observe(self):
        rows, _ = self._sample()
        remote = [r for r in rows if r.get('raddr')]
        self.samples.append({
            't': self._t(),
            'phase': self.phase,
            'remote_connections': len(remote),
            'peers': sorted({r['raddr'] for r in remote}),
            'established': sum(1 for r in remote
                               if str(r.get('status', '')).upper() == 'ESTABLISHED'),
        })

    def report(self):
        return {
            'schema': 1,
            'phases': self.phase_changes,
            'commands': self.commands,
            'marks': self.marks,
            'samples': self.samples,
            'duration_s': self._t(),
        }


def _phase_samples(report, phase):
    return [s for s in report['samples'] if s['phase'] == phase]


def _phase_marks(report, phase):
    return [m for m in report['marks'] if m['phase'] == phase]


def block_took_effect(report):
    # A block command that exited non-zero is decisive on its own: whatever the
    # sockets look like, the intervention was not applied as intended.
    for c in report.get('commands', []):
        if c['phase'] == 'blocked' and c['returncode'] not in (0, None):
            return {'effective': False,
                    'reason': f'the block command exited {c["returncode"]}: '
                              f'{c["cmd"]!r}'}
    return _block_took_effect_from_sockets(report)


def _block_took_effect_from_sockets(report):
    """Did the intervention actually cut the game's networking?

    Judged on the game's own sockets, not on whether the machine went quiet.
    A block that left established remote connections up is a failed block, and
    anything inferred from that phase would be inferred from nothing.
    """
    online = _phase_samples(report, 'baseline_online')
    blocked = _phase_samples(report, 'blocked')
    if not blocked:
        return {'effective': False, 'reason': 'no samples in the blocked phase'}
    if not online:
        return {'effective': False, 'reason': 'no baseline to compare against'}

    online_peers = set().union(*[set(s['peers']) for s in online]) or set()
    blocked_peers = set().union(*[set(s['peers']) for s in blocked]) or set()
    online_est = max(s['established'] for s in online)
    blocked_est = max(s['established'] for s in blocked)
    survivors = sorted(blocked_peers & online_peers)

    if not online_peers:
        return {'effective': False,
                'reason': 'the game held no remote connections even before the '
                          'block, so blocking cannot demonstrate anything',
                'online_peers': [], 'surviving_peers': survivors}
    if survivors and blocked_est >= online_est:
        return {'effective': False,
                'reason': 'connections that were up before the block were still '
                          'established during it — the block did not apply to '
                          'this process',
                'online_peers': sorted(online_peers), 'surviving_peers': survivors}
    return {'effective': True,
            'reason': f'{len(online_peers)} peer(s) before, {len(survivors)} '
                      f'still present during the block; established '
                      f'{online_est} -> {blocked_est}',
            'online_peers': sorted(online_peers), 'surviving_peers': survivors}


def interpret(report):
    """Verdict from the blocked phase, or an explicit refusal to give one."""
    effect = block_took_effect(report)
    kinds = [m['kind'] for m in _phase_marks(report, 'blocked')]
    counts = {k: kinds.count(k) for k in MARKS}
    out = {'block': effect, 'blocked_phase_marks': counts}

    if not effect['effective']:
        out['verdict'] = 'inconclusive'
        out['because'] = ('the network block did not take effect, so the blocked '
                          'phase is not evidence of anything: ' + effect['reason'])
        return out

    if not any(counts[k] for k in MARKS):
        out['verdict'] = 'inconclusive'
        out['because'] = ('nothing was marked during the blocked phase. Inputs '
                          'have to be issued and the result marked while the '
                          'block is up, or there is no observation.')
        return out

    progressed = counts['state-progressed']
    froze = counts['frozen']
    rolled = counts['rollback']
    interacted = counts['reset-ok'] + counts['score-ok']

    if froze and not progressed:
        out['verdict'] = 'remote_authority'
        out['because'] = ('the world stopped stepping once the game could not '
                          'reach the network, which is what a server-owned '
                          'simulation does.')
    elif progressed and rolled:
        out['verdict'] = 'local_prediction_remote_correction'
        out['because'] = ('state kept evolving while blocked and then visibly '
                          'corrected, which is client prediction reconciled '
                          'against a remote authority.')
    elif progressed and interacted and not froze:
        out['verdict'] = 'local_authority'
        out['because'] = (f'state kept evolving with no network, including '
                          f'{interacted} reset/score interaction(s) completing. '
                          f'The practice simulation runs on this machine.')
    elif progressed:
        out['verdict'] = 'local_authority_weak'
        out['because'] = ('state kept evolving while blocked, but no reset or '
                          'scoring interaction was confirmed. Repeat and drive '
                          'a fall, a recovery, a score and an arena reset while '
                          'the block is up.')
    else:
        out['verdict'] = 'ambiguous'
        out['because'] = ('the marks do not separate the cases. Next step is to '
                          'instrument command serialization and state '
                          'deserialization directly.')

    out['limits'] = (
        'This tests practice mode on this client only. It says nothing about '
        'live matches, which are separately described as server-authoritative. '
        'A local practice simulation may still differ from the server build.')
    return out


def handle_line(run, line):
    """Apply one operator line. Returns a short status for the caller to print.

    Split out from the reader thread so the vocabulary is testable: a mistyped
    mark that were silently ignored would cost a whole session, since the marks
    are the only record of what the operator actually saw.
    """
    parts = line.strip().split(None, 1)
    if not parts:
        return None
    cmd, rest = parts[0].lower(), (parts[1] if len(parts) > 1 else '')

    if cmd == 'done':
        return 'done'
    if cmd in ('block', 'unblock'):
        run.set_phase('blocked' if cmd == 'block' else 'restored')
        return f'-> phase {run.phase}'
    if cmd == 'note':
        run.mark('note', rest)
        return 'noted'
    if cmd in MARKS:
        run.mark(cmd, rest)
        return f'marked {cmd} in {run.phase}'
    return (f'? unknown mark {cmd!r}; one of: {", ".join(MARKS)}, '
            f'note, block, unblock, done')


def _reader(run, stop):
    """Operator marks, read on a background thread so sampling never stalls."""
    for line in sys.stdin:
        if stop.is_set():
            return
        status = handle_line(run, line)
        if status == 'done':
            stop.set()
            return
        if status:
            print(f'  {status}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', help='process name substring, e.g. REK')
    ap.add_argument('--pid', type=int)
    ap.add_argument('--interval', type=float, default=0.5)
    ap.add_argument('--max-seconds', type=int, default=900)
    ap.add_argument('--block-cmd', help='command that blocks the game\'s network '
                                        'access; prompted for if omitted')
    ap.add_argument('--unblock-cmd', help='command that lifts it')
    ap.add_argument('--out', default='authority_test.json')
    ap.add_argument('--replay', help='re-interpret a saved run instead of '
                                     'recording a new one')
    args = ap.parse_args()

    if args.replay:
        report = json.loads(Path(args.replay).read_text())
        verdict = interpret(report)
        print(json.dumps(verdict, indent=1))
        return 0

    if not args.name and not args.pid:
        ap.error('give --name or --pid so traffic can be attributed to the game')

    run = Run(lambda: sample(args.name, args.pid))
    stop = threading.Event()
    threading.Thread(target=_reader, args=(run, stop), daemon=True).start()

    print(__doc__.split('While it runs,')[1].split('The verdict')[0].strip())
    print('\nStart practice mode and let the arena load. Then type: block')
    if args.block_cmd:
        print(f'(--block-cmd will be run for you: {args.block_cmd})')

    last_phase = run.phase
    started = time.time()
    while not stop.is_set() and time.time() - started < args.max_seconds:
        if run.phase != last_phase:
            cmd = args.block_cmd if run.phase == 'blocked' else args.unblock_cmd
            if cmd and run.phase in ('blocked', 'restored'):
                proc = subprocess.run(cmd, shell=True)
                run.command(cmd, proc.returncode)
                if proc.returncode != 0:
                    print(f'  !! command exited {proc.returncode}: {cmd}')
            last_phase = run.phase
        run.observe()
        time.sleep(args.interval)

    report = run.report()
    report['verdict'] = interpret(report)
    Path(args.out).write_text(json.dumps(report, indent=1))

    v = report['verdict']
    print(f'\nblock effective : {v["block"]["effective"]} — {v["block"]["reason"]}')
    print(f'verdict         : {v["verdict"]}')
    print(f'because         : {v["because"]}')
    if 'limits' in v:
        print(f'limits          : {v["limits"]}')
    print(f'\nWrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
