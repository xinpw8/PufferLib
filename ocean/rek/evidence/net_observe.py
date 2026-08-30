"""Step 2: does practice-mode physics run locally or on a server?

Everything downstream forks on this. If practice is locally simulated, the
authoritative simulation is on the machine and can be instrumented directly. If
practice is also server-owned, there is no local ground truth to instrument and
the recorder has to sit above the network transport, capturing commands sent and
state replicated back.

Reflex Arc says live matches use a dedicated authoritative server. That does not
establish where *practice* physics runs, and it must not be assumed either way.

This samples the game process's sockets while you play, so the answer comes from
observed traffic rather than from inference:

    python net_observe.py --name REK --seconds 120 --out net_practice.json

Then play practice mode, alone, for the duration. Run it a second time in a live
match for contrast — a local practice mode and a server-owned match should look
obviously different, and if they look the same, practice is server-owned too.

Reads socket tables only. It does not capture packet contents, attach to the
process, or modify anything.

Interpreting the result:

  sustained remote flow at the tick rate   ->  server-owned; instrument above
                                               the transport
  no remote flow, or only telemetry-rate   ->  locally simulated; instrument the
  traffic to unrelated hosts                   local simulation directly

A confirmation worth doing either way: block the process in the firewall and see
whether practice still steps. If it does, physics is local.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def _via_psutil(name_filter, pid_filter):
    import psutil
    rows, pids = [], []
    for proc in psutil.process_iter(['pid', 'name']):
        if pid_filter and proc.info['pid'] != pid_filter:
            continue
        if name_filter and name_filter.lower() not in (proc.info['name'] or '').lower():
            continue
        pids.append(proc.info['pid'])
        try:
            for c in proc.net_connections(kind='inet'):
                if not c.raddr:
                    continue
                rows.append({
                    'pid': proc.info['pid'],
                    'proc': proc.info['name'],
                    'type': 'tcp' if c.type == 1 else 'udp',
                    'laddr': f'{c.laddr.ip}:{c.laddr.port}',
                    'raddr': f'{c.raddr.ip}:{c.raddr.port}',
                    'status': c.status,
                })
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return rows, pids


def _via_netstat(name_filter, pid_filter):
    """Fallback when psutil is not installed. Windows netstat / Linux ss."""
    rows, pids = [], []
    if sys.platform.startswith('win'):
        out = subprocess.run(['netstat', '-ano'], capture_output=True, text=True).stdout
        for line in out.splitlines():
            f = line.split()
            if len(f) < 4 or f[0].upper() not in ('TCP', 'UDP'):
                continue
            pid = int(f[-1]) if f[-1].isdigit() else None
            if pid_filter and pid != pid_filter:
                continue
            raddr = f[2]
            if raddr in ('*:*', '0.0.0.0:0', '[::]:0'):
                continue
            rows.append({'pid': pid, 'proc': None, 'type': f[0].lower(),
                         'laddr': f[1], 'raddr': raddr,
                         'status': f[3] if f[0].upper() == 'TCP' else ''})
            if pid:
                pids.append(pid)
    else:
        out = None
        for cmd in (['ss', '-tunp'], ['netstat', '-tunp']):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True).stdout
                break
            except FileNotFoundError:
                continue
        if out is None:
            raise RuntimeError(
                'no way to read the socket table: install psutil (pip install '
                'psutil), or provide ss/netstat on PATH')
        for line in out.splitlines()[1:]:
            f = line.split()
            if len(f) < 6:
                continue
            proc = f[-1]
            if name_filter and name_filter.lower() not in proc.lower():
                continue
            rows.append({'pid': None, 'proc': proc, 'type': f[0],
                         'laddr': f[4], 'raddr': f[5], 'status': f[1]})
    return rows, pids


def sample(name_filter, pid_filter):
    """psutil where available — it attributes sockets to a pid reliably on
    Windows, which netstat parsing does only awkwardly."""
    try:
        return _via_psutil(name_filter, pid_filter)
    except ImportError:
        return _via_netstat(name_filter, pid_filter)


def observe(name_filter, pid_filter, seconds, interval, out_path, note):
    try:
        sample(name_filter, pid_filter)
    except RuntimeError as e:
        sys.exit(str(e))

    started = time.time()
    timeline, peers = [], Counter()
    seen_pids = set()

    print(f'Watching for {seconds}s. Play now — one mode only, and say which in --note.')
    while time.time() - started < seconds:
        rows, pids = sample(name_filter, pid_filter)
        seen_pids.update(pids)
        timeline.append({'t': round(time.time() - started, 2),
                         'connections': rows})
        for r in rows:
            peers[(r['type'], r['raddr'])] += 1
        time.sleep(interval)

    samples = len(timeline)
    ranked = sorted(peers.items(), key=lambda kv: -kv[1])
    report = {
        'schema': 1,
        'note': note,
        'process_filter': name_filter, 'pid_filter': pid_filter,
        'pids_seen': sorted(seen_pids),
        'seconds': seconds, 'interval': interval, 'samples': samples,
        'peers': [{'proto': p, 'raddr': a, 'samples_present': n,
                   'fraction_of_run': round(n / samples, 3)}
                  for (p, a), n in ranked],
        'timeline': timeline,
    }
    Path(out_path).write_text(json.dumps(report, indent=1))

    print(f'\n{samples} samples over {seconds}s')
    if not ranked:
        print('No remote connections observed from the game process at all.')
        print('Consistent with locally simulated practice — confirm by blocking '
              'the process in the firewall and checking practice still steps.')
    else:
        print('remote peers, by how much of the run they were present for:')
        for entry in report['peers'][:15]:
            print(f'  {entry["proto"]:<4} {entry["raddr"]:<45} '
                  f'{entry["fraction_of_run"] * 100:5.1f}%')
        sustained = [e for e in report['peers'] if e['fraction_of_run'] > 0.9]
        print(f'\n{len(sustained)} peer(s) held for >90% of the run.')
        print('A sustained flow during practice points to server-owned physics. '
              'Compare against a run captured in a live match before concluding: '
              'the contrast between the two is the actual evidence, not either '
              'run alone.')
    if not seen_pids and not name_filter and not pid_filter:
        print('\nNo process filter was given, so this watched everything. '
              'Pass --name or --pid to attribute traffic to the game.')
    print(f'\nWrote {out_path}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', help='process name substring, e.g. REK')
    ap.add_argument('--pid', type=int, help='exact process id instead of a name')
    ap.add_argument('--seconds', type=int, default=120)
    ap.add_argument('--interval', type=float, default=0.5)
    ap.add_argument('--note', default='', help='which mode you played, for the record')
    ap.add_argument('--out', default='net_observe.json')
    args = ap.parse_args()
    if not args.name and not args.pid:
        ap.error('give --name or --pid so traffic can be attributed to the game')
    return observe(args.name, args.pid, args.seconds, args.interval, args.out, args.note)


if __name__ == '__main__':
    sys.exit(main())
