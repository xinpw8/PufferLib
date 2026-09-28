"""Bounded private worker benchmarks. Never controls the live viewer.

Default is plan-only. --execute requires exact candidate/launcher hashes and a
paused, identity-matching live viewer throughout. Only owned processes stop.
"""
from pathlib import Path
import argparse, base64, hashlib, json, math, os, queue, signal, statistics
import select, subprocess, threading, time, urllib.request

LIVE_ROOT = Path('/home/spark-advantage/rek-training/rek-native-clone-20260927-r1')
LIVE_PID = 1092025
LIVE_START = '125031755'
LIVE_WORKER = 1092037
LIVE_WORKER_START = '125031762'
LIVE_URL = 'http://127.0.0.1:18771/api/snapshot'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def proc_start(pid):
    s = Path(f'/proc/{pid}/stat').read_text()
    return s[s.rfind(')')+2:].split()[19]


def http(url, value=None, timeout=3):
    data = None if value is None else json.dumps(value).encode()
    req = urllib.request.Request(url, data=data,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


class Guard:
    def __init__(self, fetch=None, identities=None, max_seconds=120):
        self.fetch = fetch or (lambda: http(LIVE_URL, timeout=.75))
        self.identities = identities or (lambda: (
            proc_start(LIVE_PID) == LIVE_START and
            proc_start(LIVE_WORKER) == LIVE_WORKER_START))
        self.stop_event = threading.Event()
        self.failure = None
        self.observations = []
        self.thread = None
        self.abort_callbacks = []
        self.deadline = time.monotonic()+max_seconds

    def check(self):
        if self.failure is not None:
            raise RuntimeError(self.failure)
        try:
            assert time.monotonic() < self.deadline, 'Whole benchmark deadline'
            assert self.identities(), 'Live process identity changed'
            state = self.fetch()
            assert state.get('paused') is True, 'Human resumed live viewer'
            assert state.get('ok') is True, 'Live viewer is not healthy'
            self.observations.append({'monotonic_ns': time.monotonic_ns(),
                'paused': state['paused'], 'tick': state.get('tick')})
        except Exception as error:
            self.failure = type(error).__name__+': '+str(error)
            for callback in list(self.abort_callbacks):
                callback()
            raise RuntimeError(self.failure) from error

    def start(self):
        self.check()
        def monitor():
            while not self.stop_event.wait(.2):
                try:
                    self.check()
                except Exception:
                    return
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()

    def poll(self):
        if self.failure is not None:
            raise RuntimeError(self.failure)
        if time.monotonic() >= self.deadline:
            self.check()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=4)


class Owned:
    def __init__(self, argv, env, directory, guard):
        self.guard = guard
        self.stderr = (directory/'stderr.log').open('xb')
        try:
            self.log = (directory/'protocol.jsonl').open('x')
        except Exception:
            self.stderr.close()
            raise
        try:
            self.process = subprocess.Popen(argv, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.stderr, env=env, text=True,
                bufsize=1, start_new_session=True)
        except Exception:
            self.log.close()
            self.stderr.close()
            raise
        self.pidfd = None
        try:
            self.pidfd = os.pidfd_open(self.process.pid)
            self.identity = {'pid': self.process.pid,
                'start_ticks': proc_start(self.process.pid), 'argv': argv}
        except Exception:
            # The newly spawned child remains unreaped, so its PID cannot be
            # reused. Prefer its stable handle when one was obtained.
            try:
                if self.pidfd is not None:
                    signal.pidfd_send_signal(self.pidfd, signal.SIGTERM)
                else:
                    self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            if self.pidfd is not None:
                os.close(self.pidfd)
            self.process.stdin.close()
            self.process.stdout.close()
            self.log.close()
            self.stderr.close()
            raise
        self.closed = False
        self.handle_lock = threading.Lock()
        self.child_handles = {}
        def abort():
            with self.handle_lock:
                if not self.closed:
                    try:
                        signal.pidfd_send_signal(self.pidfd, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        self.abort = abort
        guard.abort_callbacks.append(abort)
        self.lines = queue.Queue()
        self.serial = 0
        self.reply_durations = []
        def reader():
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)
        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()

    def emit(self, kind, value):
        self.log.write(json.dumps({'monotonic_ns': time.monotonic_ns(),
            'kind': kind, 'value': value})+'\n')
        self.log.flush()

    def capture_children(self):
        if self.process.poll() is not None:
            return
        p = Path(f'/proc/{self.process.pid}/task/{self.process.pid}/children')
        try:
            children = p.read_text().split()
        except FileNotFoundError:
            return
        for item in children:
            pid = int(item)
            if pid not in self.child_handles:
                try:
                    self.child_handles[pid] = os.pidfd_open(pid)
                except ProcessLookupError:
                    pass

    def receive(self, timeout=60):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            self.guard.poll()
            try:
                line = self.lines.get(timeout=.05)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(f'Owned child exited {self.process.poll()}')
            self.emit('stdout', line.rstrip('\n'))
            try:
                return json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError('Non-JSON stdout') from error
        raise TimeoutError('Owned child response timeout')

    def call(self, op, expect_ok=True, **args):
        self.guard.poll()
        self.serial += 1
        request = {'id': self.serial, 'op': op, **args}
        self.emit('request', request)
        start = time.monotonic_ns()
        self.process.stdin.write(json.dumps(request)+'\n')
        self.process.stdin.flush()
        result = self.receive()
        ms = (time.monotonic_ns()-start)/1e6
        self.emit('reply', {'duration_ms': ms, 'result': result})
        assert result['id'] == self.serial and result['ok'] is expect_ok, result
        if op == 'step':
            self.reply_durations.append(ms)
        return result

    def close(self, server=False):
        # Children belong to our private server. Open stable handles before its
        # shutdown; never signal a process discovered after that boundary.
        try:
            if server:
                try:
                    self.capture_children()
                except OSError as error:
                    self.emit('child_capture_error', str(error))
            if self.process.poll() is None:
                if server:
                    signal.pidfd_send_signal(self.pidfd, signal.SIGTERM)
                else:
                    self.process.stdin.close()
                try:
                    self.process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    signal.pidfd_send_signal(self.pidfd, signal.SIGTERM)
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        signal.pidfd_send_signal(self.pidfd, signal.SIGKILL)
                        self.process.wait(timeout=3)
            # Our server normally closes both workers. Ensure its captured
            # children cannot continue consuming GPU after an abort.
            for fd in self.child_handles.values():
                try:
                    signal.pidfd_send_signal(fd, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if not select.select([fd], [], [], 1)[0]:
                    try:
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    assert select.select([fd], [], [], 3)[0], 'Owned child failed to exit'
        finally:
            for fd in self.child_handles.values():
                os.close(fd)
            with self.handle_lock:
                self.closed = True
                self.guard.abort_callbacks.remove(self.abort)
                os.close(self.pidfd)
            self.emit('owned_exit', {'identity': self.identity,
                'exit_code': self.process.returncode})
            self.log.close()
            self.stderr.close()


def schedule(ticks=150):
    result = []
    for i in range(ticks):
        command = {'forward': 0, 'strafe': 0, 'yaw': 0,
            'moveIndex': -1, 'cancelAction': False}
        if 30 <= i < 60:
            command['forward'] = .5
        elif 60 <= i < 90:
            command['yaw'] = .5
        elif 90 <= i < 120:
            command.update(forward=.2, strafe=-.25, yaw=.1)
        if i == 120:
            command['moveIndex'] = 0
        elif i == 121:
            command['cancelAction'] = True
        result.append(command)
    return result


def distribution(values):
    v = sorted(values)
    return {'n': len(v), 'mean': statistics.mean(v),
        'median': statistics.median(v), 'p90': v[int((len(v)-1)*.9)],
        'p99': v[int((len(v)-1)*.99)], 'min': v[0], 'max': v[-1]}


def compare(left, right):
    assert len(left) == len(right)
    fields = ['tick', 'phase', 'roundNumber', 'terminal', 'roundResult', 'winner',
        'score', 'fightResult', 'fightWinner', 'commandResults', 'mask', 'actions']
    required = set(fields+['qpos', 'qvel', 'raw'])
    discrete, numeric, first = [], {}, None
    all_state_equal = True
    for i, (a, b) in enumerate(zip(left, right)):
        sa, sb = a['state'], b['state']
        assert required <= sa.keys() and required <= sb.keys(), 'Required state fields absent'
        all_state_equal = all_state_equal and sa == sb
        for key in fields:
            if sa.get(key) != sb.get(key):
                discrete.append({'index': i, 'field': key})
        if a.get('commandEvents') != b.get('commandEvents'):
            discrete.append({'index': i, 'field': 'commandEvents'})
        for key in ['qpos', 'qvel', 'raw']:
            assert len(sa[key]) == len(sb[key])
            record = numeric.setdefault(key, {'different_values': 0,
                'max_abs': 0, 'squared_sum': 0, 'values': 0})
            for j, (x, y) in enumerate(zip(sa[key], sb[key])):
                assert math.isfinite(x) and math.isfinite(y)
                d = abs(x-y)
                record['values'] += 1
                record['squared_sum'] += d*d
                record['max_abs'] = max(record['max_abs'], d)
                if x != y:
                    record['different_values'] += 1
                    if first is None:
                        first = {'index': i, 'field': key, 'component': j,
                            'left': x, 'right': y}
    for v in numeric.values():
        v['rmse'] = math.sqrt(v.pop('squared_sum')/v['values'])
    return {'exact_discrete_fields': not discrete,
        'discrete_differences': discrete, 'numeric': numeric,
        'first_numeric_difference': first,
        'exact_observed_trajectory': not discrete and first is None and all_state_equal,
        'all_state_fields_equal': all_state_equal,
        'similar_scatter_is_not_parity_proof': True}


def prepared(run):
    identity = json.loads((run/'identity.json').read_text())
    server = json.loads((run/'server.json').read_text())
    backend = server['backends'][0]
    for name, pin in identity['filePins'].items():
        assert sha(name) == pin, f'Changed pinned file {name}'
    config = json.loads(Path(backend['workerConfig']).read_text())
    assert config == identity['worker'] and backend['env'] == identity['env']
    assert config['cuda_graph_step'] is False, 'Graph must remain disabled'
    assert backend['executable'] in identity['filePins']
    return identity, server, backend


def run_trajectory(label, executable, config, env, output, guard, commands):
    directory = output/label
    directory.mkdir()
    p = Owned([str(executable), '--config', str(config)], env, directory, guard)
    try:
        assert p.receive(timeout=30)['event'] == 'ready'
        states = [p.call('snapshot')]
        assert states[0]['state']['tick'] == 0
        started = time.monotonic()
        for i, command in enumerate(commands):
            assert time.monotonic()-started < 30, 'Trajectory30s limit'
            value = p.call('step', steps=1, humanSide=0, command=command)
            assert value['state']['tick'] == i+1
            assert value['state']['ok'] and value['state']['failureBits'] == 0
            states.append(value)
        elapsed = time.monotonic()-started
        summary = {'label': label, 'identity': p.identity,
            'binary_sha256': sha(executable), 'ticks': len(commands),
            'elapsed_seconds': elapsed, 'control_sps': len(commands)/elapsed,
            'step_rpc_ms': distribution(p.reply_durations)}
        # Audit reporting-cache semantics separately from the timed trajectory.
        fresh = p.call('snapshot')
        assert fresh['state'] == states[-1]['state'], 'Last step/cache differs from explicit snapshot'
        batch = p.call('step', steps=3, humanSide=0, command={'moveIndex': 1})
        assert batch['state']['tick'] == len(commands)+3
        attempted = [e for e in batch['commandEvents'] if e['side']==0 and e['attempted']]
        assert len(attempted) == 1, 'One-shot edge was repeated in batch'
        assert p.call('snapshot')['state'] == batch['state']
        p.call('step', expect_ok=False, command={'forward': 2})
        assert p.call('snapshot')['state'] == batch['state'], 'Invalid input advanced or mutated state'
        reset = p.call('reset')
        assert reset['state']['tick'] == 0
        assert p.call('snapshot')['state'] == reset['state']
        image = base64.b64decode(p.call('frame')['png'], validate=True)
        assert image.startswith(b'\x89PNG\r\n\x1a\n')
        (directory/'reset-frame.png').write_bytes(image)
        assert p.call('snapshot')['state'] == reset['state'], 'Frame changed reset state'
        summary['cache_behavior_checks'] = ['step then snapshot exact',
            'three-step batch reports one edge', 'invalid command does not advance',
            'reset snapshot exact', 'reset frame PNG and no state advance']
        (directory/'states.json').write_text(json.dumps(states)+'\n')
        (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps({'trajectory_completed': summary}), flush=True)
        return states, summary
    finally:
        p.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['compare', 'app'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--candidate-sha')
    parser.add_argument('--launcher', type=Path)
    parser.add_argument('--launcher-sha')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode': args.mode, 'run': str(args.run),
            'out': str(args.out), 'execute': False,
            'bounds': 'compare:3x150ticks; app:20seconds; live pause guard200ms'}))
        return
    assert os.name == 'posix', 'Execute on isolated Spark only'
    assert not args.out.exists(), 'Fresh output required'
    identity, server, backend = prepared(args.run)
    if args.mode == 'compare':
        assert args.candidate and args.candidate_sha
        assert sha(args.candidate) == args.candidate_sha
    else:
        assert args.launcher and args.launcher_sha and sha(args.launcher) == args.launcher_sha
        assert server['port'] == 18774, 'Private benchmark port must be18774'
        assert args.run.resolve() != (LIVE_ROOT/'run-r4').resolve()
    guard = Guard(max_seconds=120 if args.mode=='compare' else 90)
    guard.start()
    args.out.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('REK_')}
    env.update(backend['env'])
    summary = {'mode': args.mode, 'success': False}
    try:
        if args.mode == 'compare':
            commands = schedule()
            (args.out/'commands.json').write_text(json.dumps(commands)+'\n')
            results = []
            for label, binary in [('old-a', backend['executable']),
                    ('old-b', backend['executable']), ('new', args.candidate)]:
                guard.check()
                results.append(run_trajectory(label, binary,
                    backend['workerConfig'], env, args.out, guard, commands))
            summary.update(runs=[x[1] for x in results],
                old_old=compare(results[0][0], results[1][0]),
                old_new=compare(results[0][0], results[2][0]))
        else:
            child_dir = args.out/'app-process'
            child_dir.mkdir()
            child = Owned(['node', str(args.launcher), str(args.run)],
                env, child_dir, guard)
            base = 'http://127.0.0.1:18774'
            samples = []
            try:
                ready = child.receive()
                assert ready.get('ready') is True
                deadline = time.monotonic()+60
                while True:
                    guard.poll()
                    state = http(base+'/api/snapshot')
                    if state.get('ok') and not state.get('switching'):
                        break
                    assert time.monotonic() < deadline, 'Private app initialization timeout'
                    time.sleep(.05)
                assert state['paused'] and state['tick'] == 0
                child.capture_children()
                # Own fresh test app only. Empty input retains neutral human
                # command while the native Bot1 opponent runs normally.
                http(base+'/api/play', {'paused': False})
                start = time.monotonic()
                while time.monotonic()-start < 20:
                    guard.poll()
                    value = http(base+'/api/state')  # keep only our app active
                    samples.append({'monotonic_ns': time.monotonic_ns(), 'state': value})
                    assert value.get('ok') is True
                    time.sleep(.1)
                http(base+'/api/play', {'paused': True})
                final = http(base+'/api/snapshot')
                assert final['paused'] is True
                summary.update(elapsed_seconds=time.monotonic()-start,
                    start_tick=0, final_tick=final['tick'], pace=final.get('pace'),
                    final=final)
                (args.out/'app-samples.json').write_text(json.dumps(samples)+'\n')
            finally:
                child.close(server=True)
        summary['success'] = True
    except Exception as error:
        summary['error'] = str(error)
        raise
    finally:
        guard.close()
        summary['guard_failure'] = guard.failure
        (args.out/'guard.json').write_text(json.dumps(guard.observations)+'\n')
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        files = [{'path': str(p.relative_to(args.out)), 'bytes': p.stat().st_size,
            'sha256': sha(p)} for p in sorted(args.out.rglob('*')) if p.is_file()]
        (args.out/'MANIFEST.json').write_text(json.dumps(files, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
