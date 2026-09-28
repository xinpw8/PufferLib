"""Bounded orchestration for the new isolated original-function test only."""
import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import paramiko

LOCAL = Path(__file__).resolve().parent
REMOTE = '/home/spark-advantage/rek-training/rek-native-clone-20260927-r1'


def connect():
    config = paramiko.SSHConfig.from_path(str(Path.home() / '.ssh' / 'config')).lookup('dgx_spark')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(config['hostname'], port=int(config.get('port', 22)),
                   username=config.get('user'), key_filename=config.get('identityfile'),
                   allow_agent=True, look_for_keys=True, timeout=10)
    return client


def mkdirs(sftp, directory):
    current = ''
    for component in directory.strip('/').split('/'):
        current += '/' + component
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current, 0o700)


def upload_tree(client, source, destination):
    with client.open_sftp() as sftp:
        try:
            sftp.stat(destination)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError('Fresh destination required: ' + destination)
        mkdirs(sftp, destination)
        receipts = []
        for path in sorted(source.rglob('*')):
            if path.is_symlink():
                raise RuntimeError('Unexpected source symlink: ' + str(path))
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            relative = path.relative_to(source).as_posix()
            remote = destination + '/' + relative
            mkdirs(sftp, remote.rsplit('/', 1)[0])
            data = path.read_bytes()
            with sftp.open(remote, 'wx') as handle:
                handle.write(data)
            with sftp.open(remote, 'rb') as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            if digest != hashlib.sha256(data).hexdigest():
                raise RuntimeError('Remote readback mismatch: ' + relative)
            receipts.append({'path': relative, 'bytes': len(data), 'sha256': digest})
        receipt = json.dumps({'destination': destination, 'files': receipts}, indent=2) + '\n'
        with sftp.open(destination + '/UPLOAD.json', 'wx') as handle:
            handle.write(receipt.encode())
        print(receipt, flush=True)


def run(client, argv):
    command = shlex.join(argv)
    from datetime import datetime, timezone
    execution_root = LOCAL / 'execution'
    execution_root.mkdir(exist_ok=True)
    for serial in range(1, 10000):
        record = execution_root / ('command-%04d' % serial)
        try:
            record.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError('No fresh execution record directory')
    (record / 'command.json').write_text(json.dumps({'remote_argv': argv,
        'started_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
    print(json.dumps({'remote_argv': argv}), flush=True)
    _, stdout, stderr = client.exec_command(command, timeout=None)
    channel = stdout.channel
    # Stream both channels to avoid deadlocking on a full stderr buffer.
    import time
    with (record / 'stdout.log').open('xb') as outlog, (record / 'stderr.log').open('xb') as errlog:
        while True:
            while channel.recv_ready():
                data = channel.recv(65536)
                outlog.write(data); outlog.flush()
                sys.stdout.buffer.write(data); sys.stdout.buffer.flush()
            while channel.recv_stderr_ready():
                data = channel.recv_stderr(65536)
                errlog.write(data); errlog.flush()
                sys.stderr.buffer.write(data); sys.stderr.buffer.flush()
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            time.sleep(0.1)
    code = channel.recv_exit_status()
    (record / 'result.json').write_text(json.dumps({'exit_code': code,
        'finished_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
    print(json.dumps({'remote_exit_code': code}), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='operation', required=True)
    upload = sub.add_parser('upload')
    upload.add_argument('source', type=Path)
    upload.add_argument('remote_child')
    execute = sub.add_parser('run')
    execute.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    with connect() as client:
        if args.operation == 'upload':
            child = args.remote_child
            if '/' in child or child in ('', '.', '..'):
                raise RuntimeError('One fresh child directory name required')
            upload_tree(client, args.source.resolve(), REMOTE + '/' + child)
        else:
            if not args.argv:
                raise RuntimeError('Remote argv required')
            sys.exit(run(client, args.argv))


if __name__ == '__main__':
    main()
