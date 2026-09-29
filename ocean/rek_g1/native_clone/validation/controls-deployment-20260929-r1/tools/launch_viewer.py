"""Start a fresh, pinned viewer on an unused loopback port, initially paused."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, signal, socket, subprocess, time, urllib.request

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def validate_run(app,run,expected_binary_sha256,expected_port=18772):
    config=json.loads((run/'server.json').read_text())
    identity=json.loads((run/'identity.json').read_text())
    assert identity['schema']=='rek.native_clone.app_run.v1'
    assert isinstance(expected_port,int) and 1024<=expected_port<=65535,'Invalid loopback port'
    assert config['port']==identity['port']==expected_port,'Prepared viewer port differs from requested port'
    assert len(config['backends'])==1,'One prepared backend required'
    backend=config['backends'][0]
    assert backend['id']=='mujoco'
    assert Path(backend['workerConfig']).resolve()==run/'worker.json'
    worker=json.loads((run/'worker.json').read_text())
    assert worker==identity['worker'],'Worker differs from prepared identity'
    assert backend['env']==identity['env'],'Environment differs from prepared identity'
    executable=Path(backend['executable']).resolve()
    assert identity['filePins'].get(str(executable))==expected_binary_sha256,'Executable not bound to expected binary'
    assert sha(executable)==expected_binary_sha256,'Executable hash mismatch'
    assert Path(backend['logFile']).resolve()==run/'worker.stderr.log','Worker log outside fresh run'
    assert Path(config['leagueFile']).resolve()==run/'league.json','League outside fresh run'
    assert config['initial']=={'backend':'mujoco','opponent':'bot1','humanSide':0,
                              'roundSeconds':worker.get('round_seconds') or 120}
    assert identity['controlsSha256']==sha(app/'saved-g1-bindings.json')
    for filename,digest in identity['filePins'].items():
        assert sha(Path(filename))==digest,filename
    return config,identity

def process_identity(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    return {'pid':pid,'process_group':int(fields[2]),'session':int(fields[3]),'start_ticks':int(fields[19])}

def check_preserved_viewers(guards):
    try:
        _check_preserved_viewers(guards)
    except (OSError,ValueError) as error:
        raise RuntimeError('Preserved viewer guard unavailable') from error

def _check_preserved_viewers(guards):
    for viewer in guards:
        port=viewer['port']
        assert isinstance(port,int) and 1024<=port<=65535,'Invalid guarded port'
        for expected in viewer['processes']:
            assert process_identity(expected['pid'])['start_ticks']==expected['start_ticks'],'Preserved process identity changed'
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/snapshot',timeout=2) as response:
            state=json.load(response)
        assert state.get('ok') is True and state.get('paused') is True,'Preserved human viewer resumed or failed'

def owned_group_alive(process,start_ticks):
    try:
        leader=process_identity(process.pid)
    except FileNotFoundError:
        leader=None
    if leader:
        assert leader['process_group']==leader['session']==process.pid,'New process no longer owns its session'
        if start_ticks is not None:
            assert leader['start_ticks']==start_ticks,'PID was reused; refusing cleanup'
        else:
            assert process.poll() is None,'Cannot establish live child identity'
        return True
    # The leader may exit before its workers. Its session/group remains owned
    # until the last member exits. Refuse a new leader with a reused PID above.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():continue
        try:member=process_identity(int(entry.name))
        except (FileNotFoundError,ProcessLookupError):continue
        if member['process_group']==process.pid:
            assert start_ticks is not None and member['session']==process.pid
            assert member['start_ticks']>=start_ticks,'Unexpected older process in new group'
            return True
    return False

def cleanup_owned_group(process,start_ticks):
    if owned_group_alive(process,start_ticks):
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+20
    while owned_group_alive(process,start_ticks) and time.monotonic()<deadline:
        process.poll();time.sleep(.1)
    if owned_group_alive(process,start_ticks):
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    process.wait(timeout=5)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--app',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--binary-sha256',required=True)
    parser.add_argument('--port',type=int,default=18772,help='Explicit prepared loopback port; existing listeners are refused')
    parser.add_argument('--guard-viewers',type=Path,help='Optional preserved viewer ports and PID/start identities; observation only')
    parser.add_argument('--cpus',help='Optional measured CPU affinity, comma-separated CPU IDs')
    args=parser.parse_args()
    app=args.app.resolve();run=args.run.resolve()
    assert sha(app/'SOURCE-MANIFEST.json')==args.manifest_sha256
    manifest=json.loads((app/'SOURCE-MANIFEST.json').read_text())
    for record in manifest['files']:
        file=(app/record['path']).resolve()
        assert file.is_relative_to(app) and file.is_file()
        assert file.stat().st_size==record['bytes'] and sha(file)==record['sha256'],record['path']
    config,identity=validate_run(app,run,args.binary_sha256,args.port);port=config['port']
    guards=json.loads(args.guard_viewers.read_text()) if args.guard_viewers else []
    assert not (run/'server-process.json').exists()
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1',port))!=0,'Port already occupied'
    env=os.environ.copy();env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='1'
    if args.cpus:
        requested={int(value) for value in args.cpus.split(',')}
        assert requested and min(requested)>=0 and requested<=os.sched_getaffinity(0)
        os.sched_setaffinity(0,requested)
        assert os.sched_getaffinity(0)==requested
    argv=['node',str(app/'launch_logged.cjs'),str(run)]
    check_preserved_viewers(guards)
    with (run/'server.stdout.log').open('xb') as stdout,(run/'server.stderr.log').open('xb') as stderr:
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
    start_ticks=None
    try:
        observed=process_identity(process.pid)
        assert observed['process_group']==observed['session']==process.pid
        start_ticks=observed['start_ticks']
        record={**observed,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,
                'source_manifest_sha256':args.manifest_sha256,'binary_sha256':args.binary_sha256,
                'prepared_run_sha256':{name:sha(run/name) for name in ['identity.json','server.json','worker.json']},
                'cpu_affinity':sorted(os.sched_getaffinity(process.pid))}
        with (run/'server-process.json').open('x') as out:out.write(json.dumps(record,indent=2)+'\n')
        check_preserved_viewers(guards)
        for _ in range(120):
            check_preserved_viewers(guards)
            assert process.poll() is None,'New viewer exited'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/snapshot',timeout=2) as response:
                    state=json.load(response)
                if state.get('ok') and state.get('frame'):
                    assert state['paused'] is True and state['tick']==0
                    assert state['frame']['tick']==0 and not state.get('renderFailure')
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/frame.png',timeout=2) as response:
                        png=response.read();frame_sha=response.headers['X-Rek-Frame-Sha256']
                    assert png[:8]==b'\x89PNG\r\n\x1a\n'
                    assert hashlib.sha256(png).hexdigest()==frame_sha==state['frame']['sha256']
                    check_preserved_viewers(guards)
                    (run/'startup-preview.png').write_bytes(png)
                    receipt={'process':record,'url':f'http://127.0.0.1:{port}/','paused':True,'tick':0,
                             'frame_bytes':len(png),'frame_sha256':frame_sha,'state':state}
                    (run/'STARTED.json').write_text(json.dumps(receipt,indent=2)+'\n')
                    print(json.dumps({k:v for k,v in receipt.items() if k!='state'}));return
            except (OSError,ValueError):
                pass
            time.sleep(.5)
        raise TimeoutError('New viewer did not become ready')
    except BaseException:
        cleanup_owned_group(process,start_ticks)
        raise

if __name__=='__main__':main()
