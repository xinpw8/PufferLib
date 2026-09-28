"""Start a fresh, pinned viewer on an unused loopback port, initially paused."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, socket, subprocess, time, urllib.request

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--app',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--cpus',help='Optional measured CPU affinity, comma-separated CPU IDs')
    args=parser.parse_args()
    app=args.app.resolve();run=args.run.resolve()
    assert sha(app/'SOURCE-MANIFEST.json')==args.manifest_sha256
    manifest=json.loads((app/'SOURCE-MANIFEST.json').read_text())
    for record in manifest['files']:
        file=(app/record['path']).resolve()
        assert file.is_relative_to(app) and file.is_file()
        assert file.stat().st_size==record['bytes'] and sha(file)==record['sha256'],record['path']
    config=json.loads((run/'server.json').read_text());port=config['port']
    assert port==18772,'Dedicated improved viewer port required'
    identity=json.loads((run/'identity.json').read_text())
    for filename,digest in identity['filePins'].items():
        assert sha(Path(filename))==digest,filename
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
    with (run/'server.stdout.log').open('xb') as stdout,(run/'server.stderr.log').open('xb') as stderr:
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
    record={'pid':process.pid,'start_ticks':Path(f'/proc/{process.pid}/stat').read_text().rsplit(') ',1)[1].split()[19],
            'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,'source_manifest_sha256':args.manifest_sha256,
            'cpu_affinity':sorted(os.sched_getaffinity(process.pid))}
    (run/'server-process.json').write_text(json.dumps(record,indent=2)+'\n')
    try:
        for _ in range(120):
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
        process.terminate()
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        raise

if __name__=='__main__':main()
