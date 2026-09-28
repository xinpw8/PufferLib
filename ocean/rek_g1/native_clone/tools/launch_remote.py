"""Start only this clone's prepared loopback viewer, initially paused."""
from pathlib import Path
import datetime,hashlib,json,os,socket,subprocess,time,urllib.request

root=Path('/home/spark-advantage/rek-training/rek-native-clone-20260927-r1')
run=root/'run-r4';config=json.loads((run/'server.json').read_text());port=config['port']
identity=json.loads((run/'identity.json').read_text())
for name,digest in identity['filePins'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
assert port==18771
with socket.socket() as probe:
    if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Viewer port already occupied; no process replaced')
assert not (run/'server-process.json').exists()
env=os.environ.copy();env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='1'
with (run/'server.stdout.log').open('xb') as stdout,(run/'server.stderr.log').open('xb') as stderr:
    process=subprocess.Popen(['node',str(root/'app-r3/launch_logged.cjs'),str(run)],stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
record=dict(pid=process.pid,start_ticks=Path('/proc/%d/stat'%process.pid).read_text().split(') ')[1].split()[19],
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=['node',str(root/'app-r3/launch_logged.cjs'),str(run)])
(run/'server-process.json').write_text(json.dumps(record,indent=2)+'\n')
try:
    for _ in range(60):
        if process.poll() is not None:raise RuntimeError('New viewer exited: '+str(process.returncode))
        try:
            with urllib.request.urlopen('http://127.0.0.1:%d/api/snapshot'%port,timeout=2) as response:state=json.load(response)
            if state.get('ok'):
                assert state['paused'] is True and state['tick']==0,state
                with urllib.request.urlopen('http://127.0.0.1:%d/frame.png'%port,timeout=2) as response:png=response.read()
                assert png[:8]==b'\x89PNG\r\n\x1a\n'
                (run/'startup-preview.png').write_bytes(png)
                receipt=dict(process=record,url='http://127.0.0.1:%d/'%port,paused=True,tick=0,frame_bytes=len(png),frame_sha256=hashlib.sha256(png).hexdigest(),state=state)
                (run/'STARTED.json').write_text(json.dumps(receipt,indent=2)+'\n')
                print(json.dumps({k:v for k,v in receipt.items() if k!='state'}),flush=True);break
        except (OSError,ValueError):pass
        time.sleep(.5)
    else:raise TimeoutError('New viewer did not become ready')
except BaseException:
    process.terminate()
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:process.kill();process.wait()
    raise
