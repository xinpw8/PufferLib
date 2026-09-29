"""Explicit root-operated startup. Never called by preparation or a timer."""
import datetime,json,subprocess,sys
import deployment_config as c
import launch_viewer as launcher

def command():
    return [sys.executable,str(c.HERE/'launch_viewer.py'),'--app',str(c.APP),'--run',str(c.RUN),
        '--manifest-sha256',c.APP_SHA,'--binary-sha256',c.BINARY_SHA,'--port',str(c.PORT),
        '--guard-viewers',str(c.HERE/'human-viewers.json'),'--cpus',c.CPUS]

def cap_command(parent):
    return [sys.executable,str(c.HERE/'disk_cap.py'),'--pid',str(parent['pid']),'--start-ticks',str(parent['start_ticks']),
        '--run',str(c.RUN),'--cap-bytes',str(c.RUN_CAP_BYTES),'--disk-fraction',str(c.DISK_CAP_FRACTION)]

def start_disk_cap():
    started=json.loads((c.RUN/'STARTED.json').read_text());parent=started['process']
    assert parent==json.loads((c.RUN/'server-process.json').read_text())
    assert launcher.process_identity(parent['pid'])['start_ticks']==parent['start_ticks']
    assert not (c.RUN/'disk-cap-process.json').exists()
    argv=cap_command(parent)
    with (c.RUN/'disk-cap.stdout.log').open('xb') as stdout,(c.RUN/'disk-cap.stderr.log').open('xb') as stderr:
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
    record={**launcher.process_identity(process.pid),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'argv':argv,'parent_pid':parent['pid'],'parent_start_ticks':parent['start_ticks'],
        'scope':'Stops only this viewer session when run-r9 exceeds its size cap or the disk passes the fraction'}
    with (c.RUN/'disk-cap-process.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

def main():
    c.verify_package()
    receipt=json.loads((c.HERE/'PREPARED.json').read_text());c.validate_prepared(receipt)
    launcher.check_preserved_viewers(c.guards())
    assert not (c.RUN/'server-process.json').exists(),'Never start over an existing viewer'
    subprocess.run(command(),check=True)
    start_disk_cap()

if __name__=='__main__':main()
