"""Start only the passive watcher bound to the newly verified paused viewer."""
import datetime,json,subprocess,sys,time,urllib.request
import deployment_config as c
import launch_viewer as launcher

def main():
    c.verify_package()
    prepared=json.loads((c.HERE/'PREPARED.json').read_text());c.validate_prepared(prepared)
    started=json.loads((c.RUN/'STARTED.json').read_text());parent=started['process']
    assert started['paused'] is True and started['tick']==0 and started['url']==f'http://127.0.0.1:{c.PORT}/'
    assert parent==json.loads((c.RUN/'server-process.json').read_text())
    assert launcher.process_identity(parent['pid'])['start_ticks']==parent['start_ticks']
    with urllib.request.urlopen(f'http://127.0.0.1:{c.PORT}/api/snapshot',timeout=2) as response:state=json.load(response)
    assert state.get('ok') is True and state.get('paused') is True and state.get('tick')==0
    launcher.check_preserved_viewers(c.guards())
    output=c.RUN/'resource.jsonl';stop=c.RUN/'STOP_RESOURCE_WATCH'
    assert not output.exists() and not stop.exists() and not (c.RUN/'resource-watch-process.json').exists()
    argv=[sys.executable,str(c.HERE/'resource_watch.py'),'--pid',str(parent['pid']),'--start-ticks',str(parent['start_ticks']),
        '--native-exe',str(c.BINARY),'--output',str(output),'--stop-file',str(stop)]
    process=None
    try:
        with (c.RUN/'resource-watch.stdout.log').open('xb') as stdout,(c.RUN/'resource-watch.stderr.log').open('xb') as stderr:
            process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        identity=launcher.process_identity(process.pid)
        assert identity['session']==identity['process_group']==process.pid
        for _ in range(40):
            assert process.poll() is None,'Resource watcher exited during startup'
            if output.exists() and output.stat().st_size:
                with output.open() as stream:header=json.loads(stream.readline())
                assert header['event']=='header' and header['viewer_pid']==parent['pid'] and header['viewer_start_ticks']==parent['start_ticks']
                assert header['native_exe']==str(c.BINARY)
                break
            time.sleep(.1)
        else:raise TimeoutError('Resource watcher did not publish header')
        record={**identity,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,
            'source_sha256':c.RESOURCE_SHA,'parent_pid':parent['pid'],'parent_start_ticks':parent['start_ticks'],
            'output':str(output),'stop_file':str(stop),'scope':'Passive telemetry only; own STOP or parent exit terminates watcher'}
        with (c.RUN/'resource-watch-process.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record))
    except BaseException:
        if process is not None:
            with stop.open('x') as f:f.write('Watcher startup failed; own passive process only.\n')
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:pass
        raise

if __name__=='__main__':main()
