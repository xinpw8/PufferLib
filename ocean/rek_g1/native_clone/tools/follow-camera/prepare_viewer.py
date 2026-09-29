"""CPU-only preparation on Spark. No Node server, native worker or watcher launch."""
import datetime,json,socket,subprocess
import deployment_config as c
import launch_viewer as launcher

def main():
    c.verify_package()
    assert not c.RUN.exists(),'Fresh run-r9 required'
    assert not (c.HERE/'prepared-inputs').exists(),'Fresh preparation inputs required'
    assert not (c.HERE/'PREPARED.json').exists(),'Existing preparation must be preserved'
    assert c.sha(c.APP/'SOURCE-MANIFEST.json')==c.APP_SHA
    manifest=json.loads((c.APP/'SOURCE-MANIFEST.json').read_text())
    assert manifest['revision']==10
    for row in manifest['files']:
        q=(c.APP/row['path']).resolve()
        assert q.is_relative_to(c.APP) and q.is_file() and not q.is_symlink()
        assert q.stat().st_size==row['bytes'] and c.sha(q)==row['sha256']
    assert c.sha(c.BINARY)==c.BINARY_SHA
    for name,digest in c.BASE_PINS.items():assert c.sha(c.BASELINE_RUN/name)==digest,'Existing baseline configuration changed'
    launcher.check_preserved_viewers(c.guards())
    with socket.socket() as probe:assert probe.connect_ex(('127.0.0.1',c.PORT))!=0,'Port 18775 already occupied'
    baseline=json.loads((c.HERE/'baseline/worker.json').read_text())
    worker=c.worker_template(baseline)
    env=json.loads((c.HERE/'baseline/server.json').read_text())['backends'][0]['env']
    inputs=c.HERE/'prepared-inputs';inputs.mkdir()
    for name,value in [('worker-template.json',worker),('env.json',env)]:
        with (inputs/name).open('x') as f:f.write(json.dumps(value,indent=2)+'\n')
    command=['node',str(c.APP/'prepare.cjs'),str(inputs/'worker-template.json'),str(c.BINARY),str(c.RUN),str(inputs/'env.json'),str(c.PORT)]
    result=subprocess.run(command,capture_output=True,text=True)
    for name,value in [('prepare.stdout.txt',result.stdout),('prepare.stderr.txt',result.stderr),('prepare.exit.txt',str(result.returncode)+'\n')]:
        with (c.HERE/name).open('x') as f:f.write(value)
    result.check_returncode()
    c.validate_worker(json.loads((c.RUN/'worker.json').read_text()),baseline)
    server,identity=launcher.validate_run(c.APP,c.RUN,c.BINARY_SHA,c.PORT)
    assert identity['env']==env
    launcher.check_preserved_viewers(c.guards())
    receipt={'schema':'rek.follow_camera_viewer_prepared.v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'run':str(c.RUN),'app':str(c.APP),'port':c.PORT,'app_manifest_sha256':c.APP_SHA,'binary_sha256':c.BINARY_SHA,
        'graph_mode':'on','changes':'r10 app and render-only follow camera binary; r10 render-model path. All other worker fields/environment equal pinned run-r8.',
        'prepared_files':{name:c.sha(c.RUN/name) for name in ['worker.json','server.json','identity.json','league.json']},
        'guard_sha256':c.GUARDS_SHA,'launcher_sha256':c.LAUNCHER_SHA,'launched':False,'command':command}
    with (c.HERE/'PREPARED.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
