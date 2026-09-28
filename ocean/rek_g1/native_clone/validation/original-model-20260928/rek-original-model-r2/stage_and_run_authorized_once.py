"""Stage then execute the one parent-approved r2 passive inventory; no retry."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,shlex
ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'package';archive=ROOT/'original-model-inventory-package-r2.tar';m=PACKAGE/'MANIFEST.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(archive)=='0631a136d4cc5cd35d4cf4cbbdf7f95bc62415a66a3f468b401aa926e779a951'
assert sha(m)=='9e9a8bcd20d211e6db2597f88d7bab22a3ed08919fb763f28e1dd832ee6a46a8'
entries=json.loads(m.read_text())['files']
for e in entries:assert (PACKAGE/e['path']).stat().st_size==e['bytes'] and sha(PACKAGE/e['path'])==e['sha256']
spec=importlib.util.spec_from_file_location('owned_passive',r'C:\rekagent\work\rek-native-clone-20260927-r1\passive-support-r2\passive.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
client=mod.connect();sftp=client.open_sftp();remote='/home/spark-advantage/rek-training/rek-original-model-20260928-r2';execution=ROOT/'execution';execution.mkdir(exist_ok=False)
def run(command,name,timeout=120,required=True):
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    inp,out,err=client.exec_command(command,timeout=timeout);stdout=out.read();stderr=err.read();code=out.channel.recv_exit_status()
    (execution/(name+'.stdout')).write_bytes(stdout);(execution/(name+'.stderr')).write_bytes(stderr)
    receipt={'command':command,'exit_code':code,'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (execution/(name+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'stage':name,**receipt}),flush=True)
    if required:assert code==0,f'{name} failed {code}'
    return stdout.decode()
try:
    image=run("docker image inspect --format '{{.Id}}' rek-original-functions:20260927-r1",'00-image').strip();assert image=='sha256:5fb2c698c065d4502b3c09f3b847efac6aa5428541c4b4f26110e3a530cae879'
    run('test ! -e '+shlex.quote(remote)+' && mkdir '+shlex.quote(remote),'01-fresh-root')
    target=remote+'/'+archive.name;sftp.put(str(archive),target)
    with sftp.open(target,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==sha(archive)
    pkg=remote+'/package-r2';run('mkdir '+shlex.quote(pkg)+' && tar -xf '+shlex.quote(target)+' -C '+shlex.quote(pkg),'02-unpack')
    for e in entries+[{'path':'MANIFEST.json','bytes':m.stat().st_size,'sha256':sha(m)}]:
        with sftp.open(pkg+'/'+e['path'],'rb') as f:b=f.read()
        assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
    run('python3 '+shlex.quote(pkg+'/isolation/prepare_runtime.py')+' --prepare','03-prepare-runtime',180)
    run('python3 '+shlex.quote(pkg+'/isolation/prepare_runtime.py')+' --install-plugin '+shlex.quote(pkg+'/bin/RekOriginalModelInventory.dll')+' --plugin-sha256 bc8df76dbc4664a539fd80c2bb6634c3f03f8ac0bf1bda19f6c550cb5014cb35','04-plugin')
    session=remote+'/unity-harness-inventory-r2'
    run('python3 '+shlex.quote(pkg+'/prepare_inputs.py')+' --fixture '+shlex.quote(pkg+'/fixture.json')+' --input '+shlex.quote(session+'/input')+' --private-game '+shlex.quote(session+'/game')+' --run-id original_model_inventory_20260928_r2','05-inputs')
    run('python3 '+shlex.quote(pkg+'/isolation/orchestrate.py')+' --freeze-input','06-freeze')
    run('python3 '+shlex.quote(pkg+'/isolation/orchestrate.py')+' --create','07-create')
    actual=run("docker inspect --format '{{.Image}}' rek-original-model-inventory-20260928-r2",'08-created-image').strip();assert actual==image
    (execution/'STAGED-READY.json').write_text(json.dumps({'remote_package':pkg,'remote_session':session,'package_sha256':sha(archive),'source_manifest_sha256':sha(m),'container_image':actual,'created_not_started':True},indent=2)+'\n')
    run('python3 '+shlex.quote(pkg+'/isolation/orchestrate.py')+' --start','09-start',240,False)
finally:sftp.close();client.close()
