from pathlib import Path
import datetime,hashlib,importlib.util,json,shlex,tarfile
ROOT=Path(__file__).resolve().parent;PACKAGE=ROOT/'package';PACKAGE.mkdir(exist_ok=False)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
files=['README.md','BUILD-RECEIPT.json','fixture.json','prepare_inputs.py','test_prepare_inputs.py','probe/Plugin.cs','probe/RekOriginalModelInventory.csproj','isolation/prepare_runtime.py','isolation/orchestrate.py']
files += [p.relative_to(ROOT).as_posix() for base in ['api','review'] for p in (ROOT/base).glob('*') if p.is_file()]
entries=[]
for rel in files+['probe/bin/Release/net6.0/RekOriginalModelInventory.dll']:
    src=ROOT/rel;destrel='bin/RekOriginalModelInventory.dll' if rel.endswith('.dll') else rel;dest=PACKAGE/destrel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes());assert sha(dest)==sha(src)
    entries.append({'path':destrel,'source_path':str(src),'bytes':dest.stat().st_size,'sha256':sha(dest)})
m=PACKAGE/'MANIFEST.json';m.write_text(json.dumps({'schema':'rek.original_model_inventory.source_package.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':entries,'runtime_executed_at_package_creation':False},indent=2)+'\n')
archive=ROOT/'original-model-inventory-package-r1.tar'
with tarfile.open(archive,'x') as t:
    for p in sorted(PACKAGE.rglob('*')):
        if p.is_file():t.add(p,arcname=p.relative_to(PACKAGE).as_posix(),recursive=False)
with tarfile.open(archive) as t:
    assert all(x.isfile() and not x.name.startswith('/') and '..' not in Path(x.name).parts for x in t.getmembers())
spec=importlib.util.spec_from_file_location('owned_passive',r'C:\rekagent\work\rek-native-clone-20260927-r1\passive-support-r2\passive.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
client=mod.connect();sftp=client.open_sftp();remote='/home/spark-advantage/rek-training/rek-original-model-20260928-r1';execution=ROOT/'execution';execution.mkdir(exist_ok=False)
def run(command,name,timeout=120):
    inp,out,err=client.exec_command(command,timeout=timeout);stdout=out.read();stderr=err.read();code=out.channel.recv_exit_status();(execution/(name+'.stdout')).write_bytes(stdout);(execution/(name+'.stderr')).write_bytes(stderr)
    receipt={'command':command,'exit_code':code,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(execution/(name+'.json')).write_text(json.dumps(receipt,indent=2)+'\n');assert code==0, f'{name} failed code{code}';return stdout.decode()
try:
    image=run("docker image inspect --format '{{.Id}}' rek-original-functions:20260927-r1",'00-image').strip();assert image=='sha256:5fb2c698c065d4502b3c09f3b847efac6aa5428541c4b4f26110e3a530cae879','image changed'
    run('test ! -e '+shlex.quote(remote)+' && mkdir '+shlex.quote(remote),'01-fresh-root')
    target=remote+'/original-model-inventory-package-r1.tar';sftp.put(str(archive),target)
    with sftp.open(target,'rb') as f:remotehash=hashlib.sha256(f.read()).hexdigest()
    assert remotehash==sha(archive)
    pkg=remote+'/package-r1';run('mkdir '+shlex.quote(pkg)+' && tar -xf '+shlex.quote(target)+' -C '+shlex.quote(pkg),'02-unpack')
    # Verify every staged member before executing any package code.
    for e in entries+[{'path':'MANIFEST.json','bytes':m.stat().st_size,'sha256':sha(m)}]:
        with sftp.open(pkg+'/'+e['path'],'rb') as f:b=f.read()
        assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],e['path']
    run('python3 '+shlex.quote(pkg+'/isolation/prepare_runtime.py')+' --prepare','03-prepare-runtime',180)
    run('python3 '+shlex.quote(pkg+'/isolation/prepare_runtime.py')+' --install-plugin '+shlex.quote(pkg+'/bin/RekOriginalModelInventory.dll')+' --plugin-sha256 79aa0a0ee713401a19d7e3ca192b577f5fc3981910bff86c652a149a87a0334c','04-plugin')
    session=remote+'/unity-harness-inventory-r1'
    run('python3 '+shlex.quote(pkg+'/prepare_inputs.py')+' --fixture '+shlex.quote(pkg+'/fixture.json')+' --input '+shlex.quote(session+'/input')+' --private-game '+shlex.quote(session+'/game')+' --run-id original_model_inventory_20260928_r1','05-inputs')
    run('python3 '+shlex.quote(pkg+'/isolation/orchestrate.py')+' --freeze-input','06-freeze')
    run('python3 '+shlex.quote(pkg+'/isolation/orchestrate.py')+' --create','07-create')
    actual=run("docker inspect --format '{{.Image}}' rek-original-model-inventory-20260928-r1",'08-created-image').strip();assert actual==image
    receipt={'remote_package':pkg,'remote_session':session,'package_sha256':sha(archive),'source_manifest_sha256':sha(m),'container_image':actual,'created_not_started':True,'archive_bytes':archive.stat().st_size};(execution/'STAGED-READY.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
finally:sftp.close();client.close()
