"""Copy only the closed, named private inventory evidence; no runtime writes."""
import datetime,hashlib,io,json,pathlib,subprocess,zipfile
ROOT=pathlib.Path(__file__).resolve().parent
DEST=ROOT/'closed-inventory-r2'
assert not DEST.exists(), 'Preserve prior readback'
remote=r'''
import pathlib,json,subprocess,hashlib,datetime,zipfile,io,sys
base=pathlib.Path('/home/spark-advantage/rek-training/rek-original-model-20260928-r2')
root=base/'unity-harness-inventory-r2'
container='rek-original-model-inventory-20260928-r2'
inspection=json.loads(subprocess.check_output(['docker','inspect',container]))[0]
assert inspection['State']['Status']=='exited' and not inspection['State']['Running']
paths={p: 'session/'+p.relative_to(root).as_posix() for p in root.glob('*.json') if p.is_file()}
for dirname in ['input','out']:
 for p in (root/dirname).rglob('*'):
  if p.is_file():paths[p]='session/'+p.relative_to(root).as_posix()
for p in [root/'run.started',root/'game/BepInEx/LogOutput.log',root/'game/BepInEx/ErrorLog.log']:
 if p.is_file():paths[p]='session/'+p.relative_to(root).as_posix()
for p in (base/'package-r2/isolation').glob('container-*-receipt.json'):
 paths[p]='isolation/'+p.name
buf=io.BytesIO();files=[]
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p,rel in sorted(paths.items()):
  assert p.resolve().is_relative_to(base.resolve()) and not p.is_symlink()
  before=p.stat();b=p.read_bytes();after=p.stat()
  assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  z.writestr(rel,b)
  files.append({'path':rel,'source_path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'source_mtime_ns':after.st_mtime_ns})
 z.writestr('container-inspect.json',json.dumps(inspection,indent=2)+'\n')
 z.writestr('SOURCE-MANIFEST.json',json.dumps({'schema':'rek.original_model_inventory.closed_evidence.v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_root':str(root),'container':container,'state':inspection['State'],'files':files},indent=2)+'\n')
sys.stdout.buffer.write(buf.getvalue())
'''
archive=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=7','-o','ForwardX11=no','-o','ClearAllForwardings=yes','dgx_spark','python3','-'],input=remote.encode())
DEST.mkdir()
with zipfile.ZipFile(io.BytesIO(archive)) as z:
 for entry in z.infolist():
  rel=pathlib.PurePosixPath(entry.filename)
  assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in entry.filename
  target=DEST.joinpath(*rel.parts);assert target.resolve().is_relative_to(DEST.resolve())
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(entry))
manifest=json.loads((DEST/'SOURCE-MANIFEST.json').read_text())
for rec in manifest['files']:
 data=(DEST/rec['path']).read_bytes();assert len(data)==rec['bytes'] and hashlib.sha256(data).hexdigest()==rec['sha256']
result={'schema':'rek.original_model_inventory.closed_copy_verification.v1','source_manifest_sha256':hashlib.sha256((DEST/'SOURCE-MANIFEST.json').read_bytes()).hexdigest(),'container_inspect_sha256':hashlib.sha256((DEST/'container-inspect.json').read_bytes()).hexdigest(),'transport_archive_sha256':hashlib.sha256(archive).hexdigest(),'files':len(manifest['files']),'bytes':sum(x['bytes'] for x in manifest['files']),'all_source_local_hashes_match':True,'destination':str(DEST)}
(DEST/'COPY-VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

