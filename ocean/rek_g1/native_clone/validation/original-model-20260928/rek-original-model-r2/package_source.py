"""Create an immutable local source package only. No SSH or runtime operations."""
from pathlib import Path
import datetime,hashlib,json,tarfile
root=Path(__file__).resolve().parent
package=root/'package';package.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=['README.md','BUILD-RECEIPT.json','fixture.json','prepare_inputs.py','test_prepare_inputs.py','probe/Plugin.cs','probe/RekOriginalModelInventory.csproj','isolation/prepare_runtime.py','isolation/orchestrate.py','revision/r1-to-r2.diff','revision/INDEPENDENT-REVIEW.json']
files += [p.relative_to(root).as_posix() for sub in ['api','review'] for p in (root/sub).iterdir() if p.is_file()]
entries=[]
for rel in files+['probe/bin/Release/net6.0/RekOriginalModelInventory.dll']:
 src=root/rel;targetrel='bin/RekOriginalModelInventory.dll' if rel.endswith('.dll') else rel
 dest=package/targetrel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes());assert sha(dest)==sha(src)
 entries.append({'path':targetrel,'source_path':str(src),'bytes':dest.stat().st_size,'sha256':sha(dest)})
manifest={'schema':'rek.original_model_inventory.source_package.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'revision':'r2','files':entries,'runtime_executed_at_package_creation':False,'selection':'direct_observed_catalog_enumeration','original_TryGetById_invoked':False}
(package/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=root/'original-model-inventory-package-r2.tar'
with tarfile.open(archive,'x') as tar:
 for p in sorted(package.rglob('*')):
  if p.is_file():tar.add(p,arcname=p.relative_to(package).as_posix(),recursive=False)
with tarfile.open(archive) as tar:
 for member in tar:
  assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
  assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==sha(package/member.name)
receipt={'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'source_manifest_sha256':sha(package/'MANIFEST.json'),'files':len(entries),'runtime_executed':False}
(root/'PACKAGE-RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

