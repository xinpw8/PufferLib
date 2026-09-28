"""CPU-only recheck of the exact existing-host dependency paths in a manifest."""
from pathlib import Path
import argparse, hashlib, json, sys

def verify(manifest):
    failures=[]; total=0
    for entry in manifest['files']:
        path=Path(entry['path'])
        if not path.is_file():
            failures.append({'path':str(path),'reason':'missing'});continue
        if path.stat().st_size!=entry['bytes']:
            failures.append({'path':str(path),'reason':'size'});continue
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=entry['sha256']:
            failures.append({'path':str(path),'reason':'sha256'});continue
        total+=1
    return {'success':not failures,'verified_files':total,'failures':failures,
            'scope':'File identity only; no process, library, GPU, or simulator initialization.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path)
    args=parser.parse_args();result=verify(json.loads(args.manifest.read_text()))
    print(json.dumps(result,indent=2));sys.exit(0 if result['success'] else 1)
