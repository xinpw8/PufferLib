"""Read closed, explicitly selected evidence from this task's Spark root."""
import argparse,hashlib,json,stat
from pathlib import Path,PurePosixPath
from remote_task import transport,REMOTE,ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('destination');p.add_argument('paths',nargs='+');a=p.parse_args()
    dest=ROOT/a.destination
    if dest.exists():raise ValueError('fresh readback directory required')
    dest.mkdir();records=[]
    with transport.connect() as client,client.open_sftp() as sftp:
        def fetch(rel):
            parts=PurePosixPath(rel).parts
            if not parts or '..' in parts or rel.startswith('/') or any(x in ('game','home','wineprefix') for x in parts):raise ValueError('out-of-scope path')
            remote=REMOTE+'/'+rel;info=sftp.lstat(remote)
            if stat.S_ISLNK(info.st_mode):raise ValueError('symlink')
            if stat.S_ISDIR(info.st_mode):
                for item in sorted(sftp.listdir(remote)):fetch(rel+'/'+item)
                return
            if not stat.S_ISREG(info.st_mode):raise ValueError('not regular')
            with sftp.open(remote,'rb') as f:before=f.read()
            target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(before)
            with sftp.open(remote,'rb') as f:after=f.read()
            digest=hashlib.sha256(before).hexdigest()
            if before!=after or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('unstable readback')
            records.append({'path':rel,'bytes':len(before),'sha256':digest})
        for rel in a.paths:fetch(rel)
    receipt={'remote_root':REMOTE,'files':records}
    with (dest/'COPY-VERIFICATION.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({'destination':str(dest),'files':len(records),'bytes':sum(x['bytes'] for x in records)}))
if __name__=='__main__':main()
