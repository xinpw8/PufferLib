from pathlib import Path
import datetime,hashlib,json,re,subprocess

repo=Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile')
out=Path(r'R:\pufferlib\rek-evidence\2026-09-28\github-checkpoint-r1\before')
out.mkdir(parents=True,exist_ok=False)
def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
assert not git('diff','--cached','--name-only'), 'Preserve existing staged changes before checkpoint'
paths=sorted(set(x.decode() for x in (git('diff','--name-only','-z')+git('ls-files','--others','--exclude-standard','-z')).split(b'\0') if x))
records=[];findings=[]
patterns={
    'github_token':rb'gh[pousr]_[A-Za-z0-9]{30,}',
    'github_fine_grained':rb'github_pat_[A-Za-z0-9_]{40,}',
    'slack_token':rb'xox[baprs]-[A-Za-z0-9-]{25,}',
    'aws_key':rb'\bAKIA[0-9A-Z]{16}\b',
    'private_key':rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\r?\n[A-Za-z0-9+/=]{30,}',
    'jwt':rb'\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}',
}
for name in paths:
    source=repo/name
    if not source.is_file():raise RuntimeError('Unexpected missing/deleted path: '+name)
    data=source.read_bytes()
    for kind,pattern in patterns.items():
        if re.search(pattern,data):findings.append({'path':name,'kind':kind})
    target=out/'files'/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(data)
    digest=hashlib.sha256(data).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
    records.append({'path':name,'bytes':len(data),'sha256':digest})
(out/'tracked.patch').write_bytes(git('diff','--binary'))
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repo':str(repo),
    'head':git('rev-parse','HEAD').decode().strip(),'branch':git('branch','--show-current').decode().strip(),
    'files':records,'credential_pattern_findings':findings}
(out/'SNAPSHOT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'destination':str(out),'files':len(paths),'bytes':sum(r['bytes'] for r in records),'credential_pattern_findings':findings}))
assert not findings, 'Review credential pattern findings before publication'
