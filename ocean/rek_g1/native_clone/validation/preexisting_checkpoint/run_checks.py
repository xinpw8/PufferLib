from pathlib import Path
import subprocess,json,hashlib,datetime,re,os
root=Path(__file__).resolve().parent
repo=Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile')
node=Path(r'C:\Users\Daniel\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
files=sorted(set(subprocess.check_output(['git','diff','--name-only'],cwd=repo,text=True).splitlines()+subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=repo,text=True).splitlines()))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
pins=[{'path':f,'bytes':(repo/f).stat().st_size,'sha256':sha(repo/f)} for f in files]
(root/'preexisting-source-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
patterns={
'private_key_header':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
'github_credential_shape':re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})'),
'aws_access_key_shape':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
'google_api_key_shape':re.compile(r'AIza[A-Za-z0-9_-]{30,}'),
'credential_in_url':re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
'literal_secret_assignment':re.compile(r'''(?i)(?:api[_-]?key|client[_-]?secret|access[_-]?token|password)\s*[:=]\s*["'][A-Za-z0-9_+/=-]{16,}["']'''),
}
findings=[]
for f in files:
 text=(repo/f).read_text(errors='replace')
 for reason,pattern in patterns.items():
  matches=list(pattern.finditer(text))
  if matches:findings.append({'path':f,'reason':reason,'count':len(matches)})
(root/'secret-shape-scan.json').write_text(json.dumps({'scope':'all19modified+26untracked source and derived-evidence files; values not emitted','findings':findings},indent=2)+'\n')
commands=[('node-suites',[str(node),'--test','ocean/rek_g1/league/public/app.test.cjs','ocean/rek_g1/native5/live_transfer_run.test.cjs','ocean/rek_g1/native5/score_head_data.test.cjs']),('diff-check',['git','diff','--check'])]
results=[]
for name,cmd in commands:
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 p=subprocess.run(cmd,cwd=repo,capture_output=True)
 (root/(name+'.stdout.txt')).write_bytes(p.stdout);(root/(name+'.stderr.txt')).write_bytes(p.stderr)
 (root/(name+'.exit-code.txt')).write_text(str(p.returncode)+'\n')
 results.append({'name':name,'command':cmd,'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode})
(root/'node-receipt.json').write_text(json.dumps({'runs':results,'source_changes_during_tests':[x['path'] for x in pins if sha(repo/x['path'])!=x['sha256']]},indent=2)+'\n')
print(json.dumps({'runs':results,'secret_shape_findings':findings}))
