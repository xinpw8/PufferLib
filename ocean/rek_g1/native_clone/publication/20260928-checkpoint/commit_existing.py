from pathlib import Path
import hashlib,json,subprocess
repo=Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile')
snapshot=json.loads(Path(r'R:\pufferlib\rek-evidence\2026-09-28\github-checkpoint-r1\before\SNAPSHOT.json').read_text())
for f in snapshot['files']:
    assert hashlib.sha256((repo/f['path']).read_bytes()).hexdigest()==f['sha256'],f['path']
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==snapshot['head']
subprocess.run(['git','-C',str(repo),'add','--',*[f['path'] for f in snapshot['files']]],check=True)
subprocess.run(['git','-C',str(repo),'diff','--cached','--check'],check=True)
subprocess.run(['git','-C',str(repo),'commit','-m','Preserve REK observable-action and contact diagnostics'],check=True)
print(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
