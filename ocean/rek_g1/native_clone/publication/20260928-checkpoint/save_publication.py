from pathlib import Path
import datetime, hashlib, json, shutil, subprocess

root = Path(__file__).resolve().parent
repo = Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile')
target = Path(r'R:\pufferlib\rek-evidence\2026-09-28\github-checkpoint-r1\published-r1')
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
remote = subprocess.check_output(['git', 'ls-remote', '--exit-code', 'github', 'refs/heads/codex/puffysics-training-profile'], cwd=repo, text=True).split()[0]
assert head == remote == '19e3fe90a4a3a14394942776cf13073d887e63fd'
assert not subprocess.check_output(['git', '-c', 'core.longpaths=true', 'status', '--porcelain'], cwd=repo)
target.mkdir(parents=True, exist_ok=False)
records = []
for name in ['COPIED.json', 'STAGE-MANIFEST.json', 'STAGED-VERIFIED.json', 'snapshot_worktree.py', 'commit_existing.py', 'copy_payload.py', 'finish_package.py', 'verify_stage.py', 'save_publication.py']:
    source = root / name
    destination = target / name
    shutil.copyfile(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    records.append({'path': name, 'bytes': destination.stat().st_size, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
receipt = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'repository': 'https://github.com/xinpw8/PufferLib',
           'branch': 'codex/puffysics-training-profile', 'verified_remote_head': remote, 'local_head': head,
           'worktree_clean': True, 'checkpoint_commits': ['a1ab1f5eb8729136f98249feac50e523ac52955f', head],
           'preserved_files': records, 'raw_live_captures': 'Remain on NAS, excluded from public snapshot.'}
(target / 'PUBLICATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'receipt': str(target / 'PUBLICATION.json'), 'head': head, 'verified': True}))
