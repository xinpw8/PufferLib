from pathlib import Path
import datetime, hashlib, json

root = Path(__file__).resolve().parent
repo = Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile')
pins = json.loads((root / 'preexisting-source-pins.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
changed = [x['path'] for x in pins if sha(repo / x['path']) != x['sha256']]
assert not changed, changed
assert (root / 'cpu/exit-code.txt').read_text().strip() == '0'
assert all(x['exit_code'] == 0 for x in json.loads((root/'node-receipt.json').read_text())['runs'])
files = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
         for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'FINAL-RECEIPT.json']
receipt = {
 'schema':'rek.preexisting_checkpoint_checks.v1',
 'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'Read-only repository audit and CPU/Node tests of 45 preexisting modified/untracked files. No repository edits, Git mutation, GPU, physics, policy or application input.',
 'repo':str(repo), 'baseline_head':'1d8f42b83c6aac6ffd3b88f3fe29f80120de8179',
 'source_changes_during_all_checks':changed, 'passed':True,
 'checks':{'node_tests':83,'will_connect':34,'will_connect_regression':30,
   'will_connect_diagnostics':'passed, no numeric assertion count emitted',
   'previous_action_assertions':11462,
   'protocol_assertions':{'scaled_polar_xy':112,'observable_balance':134,'observable_balance_prev_action':368},
   'encoder_assertions':{'observable_balance':20377,'observable_balance_prev_action':20571},
   'encoder_cases_each':92,'git_diff_check_exit_code':0},
 'secret_pattern_scan':{'files':45,'findings':[], 'limit':'Bounded credential-pattern scan, not an exhaustive guarantee.'},
 'encoder_model_fixture':'Explicit synthetic identity-only XML; geometry comes from test fixtures, no private model or dynamics.',
 'integration_verdict':'Preserve the preexisting 45 files unchanged in a first checkpoint. Package the tested native clone as its own source overlay and dedicated build path. Do not overwrite shared eval_worker.cpp: the legacy fast runtime lacks its two unconditional direct-command APIs.',
 'files':files
}
out = root / 'FINAL-RECEIPT.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt,f,indent=2); f.write('\n')
print(json.dumps({'path':str(out),'sha256':sha(out),'pinned_files':len(files),'passed':True}))
