'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawn} = require('node:child_process');
const assert = require('node:assert/strict');

async function fixture(exitCode, delayVerification) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'rek-checkpoint-wait-test-'));
  const script = path.join(root, 'run_checkpoint_screen.sh');
  const run = path.join(root, 'train-test');
  const checkpointDir = path.join(run, 'checkpoints', 'rek_native5', 'train-test');
  fs.mkdirSync(checkpointDir, {recursive: true});
  fs.copyFileSync(path.join(__dirname, 'run_checkpoint_screen.sh'), script);
  // This stub replaces the evaluator only in the owned temporary fixture.
  // It cannot launch a policy, access CUDA, or invoke any real environment.
  fs.writeFileSync(path.join(root, 'run_diverse_policy_eval.sh'), '#!/usr/bin/env bash\nprintf "called\\n" >> "' + path.join(root, 'calls.txt') + '"\n');
  fs.writeFileSync(path.join(checkpointDir, '0000000000000016.bin'), 'not a model; hash-only test fixture\n');
  fs.writeFileSync(path.join(run, 'exit-code.txt'), String(exitCode) + '\n');
  if (!delayVerification) fs.writeFileSync(path.join(run, 'verified-warm-start.txt'), 'test verification\n');
  const child = spawn('bash', [script, root, 'train-test', '16', 'test-eval'], {stdio: ['ignore', 'pipe', 'pipe']});
  let stderr = '', stdout = '';
  child.stdout.on('data', chunk => { stdout += chunk; });
  child.stderr.on('data', chunk => { stderr += chunk; });
  const started = Date.now();
  const timer = delayVerification ? setTimeout(() => fs.writeFileSync(path.join(run, 'verified-warm-start.txt'), 'delayed test verification\n'), 200) : null;
  const deadline = setTimeout(() => child.kill('SIGKILL'), 12000);
  const code = await new Promise((resolve, reject) => { child.on('error', reject); child.on('exit', resolve); });
  clearTimeout(timer);clearTimeout(deadline);
  const calls = fs.existsSync(path.join(root, 'calls.txt')) ? fs.readFileSync(path.join(root, 'calls.txt'), 'utf8').trim().split('\n').length : 0;
  return {code, calls, elapsedMilliseconds: Date.now() - started, stderr, stdout, ownedFixtureDirectory: root};
}
(async () => {
  const delayed = await fixture(0, true);
  assert.equal(delayed.code, 0, delayed.stderr);
  assert.equal(delayed.calls, 3);
  assert(delayed.elapsedMilliseconds >= 200, 'Must not run before verification appears');
  const failed = await fixture(7, true);
  assert.equal(failed.code, 2);
  assert.equal(failed.calls, 0);
  const ready = await fixture(0, false);
  assert.equal(ready.code, 0, ready.stderr);
  assert.equal(ready.calls, 3);
  console.log(JSON.stringify({tests: 3, status: 'passed', delayedVerificationWaited: true, failedTrainingRejected: true, alreadyReadyAccepted: true, gpuInvocations: 0}));
})().catch(error => {console.error(error);process.exitCode=1;});
