'use strict';
// Mechanical extraction of the pinned pre-change scheduler kernel bodies.
const fs = require('node:fs');
const path = require('node:path');
const cp = require('node:child_process');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const output = process.argv[2];
assert(output, 'Usage: node test_g1_semantic_direct_baseline.cjs NEW_OUTPUT_HEADER');
assert(!fs.existsSync(output), 'Refusing existing baseline file');
const commit = '4abf6a9e62a452d4638d22da04d7a29177297741';
const source = cp.execFileSync('git', ['show', commit + ':ocean/rek_g1/g1_semantic_scheduler_cuda.cu'], {cwd: __dirname});
const sha = data => crypto.createHash('sha256').update(data).digest('hex');
assert.equal(sha(source), '2835224ddf4fdc06fd27beee06b6ded4f5dd5d82c991becd04a23d317fac45f7');
const text = source.toString('utf8');
assert(text.startsWith('#define REK_G1_CUDA_DEVICE 1\n'));
const boundary = text.indexOf('extern "C" cudaError_t rek_g1_cuda_semantic_table_init(');
assert(boundary > 0);
const kernelBodies = text.slice(text.indexOf('\n') + 1, boundary);
fs.mkdirSync(path.dirname(output), {recursive: true});
fs.writeFileSync(output, kernelBodies, {flag: 'wx'});
process.stdout.write(JSON.stringify({event: 'semantic_direct_baseline_extracted', commit,
  sourceSha256: sha(source), output, extractedSha256: sha(Buffer.from(kernelBodies)),
  transformation: 'Removed only CUDA qualifier selection and host kernel-launch wrappers; kernel bodies unchanged.'}) + '\n');
