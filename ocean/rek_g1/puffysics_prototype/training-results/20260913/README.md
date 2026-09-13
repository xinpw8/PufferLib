# Training evidence

`training-summary.json` contains the comparison and identifies each source report
by path and SHA-256. Kernel summaries derive from CUDA graph-node traces of
actual training. Raw `.nsys-rep` files remain private because Nsight captures
process environment metadata, which can contain credentials.

`raw-reports.tar.gz` preserves all original training reports, invocation records,
stdout, stderr, exit codes and kernel summaries. Extract it in this directory
to inspect individual files and verify all 136 public evidence artifacts:

```sh
tar -xzf raw-reports.tar.gz
sha256sum -c SHA256SUMS.txt
```

Archive SHA-256:
`2957348287617c5b1264ea0319a5f1a03f6a2d0faef882517710c9a664047300`.

No private model export, game binary, controller weights or policy checkpoint
is included. Reports identify the locally retained files by path and hash.

## Relaxed solver follow-up

`solver-diagnostics-summary-v2.json` reports standard stepping, capsule
substitution and independent-contact trials. It separates cold-start trials
from warmed training and preserves failed attempts without assigning valid SPS.

`solver-diagnostics.tar.gz` contains their raw reports, invocation records,
stdout, stderr, exit status, compiler output and collision reproducer results.
The included `puffysics-standard-cold-nsys-summary.json` and
`puffysics-independent-capsule-nsys-summary.json` summarize the actual training
traces without copying process environment metadata.

```sh
sha256sum -c SOLVER_SHA256SUMS.txt
tar -xzf solver-diagnostics.tar.gz
```

Diagnostic archive SHA-256:
`1b568f4e1a895f1c474f7c6155058c6b05bc920c5b439c279fb1021890c5ab30`.

The capsule model export itself stays private. `capsule-conversion.json`
identifies exactly which geometry IDs changed and the source hash.
