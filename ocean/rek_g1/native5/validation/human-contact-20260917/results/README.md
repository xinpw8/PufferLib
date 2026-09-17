# Numeric experiment export

All results describe the reconstructed AI simulator. Authentic REK parity is false. No authentic match with the new checkpoint is included.

- heldout-before/after.matches.jsonl: all 512 match records per checkpoint, paired initial fixtures, 120-second rounds.
- geometry-*.matches.jsonl: all 16 matches per geometry setting, same initial fixtures. Policy actions and subsequent states may diverge; these are not controlled action replays or contact false-positive rates.
- *.events.jsonl: all seven native summary/identity events for each evaluation.
- training-config.json: named numeric fields selected from the actual logged INI, plus verified scoring/geometry identity.
- training-metrics.jsonl: all 64 native logged samples, retaining all 17 metric series and duplicate final sample. These are logged aggregates, not raw per-minibatch events. Fractional averaged agent_steps values are retained. Logged final SPS differs from whole-run transitions divided by uptime. Reported VRAM utilization describes the observed device, not isolated allocation by this task.
- training-summary.json, training-rounds.json, checkpoint-identities.jsonl: timing, completed-round accounting and all 17 numeric checkpoint steps with SHA-256 identity. No checkpoint binaries are included.
- validation-tests.json: host/CUDA geometry tests, finite-value asset test, mode configuration test and passing Compute Sanitizer result. Static geometry tests do not establish dynamics parity.
- summary.json: recomputed match totals cross-checked against native summary events.
- manifest.json: source/output byte counts and SHA-256 digests; source paths are fixed relative labels only.

The legacy opponent field scripted is a native bucket label. opponent_controller=recovered_bot1_v1 identifies the actual reconstructed controller. All output JSON records explicitly set authentic_parity=false.

## Export boundary

Run node ../export-results.cjs INPUT_RESULTS_DIR NEW_OUTPUT_DIR from this folder. The destination must not exist. Source JSON records require exact schemas, including fixed enum strings and numeric-array dimensions. Unknown or missing fields fail before output creation. The INI exports only fields named in configFields and metricNames in the script; all paths, unselected strings, and sweep definitions are excluded. Mixed test logs select named JSON events. Training stderr contributes only four validated scoring/geometry identity fields. Timing selects only elapsed time. Checksum lists contribute only 64-digit lowercase hashes and numeric checkpoint steps. No raw logs, commands, credentials, human capture data, game assets or model binaries are copied.

Source/output hashes support byte-level reproduction of this export; they do not independently authenticate the original experiment or establish simulator-to-REK transfer.

Recorded initial-coordinate qualification: the held-out before/after runs share seeds and all 512 match keys, but six initial_xy records differ, with maximum absolute coordinate difference 0.001600027 m. The exporter does not claim identical captured starting coordinates for these runs. All four geometry variants have identical initial_xy records. The reason for the six differences is not established by the exported data.
