# Native hit receipt evidence

This export augments the separate visual-relay transition audit with packet evidence from all ten of its prespecified policy captures. The original audit and its fitted results remain the baseline. The exporter does not fit a model or change simulation behavior.

## Measured coverage

The exact native captures contain 151 received hit-effect packets and 169 score packets, including 18 five-point awards. Every packet has both a preceding native paired root pose and a preceding paired 30-bone relay pose. Maximum pose age is 0.0362442 s for native roots and 0.0365849 s for relay bones. Binding uses matching Unity frames, QPC, Unity time, and both actors' root coordinates over at least 119.56 s per capture. All native source hashes, packet body hashes, packed lengths, and decoded values are checked.

Received hit fields include contact-effect position, surface normal, relative speed, and `is_kick` (27 true, 124 false). Relative speed ranges from 1.7571826 to 6.600131 in uncalibrated Unity numeric units. All 151 hit receipts have a unique same-frame score receipt association. This association has no shared causal event ID and remains explicitly noncausal.

The original transition audit read `relay.stdout.jsonl`, whose visual policy fields do not contain these native packets. `RekEvidenceRecorder` writes them separately under the Wine runtime evidence directory. Therefore the original finding that relay contact data was unavailable does not imply the native recording lacks contact-effect data.

Attacker, defender, executed move, active clip, causal request, physical contact time, and server clock remain unknown. Latest requested action is never relabeled as executed action. All 59,988 native input samples have `action_playing=false`, null clip, negative clip frame, and zero clip FPS. Received effect geometry plus rendered poses does not establish authoritative server-time collision geometry.

The separate [referee audit](../native-referee/README.md) now resolves the five-point awards using explicit referee calls. This receipt exporter intentionally retains its original unassigned-cause score schema.

## Reproduction and storage

Run `node native_hit_receipt_data.cjs TRIAL_BASE NATIVE_DIRECTORY BASELINE_BALANCE_AUDIT NEW_PRIVATE_OUTPUT`. The exporter requires all ten prespecified A/B captures and preserves the complete six-session split manifest. The ten policy captures occupy four of those sessions; passive captures have no packet export here. No random frame split is introduced.

Final private output: `/home/spark-advantage/rek-training/native-hit-receipt-20260919-r2`. Raw hit/score teacher JSONL files remain on Spark, with file mode 0600 and directory mode 0700. Only the aggregate audit and schema are checked in. The aggregate report SHA-256 is `894c7f66fe04a32fe98cd47ef937f9691d0a614ae26575774f3acf6e250ac96e`.

Nine tests cover byte/hash agreement, source mutation, exact capture selection, clock/pose binding, future-pose exclusion, and ambiguous associations. They pass on Windows Node 25.2.1 and Spark Node 18.19.1. No RL training or simulator changes are part of this export.
