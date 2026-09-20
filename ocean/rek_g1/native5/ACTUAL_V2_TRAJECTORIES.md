# Exporting actual owned-yaw-v2 trajectories

The existing exporter defaults to strict legacy v1. Actual recorded v2 fights require this exact opt-in:

```text
node authentic_trajectory_data.cjs ROOT NEW_OUTPUT 0.9998844821426083 0.9978673240629938 live-SELECTED_ROUND --observation-schema=rek.native5.scaled_polar_xy.owned_yaw_v2
```

Use only completed rounds with finalized strict contact/referee analyses and a single behavior checkpoint. No inference, optimizer, client interaction or GPU is involved in export.

The shared exporter keeps existing acknowledgement, score, reward, elapsed-time and recurrent-history checks. V2 additionally verifies ready/action/encoder schema, full-hash equality of the same pre-action relay and encoder input, recorded worker/encoder arrays, source QPC, saved busy provenance and active owned desired category 1..15. It checks recorded column 187 against that evidence; it does not replace the column or derive intent from the newly sampled action. Inactive terminal-only inputs require explicit schema and zero column 187. The last active decision retains its yaw even when `terminal_after` is true or its actor weight is zero.

Output is `authentic-trajectories-owned-yaw-v2.bin`, magic `REKRL002`, version 2, with the existing 256-byte header and 1,128-byte rows. Its manifest declares actual recorded v2 origin and the actual behavior checkpoint. No migration is claimed. Native replay bound to this dataset/checkpoint is still required before the explicit-v2 authentic PPO loader may optimize it. Legacy export remains strict and rejects v2; unknown or duplicate CLI options reject.

CPU regression coverage includes 24 exporter/upgrader tests. Refactoring into `owned_yaw_export_evidence.cjs` preserved the full historical r21/r22/r23 upgrade: 17,558 rows, 7,793 nonzero column-187 values, SHA256 `a58502e1a36b1b7a730e572325cc9e58a7e55b33d1428221b4575dab12102d77`, identical to the pre-refactor binary. The fresh private verification is under `C:\rekagent\work\consistent-fighter-20260919-r1\actual-v2-export-cpu-r1\historical-regression`.
