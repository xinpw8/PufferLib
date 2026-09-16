# Frozen V4 contact acceptance audit

Native diagnostic executed on Spark at 2026-09-16 22:46:36 UTC, exit 0.
No production implementation or checkpoint was changed for this audit.
Private build/output: `/home/spark-advantage/rek-training/policy-quality-20260916-r1/contact-audit`.

The selected V4 checkpoint (`0d612521839298ffbe5783ed4fa449286a940b62a3a6768da7cc5ab248eb843b`)
played 64 independent, randomized-start, 20 s rounds against each of neutral
and scripted opponents. Seed 10001, policy side 0, sampled BF16 inference,
shaping disabled. It won all 128 compact rounds. GPU simulation and observation
took 0.323276 s and 0.116947 s respectively. This was evaluation, not training.

The diagnostic reconstructed every awarded compact contact from read-only
before/after state. Per-tick contact counts and terminal point totals matched
exactly, with zero failures. Source included in the diagnostic and preserved
V4 source both hash `c655b11f45c8cf44c38282ccbca3155b6b6aa1b988c2914594f4b190648dd0a6`.
The actual asset build fingerprint, strike source contract SHA, all 24 route
identities/configurations, and source clip FPS/frame counts matched the
recovered catalog before GPU startup. This verifies candidate compatibility;
it does not verify the authentic server's selected build.

| Policy-scored contact filter | Neutral | Scripted |
| --- | ---: | ---: |
| Original compact points | 3046 | 2249 |
| Outside recovered apex gate, post-step source cursor | 1774 (58.24%) | 837 (37.22%) |
| Outside gate at both pre/post source cursors | 1773 (58.21%) | 832 (36.99%) |
| Every intersected target's relative sphere speed below 1.75 m/s | 831 (27.28%) | 647 (28.77%) |
| 0.3 s cooldown alone rejects | 268 (8.80%) | 269 (11.96%) |
| Apex then per-invocation apex dedup rejects additionally | 100 (3.28%) | 22 (0.98%) |
| Sequential speed, apex, cooldown, dedup retains | 1103 | 1190 |
| Move 5 jab-uppercut share of original points | 90.41% | 86.57% |

Independent filter percentages overlap. Sequential filtering removes 63.79%
and 47.09% respectively, using the most permissive speed across intersected
targets. Detailed per-move output is in `contact-audit-20260916.jsonl`.

Important limits: source-cursor apex acceptance calls the existing native C
`rek_g1_strike_intent_apex`. Contact geometry remains compact enclosing spheres.
Relative velocities are finite differences of those sphere centers, explicitly
a kinematic proxy, not measured native rigid-body/contact-point velocities.
The stream contains only contacts already scored by V4. Gates are reapplied
to that fixed stream; other contacts could change cooldown history in a full
alternative runtime. No upright, impulse, collision blocking or native contact
fields are fabricated. This is not a predicted authentic score or a new win
rate. Contact timing within a swept interval is not reconstructed; the report
also gives both endpoint cursor checks.

Build: `/usr/local/cuda/bin/nvcc` 13.0.88, C++17, O3, sm_121; exact V4
`fast_assets.o`, `native_policy.o`, and `cJSON.o` reused. CPU acceptance code
uses gcc C11 O2 with contraction disabled. No Python or CPU physics steps.
Executable SHA: `58beae9270fe8be645e9ed216a71147f3391491b1dd9e0d446e9fdcfbaaa6888`.
Original stdout SHA: `4219c60edce4fc3940ba54cd4a6fe67806aa7c467ccda934a34b07da87d54f37`.
Original stderr SHA: `4c6a26569c387aef367e9f5fdc34a9e3195c3a05cf9c79ce34581846212b7fa0`.
