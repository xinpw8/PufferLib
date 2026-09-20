# Owned-yaw development comparison, r24-r29

The predeclared six-round comparison completed with control 2 wins / 1 loss and
owned-yaw v2 treatment 0 wins / 3 losses. Every attempt was completed and retained.
The treatment field reached live inference correctly, but these results do not
support promoting this treatment checkpoint. Three rounds per arm do not
establish a population win rate or the general value of the representation.

All rounds are development evidence. None belongs to the separate frozen
20-round acceptance cohort. No outcome was replaced, retried, or excluded.

## Frozen setup

Code and interface reports were committed and pushed as `65fc8393` before r24.
The two actors are the matched one-epoch authentic GAE updates of the same
r21-r23 data, following exact migrated-initialization replay. See
[owned-yaw contract and validation](../../OWNED_YAW_OBSERVATION.md).
The treatment adds owned pending yaw at column 187. Physics, legality, busy
duration, reward definitions, and all other observation columns are unchanged.

| Arm | Frozen checkpoint SHA-256 | Observation schema |
| --- | --- | --- |
| Control | `f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4` | `rek.native5.scaled_polar_xy.v1` |
| Treatment | `c59a9f9044c8975a7cd790c5d8b44b253e6394fc2de93f2727b7d99879c4981e` | `rek.native5.scaled_polar_xy.owned_yaw_v2` |

Both used the tested native BF16 worker, seed 73, sampled categorical selection,
223 unmasked observations, and private Bot1/difficulty 0. Each round used a fresh
owned native Windows process on `WinSta0\RekPolicyEval`, with Spark inference.
Worker readiness, every encoded request, and the saved configuration agreed
on the selected checkpoint/schema. No user-desktop input was emitted.

## Results

All scores below are local : opponent. The local fighter was slot 0 in all six
rounds. Ordinary means non-five-point received awards, not causally attributed
strikes. A five-point value alone is not treated as proof of countout cause.

| Trial | Arm | Outcome | Points | Ordinary awards | Five-point awards | Attack requests | Max applied-action gap (s) |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| r24 | Control v1 | win | 14:13 | 9:8 | 5:5 | 107 | 0.054721 |
| r25 | Treatment v2 | loss | 4:25 | 4:15 | 0:10 | 89 | 0.089050 |
| r26 | Control v1 | loss | 9:24 | 9:9 | 0:15 | 100 | 0.067178 |
| r27 | Treatment v2 | loss | 3:15 | 3:15 | 0:0 | 114 | 0.069030 |
| r28 | Control v1 | win | 15:10 | 5:10 | 10:0 | 112 | 0.067727 |
| r29 | Treatment v2 | loss | 6:13 | 1:8 | 5:5 | 94 | 0.103771 |

| Arm | Complete / attempted | W / L / D | Points | Ordinary awards | Five-point awards |
| --- | --- | --- | --- | --- | --- |
| Control | 3/3 | 2/1/0 | 38:47 | 23:27 | 15:20 |
| Treatment | 3/3 | 0/3/0 | 13:53 | 8:38 | 5:15 |

The treatment's ordinary awards were also poorer in this small cohort. This
does not isolate a causal reason: stochastic matches, the learned actor update,
and unresolved native dynamics remain relevant. No attack request is labeled
as a hit, miss, executed move, or server-accepted action by this report.

## Validation and measured feature delivery

All six existing contact analyses report completed policy rounds, complete
native captures, consistent terminal evidence/points, and exact pose/clock
capture binding. The referee validator matched all 34,509 available policy
sources to genuine raw packets, covering 7,189 unique receipts, with no
unavailable source reasons. Hashes, decoded offsets, receipt clocks, round
context, and the existing 0.5 s freshness bound were unchanged.
Repeated latched referee calls were deduplicated by the existing validator.

All 34,026 policy decisions are preserved. Each round retains one terminal-race
`policy_stream_not_owned` rejection; 34,020 actions were locally applied.
No failed orchestration, incomplete round, missing capture, or threshold
relaxation occurred. Local application still does not prove server execution.

| Trial | Worker input rows | Nonzero column 187 | Invalid values / nonzero outside busy |
| --- | ---: | ---: | --- |
| r24 | 5758 | 0 | 0 / 0 |
| r25 | 5717 | 2300 | 0 / 0 |
| r26 | 5705 | 0 | 0 / 0 |
| r27 | 5743 | 2745 | 0 / 0 |
| r28 | 5763 | 0 | 0 / 0 |
| r29 | 5340 | 1862 | 0 / 0 |

Legacy column 187 remained zero for all 17,226 control inputs. V2 delivered
nonzero pending yaw in 6,907 of 16,800 treatment inputs, always during the
existing declared busy projection. This verifies interface delivery only.

All owned clients closed. A post-batch read observed zero REK processes.
The final client closed at `2026-09-20T03:32:51.1661506Z`; the GPU was released.
The paired recorder remained `Enabled:false` throughout. No further trial
or optimizer was started by this evaluation task.

## Reproducibility

Private trial root: `C:\rekagent\work\consistent-fighter-20260919-r1`.
Each `live-round_outcome_v1-rN` contains `result.json`, the immutable trial
streams, `contact-analysis/`, and `referee-validation/`. The latter records
the exact complete native filename, PID, content hash, and packet binding.

| Trial | Owned PID | Result SHA-256 | Native capture SHA-256 |
| --- | ---: | --- | --- |
| r24 | 101472 | `bf8f73e4403016f1a23c4f29a1e037a7f88bb1d6e2b0aad0dcf97a62d41586c1` | `fea73c3b69580e6dd19cca54a6acd47998ed4e11a911cd07623c1335e32d53bb` |
| r25 | 230896 | `62da76204605629d9393b5ef8600361519281fefa3f5931c09c3161eb430ed72` | `062fbb29170cfe76815711ad4f1563a931233904119ff6d45d67059594e4665f` |
| r26 | 398664 | `9ed7199e0c9990e1c6728ba6f5397f2aed007ed2525ba5a368a95d93e952a4fa` | `f56a4e98c679ba0ad1dea21680c97104c69c20cc9623fadb76b13a1e2026370d` |
| r27 | 395172 | `9863205280503738f2c0fb042927cce924b9bec6777665519553162cc104c14c` | `0e13b41d6b2339c407acc7ff2212250a3f9f05b04df861866538a7d2e841139f` |
| r28 | 84704 | `26a12b5b088293dc5efbecdd64517e943574fb76e60862f973fd04524116c983` | `c7091802328d5651e792ef34dfb2a5fe1e0ceac63af9033c190fad82f77a49df` |
| r29 | 113868 | `b42c8453b6078af9eda6e0d3cecff4ed896e5d447c3f63d4b66919ad4af08561` | `a10a904c2ccc207bd8c1dc9b546e88248fd3a8b25ddf2a21a241f00a384fe8d6` |

Configuration file hashes:

- `authentic-owned-yaw-control-v1.json`: `6c9922eed1c8fa9961a634f3481b880ebc9bc0e354393a0bab1fd08744f2d2e6`.
- `authentic-owned-yaw-treatment-v2.json`: `ec1a3db1fdd536afde64d76ff2fe07fa98a63f690dbb46b0eee1fcf218d2a740`.
- `run-native-trial-owned-yaw.ps1`: `469ec2192bf22266b73e74b8f3285e0030d0e01a9416117879c25e167972e5f2`.

The private file-only `owned-yaw-observation/summarize-cohort.cjs` reuses the
checked-in `aggregate_authentic_trials.cjs` completion classifications and
counts delivered observation values without connecting to the game.
Its final output `owned-yaw-observation/cohort-final-r24-r29.json` has SHA-256
`ae0ac5ef31cde8dfe5e1611a9fd9fed717cec266177508725d69ec9123b04109`.

The pre-evaluation NAS archive `owned-yaw-learning-r1` contains 966 files and
120,486,011 bytes, with manifest SHA-256
`e73714128a7ce70865d8572173a677d292633927ac40eafa54961fa4116059e3`.
It preserves the prior learning/data/build/performance setup and predeclared
plan/configuration/helper. It predates these six trials and is not claimed
to contain their new captures. Raw account records, proprietary payloads,
and checkpoint binaries are excluded from this public report.

## Finalized trial archive

All six finalized trial directories and their exact native PID captures were
copied to the new NAS directory:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\windows-development-r24-r29`

The archive contains 257 copied source files totaling 2,367,225,788 bytes,
plus its generated manifest and transcript. It includes the private
`cohort-report/summarize-cohort.cjs`, `cohort-report/cohort-final-r24-r29.json`,
and `aggregate-r11/{summary.json,README.md}`. Its current-development cohort
preserves 28 complete rounds out of 29 attempts, including the earlier r6 partial
attempt. Two prior development rounds remain separate; zero rounds are assigned
to the future frozen acceptance cohort.

Manifest SHA-256:
`16fb9393ad588ad464ec1b928f946caa32261f1cf0fdebbdc2ef66e2a52ceb48`.

Every copied file passed source-before, NAS readback, and source-after SHA-256
comparison, with unchanged source length and modification time. Each native
capture was bound to the exact closed owned PID/lifetime and existing referee
proof. Persisted manifest contents and all six trial identities were read back
and checked. Existing archive directories and original source files were
preserved. Archiving performed no client, input, or GPU operations.
