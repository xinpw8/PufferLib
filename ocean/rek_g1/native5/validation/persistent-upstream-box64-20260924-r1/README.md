# Pinned upstream Box64 runtime comparison

The candidate started, but private-arena capacity blocked both initial zero-action attempts. The controller stopped at 2026-09-24T14:18:00.286Z after exhausting its two-attempt budget. There are zero completed rounds and no combat stability result. At 14:22:32.929Z, a separate process receipt confirmed the same new-binary client alive after 382.53 s of startup/lobby exposure, no recorded fault or exit, and no remaining owned policy endpoints. `EXECUTION.json` preserves that initial cutoff.

A separate same-client retry accepted EnterSolo at 14:23:30.014Z, then timed out on the relay and on Stop/Release. All three child endpoints closed cleanly at 14:24:01.366Z, but native Stop/Release were not acknowledged. The controller exhausted its single-attempt budget at 14:24:06.412Z. Again, zero source observations, actions or completed rounds. `RETRY-EXECUTION.json` records this separately; its cause is unknown, and the earlier capacity message does not establish the retry's cause. No public fallback was used.

By 14:26:15.145Z the same client remained alive without a recorded fatal event, but Unity had stopped logging progress after loading Arena at 14:23:34.849Z. Over 10.013 s the main thread accrued zero CPU time and was observed in `pipe_read`; total process CPU time increased by 0.63 s. This supports a scene/main-loop servicing stall. It does not identify a deadlock, blocking object or causal component. The two closed controller stages, 93 files, were archived with NAS readback verification; the receipt is referenced in `RETRY-EXECUTION.json`.

The launcher changes one executable path, from the preserved Box64 build `2544543b3ace019bf2c9073a3afdb49b68d28bb1` to the separately built official revision `e4dc7b5e3c6cdd82272fff612c004da189c79d74`. Its binary SHA256 is `12a50a0f629f1ddeb08524c8f7399829e0b7101f79f4e20c094376c73a1af7ae`. All launcher flags, Wine 11.13, CoreCLR 6.0.7, rendering, bridge and private-session guards remain unchanged. BIGBLOCK and CALLRET retain their defaults; STRONGMEM=2 and WEAKBARRIER=0 remain explicit.

## Reason for the comparison

The older baseline eventually failed at invalid guest execution target 0x80, followed by a secondary CoreCLR exception. BIGBLOCK0 separately failed during startup after 24 s with INT29/c0000409, before any controller or rollout. Neither establishes the defective component. See `PRIOR-FAILURES.json` for pinned evidence and limits.

The new source is 384 commits ahead. Relevant upstream fixes cover [guest RIP/RSP before dispatcher signals](https://github.com/ptitSeb/box64/commit/9e13288ed247df5466abb785c94a6ecd9a67f89f), [CALLRET>1 JIT-block lifetime](https://github.com/ptitSeb/box64/commit/92527ded34d51cdfd9d9f785a7157df51e8c14d8), its [orphan-block](https://github.com/ptitSeb/box64/commit/6a8433a019af34b54e7eba79c9d0aa12e70bbc75) and [regression](https://github.com/ptitSeb/box64/commit/729c25da67f520bf2a5957d66f5015e34391b86b) fixes, and [signal-frame layout](https://github.com/ptitSeb/box64/commit/806d23db92cb0bcce9c83190a57607b6cb8ca88a). These provide a concrete test rationale, not a proven REK fix. The separate CALLRET0 failure is not explained by the CALLRET>1 lifetime defect. Changed cadence or regressions remain possible; outcomes cannot identify one commit among 384 changes.

## Frozen trial and reproducibility

Plan SHA256: `b0c18aedf660f0fe7bd45039938d8d35a590ca5aa9c7569e0230b004fb43dcc5`. Four planned full-round labels `upstream-s1801..1804`, frozen f147 policy and 9c55 driver, unchanged 773 controller/9a29 recorder/482 handoff helper. Maximum two attempts per label and zero automatic relaunches. Known private AI opponents are recorded separately; native 30 s redos remain auxiliary. No routine between-round process kill is introduced. No public-arena fallback was used.

Run `node --test launcher.test.cjs` here. Three CPU tests reconstruct the parent EventPipe launcher and candidate in memory from the previously published launcher plus these exact patches, verify both hashes, and verify unchanged flags/guards. They create no files and launch no game or controller. The private candidate also passed Bash syntax validation. These are source checks, not gameplay or stability validation.

`BUILD-RECEIPT.json` contains sanitized build/provenance facts. No executable, game source, raw observations, trace, register/stack dump or method map is published here.
