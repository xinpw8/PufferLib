# Physical observable-balance adapter and binding smoke

The physical runtime now opts into `rek.native5.observable_balance.v1` through
`REK_OBSERVATION_SCHEMA`. Unset or `rek.native5.scaled_polar_xy.v1` retains the
legacy path. The runtime API ABI and raw inspection buffers are unchanged.
Old checkpoints are observation-incompatible with this opt-in schema.

## Production integration

`runtime.cu` verifies both model roots are world-parented free joints at their
expected qpos addresses. It supplies those root-origin positions and WXYZ
quaternions, not inertial COM coordinates. Native awarded points, round duration,
remaining time, active/terminal state and `count_active[]` feed the shared
`rek_observable_balance::project` function. No pose threshold guesses a fall.

There is one independent history/cache per arena. Each successful exported
50 Hz observation advances an integer tick clock; its timestamp is `ticks*0.02`
seconds. Both fighter perspectives use the preceding observation snapshot.
Explicit runtime reset clears history. Native `episode_reset` advances the
round key, because the combat module reinitializes its round number between
training episodes. Physical contact samples and same-round counted body resets
never touch this history. Repeated calls to encode either perspective only read
the cached projection. The learner receives the same cached actor-0 row.

Raw internal observations remain available to the existing
`CandidateApproachDummy` and diagnostic snapshot API. They are not the opt-in
learner input. This adapter does not change the opponent, rewards, controller,
physics or action masks. The model/client hinge correspondence remains unproven:
both joint-pose availability bits are zero, with joint positions/rates explicitly
unavailable. Raw hinge qpos is never substituted for projected bone angles.

## Adapter verification

Fresh private stage:
`/home/spark-advantage/rek-training/physical-observable-balance-20260921-r1`.

The final adapter build/run are `build-r2` and `run-r2`. Commands:

```bash
stage=/home/spark-advantage/rek-training/physical-observable-balance-20260921-r1
bash "$stage/source/validation-quality/build_physical_observable_balance_probe.sh" \
  "$stage/build-r2" "$stage/source"
bash "$stage/build-r2/run_physical_observable_balance_probe.sh" \
  "$stage/build-r2" "$stage/run-r2"
```

Both legacy-unset and observable modes exited 0. Each used four arenas, 120
actual CUDA+SONIC ticks and deliberately short **1 s diagnostic rounds**. These
are adapter tests, not competitive rounds or training-performance benchmarks.

- Native CPU adapter test: 700 checks. Covers free-root inputs, both
  perspectives, point/count ordering, explicitly missing joints, terminal phase,
  same-round body-reset-like displacement, new-round/explicit reset history and
  rejection of regressing same-round point counters.
- CUDA adapter fixture: 10,704 feature comparisons over three captured graph
  replays, maximum CPU/GPU discrepancy zero. These synthetic adapter fixtures
  are not physical fall evidence.
- Physical integration per mode: 968 fighter-perspective observations, eight
  new-round arena transitions and eight terminal arena transitions. New-schema
  CPU-reference discrepancy zero. Repeated encoding is byte-stable; learner and
  actor-0 encoded rows are byte-identical; excluded privileged columns remain
  zero. Explicit API reset clears history. Legacy projection and raw root
  inspection checks pass.
- CPU physics entry points are linked to abort wrappers. No CPU physics calls,
  PPO updates, fall/strength claims or performance claims are made by the probe.

| Artifact | SHA256 |
|---|---|
| `runtime.cu` | `2e2112f19f749893d3a8fb44f5fb5c8b04928eb7e130a228473e8f5d93b5df59` |
| `observable_balance.h` | `4b651a2f02b7a335cb781bb886b048d83025acc88dca4a522f62bc4cea7a0d3a` |
| `build-r2/runtime.o` | `5ee1d0732ad1268e7fca9209526131e0e8564c09b0f2d977f8c7d7956dbb26ae` |
| `build-r2/physical-observable-balance-probe` | `724cdd6523393b670c630b9c048d2a8c84922575aa4f8b5c637734466189e453` |
| `physical_observable_balance_probe.cu` | `c597657379defcf274c4cdca8a0cac0cee8c10b2cffe0bc2559735b4ec1c9b37` |

## End-to-end native PPO binding

The wrapper source came from the separate coordinated
`observable-balance-binding-20260921-r1/source/ocean/rek_g1/native5` stage.
Its `puffer_env.cu` SHA is
`d3ae2b39d9627f18e027442eb60bc8ec617c0fa2b46a30d2b13e553772d7d627`.
It requires explicit `REK_PHYSICS_BACKEND=mujoco_cuda` and rejects an
observation-schema-unverified frozen opponent. The build manifest includes the
observable header. This resolves the prior wrapper rejection of the new schema.

The fresh trainer links the tested physical runtime object, corrected
measurement object and existing validated physical module set. Compact autoreset
is disabled. Dedicated link-only test instrumentation checks the actual Puffer
learner observation buffer against the runtime's projected actor-0 export at
reporting boundaries, along with finiteness, excluded zeros and unavailable
joints. It does not change trainer math or runtime stepping. CPU physics abort
wrappers also remain linked.

Each smoke used fresh random weights, four arenas, horizon/minibatch 64/256,
256 learner transitions, one PPO epoch, seed 73, environment seed 419,
learning rate 0.0001, entropy coefficient 0.001, gamma
0.9998844821426083, lambda 0.9978673240629938, normalized points/falls reward and
1 s diagnostic rounds. No old-schema checkpoint was loaded. A 30 s process
timeout bounded each GPU run. Authentic inference could run concurrently, so
timing is not a performance result.

`ppo-smoke-r1` trainer exited 0, but its receipt wrapper exited 1 because fresh
initialization does not save a step-zero checkpoint in the pinned trainer.
That successful training log and postprocessing issue are preserved. Its final
checkpoint SHA is
`5c52003d555fda883d3458c68b48ca0ba4b9c5f8e020b7499d2997a4b8d1a8f3`.

With explicit authorization, `trainer-build-r2` uses an isolated generated
trainer copy with only a test-only save immediately after `create_pufferl`.
Production trainer sources and previous binaries are untouched. Its
`ppo-smoke-r2` trainer and receipt wrapper both exited 0. Commands are preserved
in the stage's `build-binding-smoke.sh`, `build-binding-smoke-r2.sh`,
`run-binding-smoke.sh`, guarded link/compile receipts and each run's `command.txt`:

```bash
bash "$stage/build-binding-smoke-r2.sh"
bash "$stage/run-binding-smoke.sh" ppo-smoke-r2 trainer-build-r2
```

Exact fresh step-zero checkpoint:
`d8dc477e1eb33380f7d2cb260b704f88c4f2498cd9d6ccf832d44542b462593f`.

Updated step-256 checkpoint:
`5dd2151f3c80a57ee649e0dc2dfa94861b0f5c99f3204fc93578993902cdf754`.

Both files are in
`ppo-smoke-r2/checkpoints/rek_native5/ppo-smoke-r2/`; byte comparison confirms
the update changed weights. Each run passed two learner-buffer checks and
completed four diagnostic ties, with points 0:0, confirmed falls 0:0, zero
failure bits and zero reward saturations. This establishes native observation
and PPO wiring. It establishes no learned fighting strength, physical fall
exposure in this short run, live encoder parity or deployable policy.

## Prepared larger run, not launched by this implementation task

The clean, uninstrumented trainer is
`trainer-build-r1/puffer-rek-native5`, SHA256
`bd696ecc152c8b326bd6891bdf60ee97f7b0fa523f39e98696fe21708165185d`.
It has neither the learner-buffer diagnostic guard nor the test-only initial
save. Its wrapper/runtime are the same production objects used by the guarded
first smoke. The separate test-only second trainer is not the production target.

Private `run-balance-physical-training.sh` SHA256
`c8f63ca60e8c08fa0ea8c8d9255b88110ad60d73a4d39bd7017fdf6fc8430474`
prepares a fresh `train-balance-physical-r1` directory when invoked. It pins the
clean binary hash and selects the new observation schema, `mujoco_cuda`,
normalized points/falls reward, no CPU evaluation, no frozen opponent and no
warm start. It uses 512 arenas, horizon 512, minibatch 8192, 4194304 transitions,
120 s rounds, learning rate 0.0001, entropy coefficient 0.001, the gamma/lambda
above and base/environment seed 419. The existing batch-1024 SONIC graphs serve
the 1024 fighters. The hard process timeout is 1200 s. Sixteen updates cover
163.84 simulated seconds per arena; any later outcome/performance claim requires
that run's actual completion, checkpoint and round/counter receipts.

## Private evidence archive

The completed adapter builds/runs, both PPO smoke builds/runs, copied binding
sources and prepared launch scripts are preserved in the private archive below.
The larger `train-balance-physical-r1` directory is excluded. No model assets,
checkpoints or proprietary binaries are added to the repository.

NAS location:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-observable-balance-r1\physical-observable-balance-20260921-r1-evidence.tar.gz`

Size: 46,021,645 bytes. SHA256, verified on the local and NAS copies:
`ff4fe966065ce0aa4000398f1c2ea92f7b312123935bd047c2c5fa0f370fcb93`.

The matching local receipt is under
`C:\rekagent\work\physical-observable-balance-20260921-r1`.
