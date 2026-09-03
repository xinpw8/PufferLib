# rek_fight

Provisional PufferLib T800 fight environment. Two T800s share the hash-bound
MuJoCo plant recovered from the Steam client. The repository also contains a
Python adapter for the walking and supine-to-stance policies published by
EngineAI in `engineai_robotics_native_sdk`. The upstream checkout, policy
weights, and recovery trajectory are commit- and SHA-256-pinned.

The current C vector environment still applies the measured standing pose,
serialized Bot 1 walk/strafe/yaw speeds, move timing, and recovery duration
directly at 50 Hz. It has not yet been converted to use the published MNN
controllers. Do not use the C vector environment for training until that
integration and the held-out parity gate are complete.

The human evaluator uses the published 22-joint T800 walking policy at 100 Hz
over 500 Hz MuJoCo physics. When a robot falls, it switches to EngineAI's
25-joint, 301-step supine-to-stance residual policy and reference trajectory.
The six REK attack trajectories are still unknown and are disabled in this
controller-validation evaluator. No synthetic reach animation is used.

The observation is 173 floats. It contains ego-first `qpos + qvel` for both
fighters, then ego-first discrete controller state for both fighters: recovery,
move, cooldown and fallen flags; current move and all remaining counters;
scored-impact bits; hit count; the router's last move and held velocity; and a
seven-entry move-request availability mask. Episode tick, MuJoCo time, and
remaining configured ticks complete the state. Terminal wrist and foot body
geometries are included in their parent limb's contact attribution.

## Evidence boundary

This environment has not passed the held-out trajectory/event variance gate.
The C environment's root locomotion and distal-limb motion remain reduced-order
reconstructions. Contact to hit conversion, knockdown causes, rewards,
terminal/reset behavior, and spawn state remain provisional. The opponent is
a deterministic sparring controller, not the recovered Private Bot 1
controller. The human evaluator's opponent uses the official walking policy
with a deterministic approach command. The checked-in contract pins Bot 1
tuning and parts of its control flow, but the complete transition table,
authoritative decision cadence, server RNG state, and assigned move reservoir
are not yet available as a simulator-ready artifact. Those unknowns are not
filled with defaults here.

## Required artifacts

```bash
export MUJOCO_HOME=/absolute/mujoco-3.7.0
export MUJOCO_LIB="$MUJOCO_HOME/lib/libmujoco.so.3.7.0"
export REK_MJCF_PATH=/absolute/path/to/t800_t800_factory_arena.diagnostic.xml
```

`REK_MJCF_PATH` must hash to
`01caa6ed90277a90fc71b4c16d7959fb70f7702fa15bea7c68d3eaa9d5f27b2c`.

Fetch the published EngineAI controller assets at the pinned revision:

```bash
bash ocean/rek_fight/fetch_engineai_t800.sh /absolute/cache/engineai_robotics_native_sdk
```

Pinned controller files:

- EngineAI SDK commit `335c60e88772c26c7852d0abd6b3c7439037dd8f`
- walking MNN SHA-256 `cbcb90f86dbb2fde39bdc5a25c8d0530d5c79c7a8f84b1f90863d8c9065b6427`
- supine-to-stance MNN SHA-256 `deb9974b1f4f4a7e77801f8c9c6e77f599caab0ca4dd7709fe0bae55870e0e86`
- supine-to-stance trajectory SHA-256 `c2f19c164093701311634024eb27999fed4631a00d38d507f8aa306ee138c161`

## Build and smoke

```bash
cc -std=c11 -O2 -I"$MUJOCO_HOME/include" \
    ocean/rek_fight/test_rek_fight.c -o test_rek_fight \
    "$MUJOCO_LIB" -Wl,-rpath,"$(dirname "$MUJOCO_LIB")" -lm
./test_rek_fight
./test_rek_fight --benchmark 20000

./build.sh rek_fight --cpu
./build.sh rek_fight --float
python ocean/rek_fight/vector_smoke.py --mode both
REK_FIGHT_TRAIN_STEPS=8192 ./ocean/rek_fight/train_spark.sh
```

The PyTorch trainer requires the `--float` build. The default 16-bit
observation build is rejected by `torch_pufferl`. `vector_smoke.py` loads the
extension from the current checkout, exercises both CPU actions and CUDA
action/observation buffers, verifies finite outputs and terminal values, and
reports vector, match, and agent steps per second as JSON.

`train_spark.sh` runs in the current Spark checkout and defaults to an 8,192
step execution smoke. A longer run is rejected while this environment remains
provisional unless the caller explicitly sets
`REK_FIGHT_ALLOW_PROVISIONAL_LONG_RUN=1`. That override permits experiments; it
does not satisfy or bypass the held-out REK parity gate.

Validate the public controllers independently of the browser:

```bash
python ocean/rek_fight/validate_engineai_policy.py \
  --mjcf "$ENGINEAI_SDK_ROOT/assets/resource/t800.xml" \
  --policy "$ENGINEAI_SDK_ROOT/assets/config/t800/rl_walking_example/policy/t800_260618_165257_30000.mnn"

python ocean/rek_fight/validate_engineai_recovery.py \
  --mjcf "$ENGINEAI_SDK_ROOT/assets/resource/t800.xml" \
  --policy "$ENGINEAI_SDK_ROOT/assets/config/t800/rl_supine_to_stance/policy/T800_supine_to_stance.mnn" \
  --trajectory "$ENGINEAI_SDK_ROOT/assets/config/t800/rl_supine_to_stance/trajectory/T800_supine_to_stance.npy"
```

Human eval accepts WASD/QE for walking. Attack keys are disabled because joint
trajectories for the six REK move profiles are absent from both the published
EngineAI controller repositories and the inspected Steam build. The serialized
`T800_REKKeys` debug map conflicts with the observed runtime keyboard map. An
input-to-executed-move capture must resolve that map before combat playback is
implemented or claimed.

On Spark, launch the human-controlled environment with:

```bash
bash ocean/rek_fight/run_human_eval_spark.sh
```

The server binds to `127.0.0.1:18766`. Reach it through an SSH local forward;
it does not expose a network listener beyond Spark loopback. The browser sends
WASD/QE to a Python simulation using EngineAI's MNN walking and get-up policies.
Agent 1 runs the same walking policy with a deterministic approach command.
Browser input is page-local; the server does not install a global input hook or
send input to Windows applications.
