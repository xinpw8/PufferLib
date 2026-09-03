# rek_fight

Provisional PufferLib T800 fight environment. Two T800s share the hash-bound
MuJoCo plant recovered from the Steam client. Each policy currently outputs the
measured keyboard encoding `MultiDiscrete{3,3,3,7}`: forward, strafe, yaw, and
one of six move slots.

This is the stripped state-based training sim. Its framework layer applies the
measured standing pose, serialized Bot 1 walk/strafe/yaw speeds, move timing,
and recovery duration directly at 50 Hz. MuJoCo supplies the recovered robot
geometry and contact queries. It is not the Unity client and it does not
contain the missing T800 ONNX or trajectory payloads.

The observation is 173 floats. It contains ego-first `qpos + qvel` for both
fighters, then ego-first discrete controller state for both fighters: recovery,
move, cooldown and fallen flags; current move and all remaining counters;
scored-impact bits; hit count; the router's last move and held velocity; and a
seven-entry move-request availability mask. Episode tick, MuJoCo time, and
remaining configured ticks complete the state. Terminal wrist and foot body
geometries are included in their parent limb's contact attribution.

## Evidence boundary

This environment has not passed the held-out trajectory/event variance gate.
Root locomotion and distal-limb motion are reduced-order reconstructions.
Contact to hit conversion, knockdown causes, recovery poses, rewards,
terminal/reset behavior, and spawn state remain provisional. The opponent is
a deterministic sparring controller, not
the recovered Private Bot 1 controller. The checked-in contract pins Bot 1
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

Human eval of the same action interface is WASD/QE plus move categories 1-6.
It also exposes the user's observed U straight-kick and I right-side-kick
aliases. The serialized `T800_REKKeys` debug map differs: U is
`right_light_attack`, I is `switch_to_stance_idle`, and its kick bindings are
Space+Y and Space+U. That conflict remains explicit until an input-to-executed-
move capture resolves the active runtime map. Joint clips for the six policy
categories are absent from the Steam build; the current executor uses measured
timing and a distal-limb reach. Fit visual bone windows into joint playback
before claiming trajectory parity.

On Spark, launch the human-controlled environment with:

```bash
bash ocean/rek_fight/run_human_eval_spark.sh
```

The server binds to `127.0.0.1:18766`. Reach it through an SSH local forward;
it does not expose a network listener beyond Spark loopback. The browser sends
WASD/QE, U/I, and 1-6 into agent 0 of the same `RekFight` C step function used
by the PufferLib binding. Agent 1 is a deterministic approach-and-move sparring
dummy. The viewer
copies the resulting MuJoCo state into a render-only model and never advances a
second simulation.
