# rek_fight

PufferLib training environment for REK Private Bot 1. Two T800s share the
hash-bound MuJoCo plant recovered from the Steam client. Each policy outputs
the measured keyboard encoding `MultiDiscrete{3,3,3,7}`: forward, strafe, yaw,
and one of six move slots.

This is the stripped training sim. It steps the recovered 50 Hz plant with
PdStand implicit position actuators and a root-velocity tracker using the
serialized Bot 1 walk/strafe/yaw speeds. Hits use MuJoCo contacts inside the
serialized impact windows. It is not the Unity client and it does not contain
the missing T800 ONNX payloads.

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

./build.sh rek_fight --cpu
./build.sh rek_fight
python -m pufferlib.pufferl train rek_fight
```

Human eval of the same action interface is WASD/QE plus move categories 1-6,
matching `T800_REKKeys` on the real client. Joint clips for those six moves
are still absent from the Steam build; the current executor uses measured
timing and a distal-limb reach. Fit visual bone windows into joint playback
before claiming trajectory parity.
