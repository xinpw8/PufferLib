# REK server-free balance curriculum

`rek_sandbox` is a headless, one-policy Puffer environment. Each environment
contains the hash-bound two-T800 MuJoCo diagnostic plant. Fighter 0 is the
learner. Fighter 1 follows a deterministic joint-target wave and is reset to
its measured keyframe whenever it falls. No REK process, UI, network service,
or synthetic desktop input participates in stepping.

## Evidence boundary

Measured inputs:

- two-T800 MJCF SHA-256
  `01caa6ed90277a90fc71b4c16d7959fb70f7702fa15bea7c68d3eaa9d5f27b2c`
- controller evidence SHA-256
  `5b262c83fa0db89804007ec176e4aefa72bb123090e1b81391b77176d78e28d7`
- root-controller search artifact SHA-256
  `31f74cf7cb3b416760880b9ca439ffbd39323cc2540c88fc5be9a927ce687761`
- 25 PdStand Kp/Kd values and 25 actuator force limits
- 50 Hz MuJoCo plant step

Derived from measurements:

- the two-fighter keyframe fitted from client-observed named-bone poses

Provisional curriculum choices:

- keyframe pose as the continuous target baseline
- residual action scale and clipping
- PdStand gains used continuously as implicit position actuators
- deterministic dummy wave and dummy teleport reset
- balance reward, action cost, fall thresholds, and episode length
- actuator-only root stabilizer and its configurable scale

The installed client contains no T800 ONNX, active policy JSON, or T800 motion
trajectory. This environment is suitable for controller training experiments.
It is not a behavioral clone, combat-parity environment, or acceptance evidence.

The provisional root stabilizer was selected by a deterministic Spark search.
Its source artifact SHA-256 is
`31f74cf7cb3b416760880b9ca439ffbd39323cc2540c88fc5be9a927ce687761`.
At full scale, combined with the continuous implicit PdStand servos, it held
both fighters above the probe fall gate for 300 s. It uses only the first 12 leg
actuators and can be reduced or disabled with `root_stabilizer_scale`. This is a
training scaffold, not a recovered REK controller.

Both evidence paths are mandatory at runtime:

```bash
export REK_MJCF_PATH=/absolute/path/to/t800_t800_factory_arena.diagnostic.xml
export REK_CONTROLLER_PATH=/absolute/path/to/controller_path.json
export REK_ROOT_CONTROLLER_PATH=/absolute/path/to/provisional-root-balance-search.json
./build.sh rek_sandbox
python -m pufferlib.pufferl train rek_sandbox
```

The loader requires MuJoCo 3.7.0 and rejects any of the three artifacts unless
its SHA-256 matches the values above. The reported `score` is the provisional
balance episode return.
