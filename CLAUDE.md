# CLAUDE.md

This file provides guidance to AI coding assistants when working with code in this repository.

## Build Commands

```bash
# Full build (C ocean envs + Torch extensions)
python setup.py build_ext --inplace

# Build only Torch extensions (pufferlib._C)
python setup.py build_torch --inplace

# Build only C extensions (ocean environments)
python setup.py build_c --inplace

# Build a single static-linked environment (clang compile + torch link)
python setup.py build_breakout
python setup.py build_pfr_native

# Build a single dynamic .so for an ocean env
python setup.py build_pong_so

# Debug build with symbols
DEBUG=1 python setup.py build_ext --inplace --force

# Skip ocean envs or training deps
NO_OCEAN=1 python setup.py build_ext --inplace
NO_TRAIN=1 python setup.py build_ext --inplace
```

Raylib and Box2D static libraries are auto-downloaded on first build.

## Training

```bash
# CLI
puffer train puffer_breakout
puffer train puffer_pfr_native --train.total-timesteps 100000000

# Python module
python -m pufferlib.pufferl train puffer_breakout

# Eval
puffer eval puffer_breakout --load-model-path experiments/puffer_breakout/*/model_*.pt

# Distributed (multi-GPU)
torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

# CLI overrides
puffer train puffer_pong --train.learning-rate 0.02 --vec.total-agents 2048
```

## Testing

```bash
pytest tests/ -x                              # all tests, stop on first failure
pytest tests/test_pfr_native.py -v            # single file
pytest tests/test_pfr_native.py::test_name -v # single test
```

## Architecture

### Training Pipeline

`pufferl.py` is the main entry point. It loads INI configs from `pufferlib/config/`, creates environments and policies, then runs a C++ training kernel:

```
load_config(env_name) → PuffeRL.__init__()
  → _C.create_pufferl(config, vec_config, env_config, policy_config)
  → loop: _C.rollouts() → _C.train() → log
```

`python_pufferl.py` is a pure-Python fallback (no compiled extensions needed).

### Ocean Environments

Native C environments live in `pufferlib/ocean/<env>/`. Each environment follows this pattern:

```
<env>.c          — Game logic: c_init(), c_reset(), c_step(), c_close(), c_render()
<env>_env.h      — Env struct, observation/action layout, reward logic
binding.h        — Defines OBS_SIZE, NUM_ATNS, ACT_SIZES; includes env_binding.h
binding.c        — Compiled as Python C extension (dynamic .so)
<env>.py         — Python wrapper extending PufferEnv; defines gymnasium spaces
```

The shared template `pufferlib/extensions/env_binding.h` provides `StaticVec` (vectorized env runner) and GPU memory allocation via `torch::from_blob` (zero-copy).

### C++/CUDA Extensions (`pufferlib/extensions/`)

- `bindings.cpp` — pybind11 module exposing `pufferlib._C`
- `pufferlib.cpp` — Core training structs: `RolloutBuf`, `TrainGraph`, environment creation
- `models.cpp` — C++ encoder/decoder implementations (CNN, linear, per-env custom encoders)
- `ocean.cpp` — Environment-specific encoder architectures
- `env_binding.c` — Generic C env compilation target (included by each env's binding.h)
- `cuda/` — CUDA kernels (squared_torch, modules)

### Config System

INI files in `pufferlib/config/`. Sections: `[base]` (package, env_name, policy_name), `[vec]` (num_envs, threads, buffers), `[env]` (env-specific params), `[policy]` (hidden_size, layers), `[train]` (lr, gamma, clip_coef, etc.). Environment configs override `default.ini`.

### Key Modules

| Module | Purpose |
|--------|---------|
| `pufferl.py` | Training orchestrator, CLI entry point |
| `pufferlib.py` | `PufferEnv` base class, wrappers, buffer management |
| `vector.py` | Vectorized environment handling (`Serial`) |
| `models.py` | PyTorch networks: `DefaultEncoder`, `DefaultDecoder`, `MinGRULayer` |
| `pytorch.py` | `nativize_dtype/tensor`, `sample_logits`, `layer_init` |
| `emulation.py` | Gym/Gymnasium/PettingZoo ↔ PufferLib bridge |
| `policy_pool.py` | Self-play policy versioning |
| `sweep.py` | Hyperparameter sweeps |

### Data Flow

Observations, actions, rewards, and terminals are allocated as GPU tensors in C++ and exposed to Python as zero-copy torch views. The GIL is released during C++ rollout and training kernels.

### pfr_native (Pokemon FireRed)

Special build: requires `~/pokefirered-native/` source tree. The build compiles the engine (`pfr_native.c`) and data (`pfr_native_data.c`) separately, renames symbols (`c_init` → `pfr_engine_init` etc. via objcopy), and links into a static library. The env header `pfr_native_env.h` wraps the engine calls.

**CRITICAL BUILD NOTE — read this before touching any pfr_native build:**

`setup.py build_pfr_native` does NOT compile the game engine. It only compiles `env_binding.c` (the PufferLib env wrapper) and links it with pre-built engine objects. The engine objects must already exist:

```
pufferlib/extensions/pfr_native_renamed.o   — game engine (pfr_native.c compiled, symbols renamed)
pufferlib/extensions/pfr_native_data.o      — game data tables (pfr_native_data.c compiled)
```

These are built from `~/pokefirered-native/` by its own Makefile/build script. If they are missing, the build will succeed but `pfr_engine_init/reset/step` will be **undefined symbols** in `_C.so`. Training will run but produce zero rewards and garbage observations — the env is a no-op.

After `build_pfr_native` creates `libstatic_pfr_native.a`, verify the engine objects are linked:

```bash
# These must show 'T' (defined text), NOT 'U' (undefined):
nm pufferlib/_C*.so | grep 'pfr_engine_init'
# Good:  00000000000cb510 T pfr_engine_init
# Bad:                    U pfr_engine_init   ← engine not linked, training will fail silently
```

If the engine objects get deleted (e.g. by `rm -rf` during clean builds), restore them:
```bash
cp ~/pokefirered-native/build/pfr_native/pfr_native_renamed.o pufferlib/extensions/
cp ~/pokefirered-native/build/pfr_native/pfr_native_data.o pufferlib/extensions/
```

Then manually add them to the archive before relinking:
```bash
ar rcs pufferlib/extensions/libstatic_pfr_native.a \
  pufferlib/extensions/libstatic_pfr_native.o \
  pufferlib/extensions/pfr_native_renamed.o \
  pufferlib/extensions/pfr_native_data.o
python setup.py build_pfr_native --force
cp build/lib.linux-aarch64-cpython-312/pufferlib/_C*.so pufferlib/
```

**Architecture (C++ ↔ Python policy matching):**

The C++ encoder (`PfrNativeEncoder` in `ocean.cpp`) and the Python policy (`PfrNativePolicy` in `ocean/torch.py`) MUST produce identical weight names and shapes. The Python policy creates initial weights; the C++ encoder runs during training. If they mismatch (e.g. using `policy_name = Policy` which creates LSTM weights vs C++ MinGRU), vf_loss will explode immediately and clipfrac will spike to ~0.8.

Working config: `policy_name = PfrNativePolicy`, `hidden_size = 128`, `num_layers = 4`, `tile_embed_dim = 4`.

### Live Dashboard

`dashboard/server.py` serves a live training dashboard on port 53580. During training, `pufferl.py` writes `/tmp/pfr_dashboard/stats.json` (stats + losses + coverage %) and `/tmp/pfr_dashboard/heatmap.png` (exploration heatmap overlay).

```bash
python dashboard/server.py 53580 &   # start dashboard
# then visit http://<host>:53580/
```
