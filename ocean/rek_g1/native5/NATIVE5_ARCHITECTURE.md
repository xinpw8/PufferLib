# Native PufferLib 5.0 REK integration

This directory connects the REK CUDA runtime to the standalone PufferLib 5.0
trainer. The pinned trainer commit is
`773f923d80e73bdc255a2ba730c918b28e416aa1`, published in
`https://github.com/xinpw8/PufferLib.git` on `codex/wr64-5.0`.
Every `src/` file and `config/default.ini` has the exact same Git blob and mode
as local commit `84a89728fafd4034ae3c386a9e0665d62310780b`, used for initial
integration. That local commit is unavailable from the public remote.
The build extracts committed
`src/` and `config/default.ini` with `git archive` and checks their Git object
hashes. It does not copy the checkout's modified files.
Each build extracts another private trainer copy and applies the checked-in
`pufferlib5_action_mask.patch`. Its guarded API extensions pass the GPU
action-mask buffer to environments defining `PUFFER_ENV_GPU_ACTION_MASK` and
check each completed GPU rollout when `PUFFER_ENV_GPU_ROLLOUT_CHECK` is defined.
The build manifest records the clean source, patched source, and patch hashes.
The clean source stage and the Wave Race checkout remain unchanged.
If the selected Git source lacks the published commit, the build fetches it
into a private temporary repository. It never fetches into the caller's
existing checkout.

The separate staging directory is
`/home/spark-advantage/rek-training/native5-rek-20260913-v1/pufferlib5`.
The older repository `src/pufferlib.cu`, its bindings, and its build path are
outside this integration.

## Runtime interface

`runtime_api.h` declares an opaque C interface implemented by `runtime.cu`.
Creation receives the source XML model, compiled physics export, semantic
asset bundle, motion feature bundle, and two controller ONNX files. It may
read assets and allocate memory. The reset and step calls enqueue work on
the trainer's bound CUDA stream and must support CUDA graph capture.

Each arena supplies one learner, fighter 0, with 223 float observations and
one categorical action in `[0, 32]`. The opponent is internal to the runtime.
One semantic decision covers 20 ms and ten 2 ms physics steps. Locomotion
segments currently last one decision tick. The default durations for the
17 semantic moves are:

```
35 27 31 45 32 45 157 145 158 139 134 138 73 75 68 71 103
```

Individual durations can be overridden with `env.move_duration_0` through
`env.move_duration_16`, expressed in decision ticks. Runtime creation owns
validation of asset relationships and model compatibility.

The wrapper allocates a device `Env` record per arena and supplies its log
address and stride to the runtime. The runtime writes real observations,
rewards, terminal flags, and completed-round log totals. PufferLib reduces
the flat float log on the GPU, copies the aggregate, and calls `puf_log` on
that host aggregate. The wrapper performs no device read in `puf_log`.
`log.n` counts completed rounds; logged totals are divided by that count.
Finite-state failures must fail the runtime even when no round has completed,
because the upstream reducer skips records with `n == 0`.
The wrapper checks sticky runtime status immediately after PufferLib
synchronizes a completed rollout, before any timing-related early return or
PPO update can consume it, and again during close. This includes the first
captured rollout and episodes that have not completed, without per-step host
reads. These boundary checks contribute to measured wall time.

The upstream GPU environment callback omits the action-mask buffer. This
integration supplies that pointer through the guarded patch and
`rek_native5_bind_action_mask`. The runtime copies each arena's fighter-0
scheduler mask into the learner mask buffer during binding and observation
publication. PufferLib then samples and evaluates PPO log probabilities using
the real held-action and move-validity mask. The runtime reports any rejected
choice through `actions_invalid`. The pinned GPU backend supports one policy
and one rollout buffer; its historical-opponent selfplay is unavailable.

## Build and validation

`build_native.sh` executes shell tools, Git, and the CUDA compiler. CUDA and
NCCL are linked directly. Existing NCCL native headers and libraries are
located in a Python environment's package directory; the build never starts
an interpreter or imports that package. Raylib comes from the existing ARM64
distribution. These locations can be overridden with `PUFFER5_RAYLIB`,
`PUFFER5_NCCL`, and `REK_NATIVE5_CUDA`.

The default target is Spark's `sm_121`. `REK_CUDA_ARCH` overrides it. A fresh
output directory is required so existing executables and evidence survive.

Compile the trainer and environment wrapper without launching GPU work:

```sh
bash ocean/rek_g1/native5/build_native.sh /absolute/new/build-directory --compile-trainer-only
```

For a new host, explicitly prepare the published source checkout and pass its
path. The dependency directories described above must also be available:

```sh
git clone --filter=blob:none --single-branch --branch codex/wr64-5.0 \
    https://github.com/xinpw8/PufferLib.git /absolute/new/pufferlib5-source
PUFFER5_GIT_SOURCE=/absolute/new/pufferlib5-source \
    bash ocean/rek_g1/native5/build_native.sh /absolute/new/build-directory --build-runtime
```

The script always selects the exact published commit, regardless of the
checkout's current branch tip.

This produces an object with unresolved runtime entry points. It checks the
actual 5.0 source compatibility. It does not establish runtime correctness,
produce a runnable trainer, or demonstrate training.

Build the implemented runtime, controller, physics, motion scheduler, and
combat modules together with the trainer:

```sh
bash ocean/rek_g1/native5/build_native.sh /absolute/new/build-directory --build-runtime
```

This compiles the original recovered C state machines as CUDA device objects
using their existing wrappers. Model loading links native MuJoCo 3.7.0 for
startup mapping and kinematics; simulation steps use Puffysics on CUDA.
`REK_NATIVE5_MUJOCO` overrides the native MuJoCo distribution directory.
The motion, controller, measurement, runtime, and combat modules disable
floating-point contraction and fast approximations. The physics translation
unit preserves its separately verified upstream default FMA compilation.

Alternatively, after building those modules separately, provide their exact
object paths to link the native executable:

```sh
bash ocean/rek_g1/native5/build_native.sh /absolute/new/build-directory \
    /absolute/runtime.o /absolute/controller.o /absolute/physics.o
```

The explicit object list keeps module ownership with the runtime build.
Any cross-translation-unit device calls require compatible relocatable
device-code objects. The build writes source hashes and ELF dependencies.
Missing runtime symbols cause linking to fail; there is no synthetic runtime.

The output contains `config/default.ini` and `config/rek_native5.ini`. Run
the executable from its build directory. The six required asset paths
default to `None` and must be supplied through `--env.KEY=PATH` arguments.
The small initial configuration requests eight arenas and 1024 training
steps. Evaluation is disabled by default; explicit evaluation requires
`eval --headless` and a positive `base.eval_episodes`.

The pinned trainer supports graph capture for whole rollouts and PPO updates.
`base.cudagraphs=-1` disables capture; zero also enables capture upstream.
Native rewards pass to PPO without the trainer's default `[-1, 1]` clipping.
The wrapper preserves real recurrent-state carry and terminal resets. Loss,
state, transition, checkpoint, and training validation must be performed
against the implemented runtime before reporting a working training result.

## Verified compilation and asset loading

On Spark, the pinned trainer and `puffer_env.cu` compiled successfully with
CUDA 13.0 targeting `sm_121`, using C++17 as required by the upstream source.
`motion_assets.cu` also compiled successfully. An independent native host
asset validator loaded the existing semantic bundle and foot-feature bundle,
checked file digests, shapes, finite values, unit quaternions, 24 routes, and
17 moves. The semantic manifest digest was
`7d4719a3ca1e9e5a8faf571bc3c5c70e2b4c9e841be34303fa0a3f47d3692a28`.
Its ELF dependencies contain no CUDA or Python libraries; this validation
made zero CUDA calls. These checks establish source compatibility and asset
loading, with runtime behavior and training verified separately.
