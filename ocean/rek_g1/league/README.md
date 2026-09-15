# Native REK policy league and human evaluation

This dependency-free Node component stores policy identities and empirical
within-backend tournament results. Simulation, policy inference, and training
belong to the native C++/CUDA executables. There is no Python dependency here.

## Run the human evaluator

Build the native runtime with `native5/build_native.sh`, then build the viewer:

```sh
bash ocean/rek_g1/native5/build_eval.sh /absolute/runtime-build /absolute/new-eval-build
node ocean/rek_g1/league/prepare_spark.cjs /private/new-league /absolute/new-eval-build/rek-eval-worker
node ocean/rek_g1/league/server.cjs /private/new-league/server.json
```

The supplied Spark configuration listens on `127.0.0.1:18768`. Forward that port
through the existing WSL `ssh spark` connection for a Windows browser. Existing
port 18766 evaluators are independent and need not be stopped.

The two interactive backend choices explicitly run native CPU physics, with
CUDA controller and opponent inference. The renderer uses EGL and MuJoCo
forward kinematics for pictures only. It never advances physics. These
interactive measurements are not headless GPU training SPS. The native trainer
rejects both CPU viewer selectors.

Choose the physics backend, opponent, and human side, then load. The trained
policies were originally trained as fighter 0, so the UI defaults the human to
fighter 1 for trained opponents. Keyboard handlers belong to the browser arena;
they do not inject operating-system inputs. Held locomotion and yaw are
supported. Additional attacks do not stack; the one supported buffer is an
attack interrupting held yaw. Opponent output has only 33 combat categories,
with no quit, leave, reset, disconnect, or backend-switch category.

Runtime configuration, checkpoint weights, worker logs, and full match
transcripts stay in the private run directory. The public catalog exposes
checkpoint hashes, model contracts, and measured strength without checkpoint
paths. New checkpoints can be registered with `register_checkpoint.cjs`.
Use distinct IDs for sampled and greedy evaluation variants; never replace
the protocol behind an ID with recorded results.

## Reduced CUDA candidate on port 18769

This separate viewer uses the exact `fast_runtime.o` and `fast_assets.o` from
the compact GPU trainer. Evaluation advances the same 50 Hz CUDA simulation.
CPU MuJoCo kinematics is used only to render its 72-value GPU pose snapshot.
There is no Python runtime or CPU physics integration.

```sh
REK_EVAL_RUNTIME=semantic_cuda bash ocean/rek_g1/native5/build_eval.sh /absolute/fast-build /absolute/new-fast-eval
node ocean/rek_g1/league/prepare_fast.cjs /private/new-fast-league /absolute/new-fast-eval/rek-eval-worker /private/fast-runtime.json
node ocean/rek_g1/league/server.cjs /private/new-fast-league/server.json
```

The runtime JSON explicitly selects `backend: "semantic_cuda"`, the same
`model_path`, `assets_path`, `motion_features_path`, `round_seconds`, optional
17 `move_duration_ticks`, and `locomotion_segment_ticks` used in training.
Its optional `fast` object supplies `move_speed`, `yaw_speed`, `body_radius`,
`hit_speed`, and `down_damage`. Defaults match the compact runtime; every value
is explicitly passed to the worker and included in its configuration hash.
Only the number of parallel arenas changes to one for human evaluation.
Existing ports 18766 and 18768 remain independent.

The compact model uses prerecorded joint poses, approximate planar root
movement, strike/target volumes and collision response. It omits full
rigid-body dynamics and the shipped balance policies. Authentic REK parity
has not been established. The page labels these limitations. A good policy
in this model establishes performance within this model only.

The new league starts with its scripted opponent. Register only checkpoints
trained on this compact runtime using `register_checkpoint.cjs --backend
semantic_cuda` and their actual observation encoding and precision. Existing
MuJoCo/Puffysics checkpoints are deliberately excluded, despite matching
tensor dimensions. Configuration hashes include the exact runtime objects,
model, both asset manifests and behavior settings; rankings remain separate.
Checkpoint files contain FP32 weights, while the compact trainer's default
policy computation is BF16. Register that computation as `--precision bf16`.
An FP32-compute evaluation is a separate numerical variant, labeled accordingly
in the viewer; matching checkpoint storage does not establish matching inference.

## Tests

```sh
node --test ocean/rek_g1/league/league.test.cjs ocean/rek_g1/league/tournament.test.cjs
```

## Runtime API

Keep the manifest and checkpoints in a private runtime directory outside the
repository. Never publish checkpoints or local asset paths through the UI.

```js
const {League, hashFile} = require('./league.cjs');
const league = new League({file: '/private/rek-league/league.private.json'});
const configHash = '...64 lowercase SHA256 characters...';

league.registerPolicy({backend: 'mujoco', id: 'scripted-v1',
  label: 'Scripted baseline', kind: 'scripted', configHash,
  scriptedVersion: 'exact implementation commit or version'});

const filename = '/private/rek-league/checkpoints/mujoco-step8192.bin';
league.registerPolicy({backend: 'mujoco', id: 'step8192',
  label: 'MuJoCo step 8192', kind: 'trained', configHash,
  checkpoint: {path: filename, sha256: hashFile(filename),
    format: 'pufferlib5-native-bf16-v1', trainingSteps: 8192,
    model: {architecture: 'mingru', hiddenSize: 256, layers: 2,
      observations: 223, actions: 33}}});

const {matches} = league.schedulePair({id: 'mujoco-8192-scripted-seed1',
  backend: 'mujoco', configHash, policyA: 'step8192', policyB: 'scripted-v1',
  seed: 1, maxDurationMs: 120000});

for (const match of matches) {
  // Verify checkpoints immediately before loading them into the native worker.
  for (const id of match.players) league.verifyCheckpoint({backend: match.backend, id});
  league.startMatch(match.id);
  // The runner executes these fixed sides, seed, backend, and configuration.
  // Only actual observed match results may be passed to finishMatch.
  const observed = await runNativeMatch(match);
  league.finishMatch({id: match.id, backend: match.backend,
    configHash: match.configHash, seed: match.seed, players: match.players,
    status: observed.status, reason: observed.reason,
    scores: observed.scores, durationMs: observed.durationMs,
    loserPolicyId: observed.loserPolicyId});
}

const rows = league.standings({backend: 'mujoco', configHash});
const choices = league.opponentOptions({backend: 'mujoco', configHash});
```

`runNativeMatch` above is the caller's native-worker integration, not an
implemented function in the bookkeeping module. Checkpoint format is explicit
metadata; hashing does not prove that a native worker can load that format.
The worker must reject unsupported formats and architecture mismatches.

The environment configuration hash must cover behavior-affecting settings:
physics/model version, action and observation schemas, tick timing, scoring,
round limits, and opponent-independent termination rules. Checkpoints registered
under a different backend or configuration cannot participate in that fixture.

`schedulePair` returns two matches with identical seed and reversed player
assignments. `players[0]` is left/blue and `players[1]` is right/orange. Scores
are in this same order. `durationMs` and `maxDurationMs` are **elapsed simulated
physics time**, including reset pauses, not elapsed host wall time or solely
the displayed round clock. Set a separate wall-time watchdog in the
runner so a frozen policy cannot hold a match open indefinitely.

## Adjudication contract

| Result | Allowed reasons | Ranking treatment |
| --- | --- | --- |
| `completed` | `score_limit`, `round_limit`, `time_limit` | Score-derived by default; explicit native KO/tie outcomes supported below |
| `forfeit` | `policy_crash`, `policy_timeout`, `policy_disconnect`, `policy_quit`, `illegal_action` | Responsible `loserPolicyId` loses regardless of current score |
| `engine_error` | `physics_nonfinite`, `engine_crash`, `runner_interrupted`, `infrastructure_failure`, `round_redo` | Invalid trial; no wins or losses fabricated |

For an observed native terminal, supply `roundResult` and `winnerSide`:
`1` is points, `2` is KO, `3` is tie; sides are `0`/`1`, or `-1` for tie.
A points winner must agree with the score. A KO winner may have fewer recorded
clean hits; its official winner is retained without rewriting those hit counts.
An explicit native tie remains a draw. Optional `winnerPolicyId` must agree with
the side. A native redo (`4`) is unadjudicated and uses `round_redo`, not a win.
Here `engine_error` is the existing invalid-trial storage category; `round_redo`
does not claim that the physics engine crashed. Unknown score or duration may
be `null` for an invalid trial instead of invented zero measurements.

The runner must always finish a started match, including when a policy process
fails. A policy-owned failure is a forfeit. An engine/infrastructure failure
requires an invalid trial. If cause is unknown, report an infrastructure failure
and investigate; never relabel a known policy failure as an engine fault.
Scheduled/running matches remain explicitly pending and cannot produce wins.
On runner restart, recover interrupted matches and record their actual cause.

Policies must have no quit, leave, disconnect, UI reset, or backend-switch action.
Those are human/operator controls outside the action space. A runner-level
timeout is also required. This registry records adjudications; it does not
observe or stop processes itself.

## Strength reporting

Ranking is separate for each physics backend and environment configuration.
It uses the 95% Wilson lower bound on **win probability** from complete,
side-reversed pairs. A draw is a non-win in that interval. Match-point rate
(`win + 0.5 * draw`) and sample count break equal lower bounds. No reward,
training loss, checkpoint age, or claimed training step count increases rank.

All valid results, including single-leg forfeits, are retained in `total` and
`headToHead`. Only pairs with both legs `completed` or `forfeit` contribute to
`paired` ranking. A pair containing an engine fault is excluded from the paired
ranking without deleting its other observed leg. Complete paired trials on
fresh seeds to recover coverage. Duplicate IDs and duplicate seed/policy/config
fixtures are rejected instead of counted twice.

The status is `untested` before any result, `provisional` below 20 paired games,
and `measured` at 20 or more. Twenty is only a display/sample-count threshold.
It does not establish convergence, superiority to a human, or a narrow
confidence interval. Report exact W/D/L, interval, and head-to-head records.
Rank is relative to the evaluated pool and schedule. Use equal seeds and equal
pair counts in round-robin evaluation for comparable opponent coverage.

`opponentOptions` omits local checkpoint paths. `snapshot` is a private
administrative API and must not be exposed verbatim over HTTP.

## Persistence and concurrency

Each mutation acquires an exclusive writer lock, reads the newest manifest,
writes and fsyncs a temporary file, then renames it over the manifest. Readers
observe either the previous or next complete manifest. Failed validations do
not change the file. The small registry is persisted as one JSON document;
matches are immutable after completion and form the result history.

An overlapping writer receives an error and can retry later. A process killed
inside a mutation may leave `*.lock`; inspect the recorded PID and host state
before removing a stale lock. There is no automatic lock stealing. Rename and
fsync durability follow the host filesystem; keep this on local runtime storage
rather than a network share if strict crash consistency is needed.

## Human evaluation page

`public/` contains the browser frontend for the root native-worker server.
Backend and opponent selectors show measured ranking and explicit untested
status. Keyboard handlers activate only while the arena itself is focused.
Blur, hidden tab, selection changes, reset, and window exit release held keys.
These are browser-local inputs; no desktop input hooks or OS key injection are
used. The page never sends quit/disconnect actions to a policy.

## Native paired-side tournament runner

The runner uses the same private server configuration and registered policy
identities as the human evaluator. It launches the backend's native worker in
headless mode and never requests a rendered frame.

```sh
node ocean/rek_g1/league/tournament.cjs \
  --config /private/rek-league/server.json \
  --backend mujoco \
  --policies step8192,step16384,scripted-v1 \
  --seeds 1,2,3,4,5 \
  --out /private/rek-league/tournaments/mujoco-initial \
  --timeout-ms 180000 \
  --chunk-steps 64
```

Omitting `--policies` includes all compatible registered opponents in the
backend. Every unordered pair receives both side assignments for each seed.
`--deterministic true` remains the required default tournament protocol.
Each trained policy can explicitly set immutable
`checkpoint.model.actionSelection` to `greedy` or `sampled`. Omission means
`greedy`. The runner sends `deterministic: false` only for a policy explicitly
registered as `sampled`, independently for each side. Both modes receive the
fixture seed. An invalid enum is rejected before fixtures or workers are
created. The top-level flag cannot silently change existing policy identities.

Register sampled inference under a distinct policy ID, for example
`checkpoint8192-sampled`, even when its checkpoint file SHA256 matches an
existing greedy entry. Historical greedy results remain attached to the greedy
ID. The legacy MuJoCo 82/128 measurement used sampled native categorical
actions, so its matching evaluation entry must preserve that selection mode.
Seeds must fit a nonnegative native int32. Checkpoint model metadata also
forwards `observationEncoding`,
`recurrentResetTicks`, and `legacyFastHidden` to the native worker. In particular,
legacy MuJoCo FP32/raw223/reset64 inference must retain its explicit flags.

The worker configuration must explicitly contain `round_seconds`, currently
20 for the short-match setup, within `(0, 3600]`. The worker's native semantic
clock is 50 Hz. Physics ticks continue during physical reset pauses while the
official match clock can pause, so a 20-second round can legitimately finish
after more than 20 seconds of simulated physics time.

`round_reset_budget_seconds` is an optional finite allowance in `[0, 3600]`.
It defaults to `round_seconds`. The hard simulated-time bound is
`(round_seconds + round_reset_budget_seconds) * 1000 + 20` milliseconds,
rounded up to integer milliseconds. Default 20-second rounds therefore have
a **40.02-second simulated-time watchdog**, including one timer-boundary tick.
An explicit zero removes the reset allowance. This budget bounds a stalled
round; it does not change its official 20-second timer or award a win.

For example, an official terminal at tick 1009 records `durationMs: 20180`,
without replacing it with `20000`. Terminals and nonterminals beyond the hard
budget are invalid trials. The separate host wall-time watchdog also remains
active. `stopAtRound: true` stops on the first terminal rather than allowing an
automatic reset to erase the score. Full `rounds` terminal payloads are recorded
before the next leg. The seeded private worker config and summary record the
resolved reset budget. Already scheduled fixtures keep their original budget;
choose fresh seeds when changing it instead of rewriting their history.

Each leg creates a private directory containing its seeded worker configuration,
request/response JSONL transcript, and worker stderr. `summary.private.json`
contains final outcomes and current standings. These files may contain private
local paths and should not be committed or exposed as public web assets.

Each leg recreates the worker to reset physics, controller, policy memory, and
random state. This favors interpretable bounded evaluation over startup
throughput. Tournament wall time is not training SPS. Training performance
must be measured using the native headless training executable separately.

Only errors explicitly prefixed `policy:` with a known guilty side become
forfeits. Runtime errors such as `policy: side=0 inference failed` identify the
responsible policy. During a side-specific policy load, the side is known from
the request. Unattributed crashes/timeouts remain invalid trials. Neither an
unknown crash nor a physics nonfinite error awards the other policy a win.

A per-backend/config tournament lock prevents overlapping runners from
claiming the same fixtures. After a process crash, verify the PID in a stale
lock before removing it. On restart, abandoned `running` matches are recorded
as interrupted invalid trials; completed results are never duplicated.
Use fresh seeds for additional observations, including after invalid trials.

Programmatic entry point:

```js
const {runTournament} = require('./tournament.cjs');
const result = await runTournament({configPath: '/private/rek-league/server.json',
  backendId: 'puffysics', policyIds: ['step8192', 'scripted-v1'], seeds: [1, 2],
  runDirectory: '/private/rek-league/tournaments/puffysics-initial',
  timeoutMs: 180000, chunkSteps: 64, deterministic: true,
  emit: observedMatch => console.log(observedMatch)});
```

The fake-worker tests exercise fixture execution, seeded configs, policy loading
for both sides, score preservation, KO winners, draws/redos, policy forfeits,
engine faults, timeouts, interrupted recovery, and duplicate prevention. They
do not establish native GPU-worker compatibility or measured policy strength.
