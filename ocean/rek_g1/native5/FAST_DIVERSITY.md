# Compact V4 training diversity

V4 preserves V3 motion, swept-sphere contacts, contact points, 223 policy
observations, 33 actions, and timer scaling. It adds configurable GPU opponents,
optional seeded initial states, and optional potential-based training rewards.
It adds no synthetic hit, range-based action mask, knockdown, or hit reset.
Genuine falls and authentic REK parity remain unmodeled.

## Opponents and starting states

`REK_FAST_OPPONENT_MODE` accepts these values:

| Value | Behavior |
| --- | --- |
| `scripted` | Original deterministic approach/turn/attack cycle. Default. |
| `neutral` | Release inputs and stand still. |
| `retreat` | Face the learner, then back away if gap is below 1.25 m. No attacks. |
| `strafe` | Face the learner, then alternate strafe direction every 50 ticks. No attacks. |
| `mixed` | Choose one of those four modes uniformly per arena and round. |

Both default fighter-one control and override 2 use the configured opponent.
Override 1 still supplies an external action, including a frozen checkpoint.
The runtime does not select historical checkpoint files itself.

`REK_FAST_RANDOM_RESETS=1` enables seeded positions and headings. Its default is
`0`. The following parameters apply when enabled:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `REK_FAST_RESET_GAP_MIN` | 0.55 | Minimum center separation in m. |
| `REK_FAST_RESET_GAP_MAX` | 2.5 | Maximum center separation in m. |
| `REK_FAST_RESET_HEADING_SPREAD_RAD` | pi | Maximum heading offset from facing the other fighter. |

The configuration's existing seed, arena index, and round number determine a
stateless 32-bit hash. Gaps, pair axis, feasible midpoint, and heading offsets
are sampled on GPU. Bounds require nonoverlap and room for every sampled axis,
including a 0.01 m wall margin. Roots are not clamped after sampling. Resetting
the runtime replays the first-round fixtures exactly; advancing to another
round selects new fixtures. Fixed starts with the original scripted opponent
make no random-number calls.

## Optional training reward

`REK_FAST_SHAPING_WEIGHT=0` is the default and disables shaping. Positive
weight requires an explicit `REK_FAST_SHAPING_GAMMA` matching the learner's
discount. Additional parameters are `REK_FAST_SHAPING_TARGET` (default 0.65 m,
based on the isolated I-hit fixture) and `REK_FAST_SHAPING_BEARING_WEIGHT`
(default 0). These are training choices, not recovered game dynamics.

For each fighter, let `e = abs(center_gap - target)` and let `b` be absolute
wrapped opponent bearing divided by pi. With bearing weight `a`:

```text
Phi(s) = -(e / (1 + e) + a * b) / (1 + a)
reward = own_contact_points - opponent_contact_points
       + weight * (gamma * Phi(next_state) - Phi(previous_state))
```

The potential lies in [-1, 0]. Terminal next-state potential is exactly zero;
after an automatic reset, the next transition begins from the new initial
state. An idle negative-potential state can receive a small positive
per-step shaping value when gamma is below one. This is the specified
discounted potential difference. Its discounted sum through a terminal
telescopes to the negative initial potential, rather than awarding an
additional scoring objective for remaining idle.

Only reward buffers and logged episode return include shaping. Contact
points, hit observations, falls, winners, and match duration do not change.
Evaluation must explicitly set `REK_FAST_SHAPING_WEIGHT=0`; report actual
contact points and official results separately from shaped training return.
The benchmark does not establish AFK competence without an AFK test.

## Verification

`fast_diversity_probe.sh BUILD NEW_OUTPUT` compiles and executes a native
64-arena probe against the exact runtime object. It checks fixed defaults,
bitwise seeded replay, seed/episode differences, geometric bounds, all four
opponent modes, shaping arithmetic and terminal telescoping, and invariance
of state, action masks, observations, points and falls under shaping. It also
rejects invalid mode, reset bounds, heading spread, discount and target inputs.
The native GPU runtime performs every state transition. CPU snapshots and
assertions are test instrumentation. No Python or CPU physics is used.
