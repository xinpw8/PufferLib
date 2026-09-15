# Selected V4 policy league

The actual native compact CUDA worker completed all 60 paired-side matches
with exit 0: no forfeits or invalid trials. Three opponents played every
unordered pairing on both fighter sides, using seeds 1 through 10 and
20-second rounds. Starts were fixed; seeds vary sampled policy actions.
Training shaping was zero.

| Opponent | Rank | Wins | Losses | Draws | Games |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected V4 curriculum | 1 | 39 | 0 | 1 | 40 |
| Older V3 33M | 2 | 20 | 19 | 1 | 40 |
| Scripted | 3 | 0 | 40 | 0 | 40 |

V4 beat scripted 20/20 and recorded 19 wins plus one draw against V3. The
league ranks completed side-reversed pairs using the Wilson lower bound on
observed wins. A draw does not become a win. These standings concern this
three-opponent pool; they do not establish human strength or authentic REK
parity. The separate diverse frozen suite contains the 300-second results.

`results.public.json` includes exact checkpoint identities, all outcomes,
scores, durations and per-opponent statistics. Its SHA-256 is
`5e94993c3a23b907f447c4d980dcb6cd09d7be147eee5057c27b4101fe0c3b86`.
Config identity is
`78e1d623906959a9f954fd08be87a50c9ee94f3d2c74a4eca86e8913868c2e61`.

The tournament took 57.63 seconds of host wall time including 60 worker
startups. This is not training SPS. Node orchestrated matches; native CUDA
performed arena transitions and BF16 policy inference. No Python, CPU physics,
rendering or PPO updates occurred in the tournament. Private checkpoints,
assets and full worker-state transcripts remain on Spark.
