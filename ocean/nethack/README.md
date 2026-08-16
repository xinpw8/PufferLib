# NetHack

PufferLib environment for NetHack 3.6.6 over
[fast-nle](https://github.com/FinlaySanders/fast-nle): 26-verb factored
action space (verb, item slot, direction, spell slot), legality masking,
decomposed-score reward, custom CUDA encoder/decoder (`ocean/nethack/nethack.cu`).

## Setup

```bash
pip install -e .
./build.sh nethack    # clones + builds vendor/fast-nle, then the training backend
```

Run from the repo root — the engine finds its data at
`vendor/fast-nle/build/dat` (override with `NETHACKDIR`).

## Train

```bash
./puffer train nethack
```

Reward coefficients and hypers live in `config/nethack.ini`.

## Watch a policy

```bash
./build.sh nethack --fast    # builds the ./nethack demo binary
./nethack                    # interactive TTY demo
NH_WEIGHTS=checkpoints/nethack/<run>/<step>.bin ./nethack
```

Interactive controls (default when stdin is a TTY):

| Key | Action |
|-----|--------|
| `Space` | one policy step; hold advances at **5 Hz** |
| `Shift+Space` | hold advances at **20 Hz** (fallback: hold `S`) |
| `q` / `Esc` | quit |

Weight resolution: `NH_WEIGHTS` if set (`score` / `depth` shorthands, or
a path), else `resources/nethack/nethack_score_weights.bin`.
`NH_MULTI=1` randomizes role/race/gender/align. `NH_TTY=1` shows the
engine tty map. Set `NH_SEED` to replay a seed.

```bash
./nethack 10000 0            # headless 10k steps (prints avg_score / avg_max_depth)
./nethack 1000 50            # auto-run at 50 ms/frame
```

GPU eval (same stack as training rollouts):

```bash
./puffer eval nethack --load-model-path=latest
# or pin a run:
./puffer eval nethack --load-model-path=checkpoints/nethack/<run>/<step>.bin
```
