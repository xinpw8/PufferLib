# PokeBattle Training Ledger

Methodical record of training runs, eval results, and changes for the Gen1 OU PokeBattle RL agent.

---

## Run 001 — Baseline (sparse reward, vs random bot)

**Date**: 2026-02-19
**Branch**: `poke-battle`
**Commit**: `2c3412c2` (eval tooling added on top of `89b7709c` training code)
**WandB**: https://wandb.ai/xinpw8/pufferlib/runs/0edxnpu9
**Run Name**: `denim-glade-1627`
**Checkpoint**: `experiments/puffer_poke_battle_0edxnpu9.pt` (epoch 954)

### Training Config
| Parameter | Value |
|---|---|
| total_timesteps | 1,000,000,000 |
| actual_steps | 1,034,944,512 |
| wall_time | ~32 minutes (1916s) |
| SPS | 2,138,192 |
| num_envs | 8,192 |
| selfplay | 0 (vs bot) |
| bot_mode | 0 (random) |
| optimizer | muon |
| learning_rate | 3e-4 (annealed to ~0) |
| min_lr_ratio | 0 (default — LR goes to zero) |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| batch_size | auto (=1,048,576) |
| minibatch_size | 32,768 |
| bptt_horizon | 128 |
| update_epochs | 2 |
| clip_coef | 0.2 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| policy | PokeBattle (hidden=256, ~152K params) |
| rnn | PokeBattleLSTM (hidden=256, ~526K params) |
| reward | terminal only: +1 win, -1 loss, 0 draw |

### Final WandB Summary Metrics
| Metric | Value |
|---|---|
| environment/p1_wins | 0.617 |
| environment/p2_wins | 0.383 |
| environment/draws | 0.0003 |
| environment/episode_return | 0.235 |
| environment/episode_length | 89.8 |
| environment/perf | 0.617 |
| losses/entropy | 1.754 |
| losses/explained_variance | -1.511 |
| losses/policy_loss | -0.00173 |
| losses/value_loss | 2.7e-13 |
| losses/approx_kl | 1.4e-8 |
| losses/clipfrac | 0.0 |
| learning_rate (final) | 8.2e-10 |

### Training Curve Observations
- **p1_wins**: Started somewhat above 0.5, drifted down over training, possibly recovered slightly at end. Jagged throughout. No clear monotonic improvement.
- **episode_return**: Similarly noisy and trendless; ~0.23 final.
- **episode_length**: Bounced initially, then gradually rose to ~90 and stayed there.
- **entropy**: Remained moderate (~1.75), suggesting the policy didn't collapse to deterministic but also didn't explore effectively.
- **explained_variance**: Negative (-1.51), meaning the value function is *worse than predicting the mean*. The value head learned nothing useful.
- **value_loss**: Essentially zero (2.7e-13), confirming the value function collapsed — it predicts ~0 everywhere, which is nearly correct given sparse reward.
- **clipfrac**: 0.0, approx_kl ~0 — by end of training (LR annealed to ~0), no policy updates were happening.

### Eval Results (at checkpoint epoch 954)
| Opponent | Episodes | Wins | Losses | Draws | Win Rate |
|---|---|---|---|---|---|
| Random bot | 100 | 60 | 40 | 0 | 60.0% |
| Heuristic bot | 100 | 1 | 99 | 0 | 1.0% |
| MCTS bot | 50 | 1 | 49 | 0 | 2.0% |
| Human | 6 | 0 | 6 | 0 | 0.0% |

### Diagnosis
**The agent barely learned anything.** 60% vs random is only marginally above the ~50% coin-flip baseline. The value function completely failed (explained_variance = -1.5), and by the end of training the learning rate had annealed to effectively zero with no meaningful policy updates occurring.

**Root causes identified (in priority order):**

1. **CRITICAL — Sparse terminal-only reward**: The agent receives +1/-1 only at game end (after ~90 steps on average). With gamma=0.99, reward at step 90 is discounted to 0.99^90 = 0.41. The agent has zero intermediate signal for damage dealt, KOs, HP preservation, or status advantages. Credit assignment is nearly impossible.

2. **CRITICAL — Value function failure**: explained_variance = -1.5 means the value head learned nothing. This is a direct consequence of sparse reward — the value function has no consistent signal to fit. With no working value baseline, advantage estimation is pure noise, and policy gradient updates are effectively random.

3. **MODERATE — Learning rate annealing to zero**: `anneal_lr=True` with `total_timesteps=1B` means LR cosine-anneals to `lr * min_lr_ratio = 0`. By epoch 954 it reached 8e-10 — effectively zero. The agent stopped learning entirely.

4. **MODERATE — Training only vs random bot**: The random bot is an extremely weak opponent. The agent may learn exploits that don't generalize.

5. **LOW — Observation encoding**: Species/types encoded as ordinals rather than categorical. Move secondary effects not encoded. These add unnecessary difficulty but are not the primary bottleneck.

---

## Run 002 — Reward shaping + LR floor (vs random bot)

**Date**: 2026-02-20
**Branch**: `poke-battle`
**Commit**: `b07380c9` (reward shaping + LR fix), eval fix `6cd7bd05`
**WandB**: https://wandb.ai/xinpw8/pufferlib/runs/hi4x9oc9
**Run Name**: `proud-sun-1628`
**Checkpoint**: `experiments/puffer_poke_battle_hi4x9oc9/model_puffer_poke_battle_000200.pt` (epoch 200)
**Status**: Killed early — perf ~1.0 within ~20 seconds, continued for ~12 minutes total

### Changes from Run 001
1. **Reward shaping** (commit `b07380c9`): Per-step intermediate rewards based on HP deltas and KOs:
   - Damage dealt to opponent: `+0.05 * (damage / opponent_max_team_hp)`
   - Damage taken: `-0.05 * (damage / own_max_team_hp)`
   - KO opponent pokemon: `+0.1` per KO
   - Lose own pokemon: `-0.1` per KO
   - Terminal win/loss: `+/-1.0` preserved on top
   - Effect: 75/200 steps now have non-zero reward (vs ~1/90 previously)
2. **LR floor** (commit `b07380c9`): Added `min_lr_ratio = 0.1` so cosine annealing floors at 3e-5 instead of zero. Prevents dead-learning state.

### Training Config (changes only)
| Parameter | Run 001 | Run 002 |
|---|---|---|
| reward | terminal only (+/-1) | shaped (damage/KO/terminal) |
| min_lr_ratio | 0 (LR -> 0) | 0.1 (LR -> 3e-5) |
| *all other params* | *same* | *same* |

### Final WandB Summary Metrics
| Metric | Run 001 | Run 002 | Notes |
|---|---|---|---|
| epoch | 954 | 200 | killed early |
| agent_steps | 1,034,944,512 | 385,875,968 | ~37% of Run 001 |
| wall_time | 1916s (~32min) | 733s (~12min) | killed early |
| SPS | 2,138,192 | 562,646 | lower due to shorter run / warm-up |
| environment/p1_wins | 0.617 | **0.996** | near-perfect vs random |
| environment/p2_wins | 0.383 | 0.004 | |
| environment/draws | 0.0003 | 0.0 | |
| environment/episode_return | 0.235 | **1.547** | shaped rewards accumulate higher |
| environment/episode_length | 89.8 | **25.2** | agent wins much faster |
| environment/perf | 0.617 | **0.996** | |
| losses/entropy | 1.754 | **0.189** | policy much more decisive |
| losses/explained_variance | -1.511 | **0.384** | value function actually works now |
| losses/policy_loss | -0.00173 | **-0.0128** | meaningful policy updates |
| losses/value_loss | 2.7e-13 | **0.00557** | value function fitting real signal |
| losses/approx_kl | 1.4e-8 | **0.000758** | healthy policy updates |
| losses/clipfrac | 0.0 | **0.00817** | clipping occurring normally |
| learning_rate (final) | 8.2e-10 | **2.12e-4** | still in useful range |

### Key Observations
- **Value function works**: explained_variance went from -1.5 (broken) to 0.38 (meaningful). The dense reward signal gives the value function something to fit.
- **Policy updates happening**: clipfrac=0.008, approx_kl=0.0008 — healthy PPO updates vs the flatlined zeros of Run 001.
- **Entropy collapsed**: 1.75 -> 0.19. The policy became very confident/decisive. May need higher ent_coef to maintain exploration for harder opponents.
- **Episode length dropped**: 89.8 -> 25.2 steps. The agent learned to win quickly against random bot.
- **LR still active**: 2.12e-4 at epoch 200 (vs 8e-10 at epoch 954 in Run 001). The min_lr_ratio=0.1 floor is working.

### Eval Results (at checkpoint epoch 200)
| Opponent | Episodes | Wins | Losses | Draws | Win Rate |
|---|---|---|---|---|---|
| Random bot | 100 | 98 | 2 | 0 | **98.0%** |
| Heuristic bot | 100 | 38 | 61 | 1 | **38.0%** |
| MCTS bot | 100 | 44 | 55 | 1 | **44.0%** |
| Human | 7 | 4 | 3 | 0 | **57.1%** |

### Comparison: Run 001 vs Run 002
| Opponent | Run 001 Win% | Run 002 Win% | Delta |
|---|---|---|---|
| Random | 60.0% | **98.0%** | +38pp |
| Heuristic | 1.0% | **38.0%** | +37pp |
| MCTS | 2.0% | **44.0%** | +42pp |
| Human | 0.0% | **57.1%** | +57pp |

### Assessment
**Reward shaping transformed learning.** In a fraction of the wall time (12min vs 32min) and fewer steps (386M vs 1B), the agent went from barely above random to competitive against search-based bots and a human player. The two changes — dense intermediate rewards and LR floor — together resolved the core credit assignment and dead-learning problems.

**Remaining weaknesses:**
- Still only 38-44% vs heuristic/MCTS — not yet dominant
- Entropy very low (0.19) — policy may be overfit to random bot patterns
- Training only vs random bot — harder opponents not seen during training

### Next Steps (Planned)
1. Train longer and/or against harder opponents (heuristic bot or curriculum)
2. Consider raising ent_coef to prevent entropy collapse when facing harder opponents
3. Try selfplay mode for more robust generalization
4. Investigate whether the policy has learned type effectiveness, switching, etc.

---

*Eval command: `python -m pufferlib.ocean.poke_battle.eval` (all bots) or `--human` (GUI play)*
*Specify checkpoint: `--model-path experiments/puffer_poke_battle_RUNID/model_puffer_poke_battle_EPOCH.pt`*
