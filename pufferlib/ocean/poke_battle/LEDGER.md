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
| gamma | 0.99 |
| gae_lambda | 0.95 |
| batch_size | auto |
| minibatch_size | 32,768 |
| bptt_horizon | 128 |
| update_epochs | 2 |
| clip_coef | 0.2 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| policy | PokeBattle (hidden=256) |
| rnn | PokeBattleLSTM (hidden=256) |
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

3. **MODERATE — Learning rate annealing to zero**: `anneal_lr=True` with `total_timesteps=1B` means LR drops linearly. By epoch 954 it reached 8e-10 — effectively zero. The agent stopped learning. If intermediate rewards are added, either disable annealing or set a much higher timestep budget.

4. **MODERATE — Training only vs random bot**: The random bot is an extremely weak opponent. The agent may learn exploits that don't generalize (e.g., always using the same move works fine vs random but fails vs any opponent that considers type effectiveness).

5. **LOW — Observation encoding**: Species/types encoded as ordinals rather than categorical. Move secondary effects not encoded. These add unnecessary difficulty but are not the primary bottleneck.

### Next Steps (Planned)
1. Add reward shaping in `poke_battle.h` — intermediate rewards for damage, KOs, HP preservation
2. Re-evaluate with same eval suite
3. If reward shaping works, consider curriculum (random -> heuristic) and selfplay

---

*Eval command: `python -m pufferlib.ocean.poke_battle.eval` (all bots) or `--human` (GUI play)*
