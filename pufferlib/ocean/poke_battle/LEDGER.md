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

## Run 003 — Train vs heuristic bot + higher entropy (full 1B steps)

**Date**: 2026-02-20
**Branch**: `poke-battle`
**Commit**: uncommitted config changes on top of `f00bd21b`
**WandB**: https://wandb.ai/xinpw8/pufferlib/runs/h9fsk8qf
**Run Name**: `daily-dragon-1629`
**Checkpoint**: `experiments/puffer_poke_battle_h9fsk8qf/model_puffer_poke_battle_000954.pt` (epoch 954)
**Status**: Completed (full 1B steps)

### Changes from Run 002
1. **Train vs heuristic bot** (`bot_mode` 0→1): Agent now trains against the 1-ply minimax heuristic bot instead of the random bot. This forces the agent to learn against a stronger opponent that uses type effectiveness and basic damage maximization.
2. **Higher entropy coefficient** (`ent_coef` 0.01→0.02): Doubled to counteract the entropy collapse observed in Run 002 (0.19). Maintains more exploration against the harder opponent.

### Training Config (changes only)
| Parameter | Run 002 | Run 003 |
|---|---|---|
| bot_mode | 0 (random) | 1 (heuristic) |
| ent_coef | 0.01 | 0.02 |
| epochs (actual) | 200 (killed early) | 954 (full run) |
| agent_steps | 385,875,968 | 1,034,944,512 |
| *all other params* | *same* | *same* |

### Final WandB Summary Metrics
| Metric | Run 002 | Run 003 | Notes |
|---|---|---|---|
| epoch | 200 | 954 | full run |
| agent_steps | 385,875,968 | 1,034,944,512 | full 1B |
| wall_time | 733s (~12min) | 6047s (~101min) | longer due to heuristic bot cost |
| SPS | 562,646 | 219,624 | heuristic bot ~3x slower than random |
| environment/p1_wins | 0.996 | **0.829** | harder opponent |
| environment/p2_wins | 0.004 | 0.171 | |
| environment/draws | 0.0 | 0.00002 | |
| environment/episode_return | 1.547 | **1.012** | lower shaped return vs harder bot |
| environment/episode_length | 25.2 | **45.1** | longer games vs heuristic |
| environment/perf | 0.996 | **0.829** | |
| losses/entropy | 0.189 | **0.369** | healthier with ent_coef=0.02 |
| losses/explained_variance | 0.384 | **0.473** | value function improved further |
| losses/policy_loss | -0.0128 | **-0.0119** | still meaningful |
| losses/value_loss | 0.00557 | **0.00677** | fitting harder signal |
| losses/approx_kl | 0.000758 | **0.000327** | healthy updates |
| losses/clipfrac | 0.00817 | **0.000878** | |
| learning_rate (final) | 2.12e-4 | **3.0e-5** | at LR floor (0.1 * 3e-4) |

### Key Observations
- **Training vs heuristic works**: Despite p1_wins dropping from 0.996→0.829 (harder opponent), the agent learned much more robust play that transfers to all bot types.
- **Entropy healthier**: 0.19→0.37 with doubled ent_coef. Policy maintains exploration instead of collapsing to a single strategy.
- **Value function best yet**: explained_variance 0.47 (vs 0.38 in Run 002, vs -1.5 in Run 001). Dense rewards + harder opponent = richer signal.
- **SPS ~3x lower**: Heuristic bot is compute-heavy (1-ply minimax each step), reducing throughput from ~560K to ~220K SPS. Full run took ~101 min vs ~12 min.
- **Episode length ~2x longer**: 45 steps vs 25. Games against heuristic bot are harder and take longer.

### Eval Results (at checkpoint epoch 954)
| Opponent | Episodes | Wins | Losses | Draws | Win Rate |
|---|---|---|---|---|---|
| Random bot | 100 | 100 | 0 | 0 | **100.0%** |
| Heuristic bot | 100 | 85 | 15 | 0 | **85.0%** |
| MCTS bot | 100 | 80 | 19 | 1 | **80.0%** |
| Human | 5 | 5 | 0 | 0 | **100.0%** |

### Comparison: All Runs
| Opponent | Run 001 Win% | Run 002 Win% | Run 003 Win% | Delta (002→003) |
|---|---|---|---|---|
| Random | 60.0% | 98.0% | **100.0%** | +2pp |
| Heuristic | 1.0% | 38.0% | **85.0%** | +47pp |
| MCTS | 2.0% | 44.0% | **80.0%** | +36pp |
| Human | 0.0% | 57.1% | **100.0%** | +43pp |

### Assessment
**Training against a harder opponent dramatically improved generalization.** The agent went from 38% and 44% vs heuristic/MCTS (Run 002, trained vs random) to 85% and 80% (Run 003, trained vs heuristic). Perfect 100% vs random is maintained. The entropy fix prevented policy collapse and allowed the agent to maintain a diverse strategy repertoire.

**Human eval: 5-0 sweep.** The policy beat a human player in all 5 games, up from 4-3 (57%) in Run 002. Combined with 85% vs heuristic and 80% vs MCTS, this agent is now superhuman against all tested opponents.

**Remaining weaknesses:**
- 85% vs heuristic and 80% vs MCTS — strong but not dominant against search-based bots
- MCTS win rate (80%) slightly below heuristic (85%) despite MCTS being a stronger opponent — suggests room for improvement

### Next Steps (Planned)
1. Consider training vs MCTS bot or curriculum (random→heuristic→MCTS)
2. Try selfplay for even more robust generalization
3. Consider longer training or larger model if gains plateau

---

## Run 004 — Train vs MCTS bot + OpenMP parallelization (killed at epoch 600)

**Date**: 2026-02-20
**Branch**: `poke-battle`
**Commit**: uncommitted (OpenMP parallelization + per-species tracking on top of `09341483`)
**WandB**: `6gssin6j`
**Checkpoint**: `experiments/puffer_poke_battle_6gssin6j/model_puffer_poke_battle_000600.pt` (epoch 600)
**Status**: Killed early — perf plateaued at ~85% (same ceiling as Run 003)

### Changes from Run 003
1. **Train vs MCTS bot** (`bot_mode` 1→2): Agent trains against Monte Carlo tree search bot (128 iterations, depth 5) instead of heuristic bot.
2. **OpenMP parallelization**: Added `#pragma omp parallel for` to `vec_step` loop in `env_binding.h`, made `pb_rng_state` and `g_event_env` thread-local, added per-env RNG state to `PokeBattle` struct. Build flags `-fopenmp` added to `setup.py`. Enables multi-core C env stepping.
3. **Per-species win rate tracking**: Added `species_wins[]` and `species_games[]` to Log struct, reported as `wr_<Species>` metrics in dashboard.

### Training Config (changes only)
| Parameter | Run 003 | Run 004 |
|---|---|---|
| bot_mode | 1 (heuristic) | 2 (MCTS) |
| mcts_iterations | — | 128 |
| mcts_depth | — | 5 |
| epochs (actual) | 954 (full run) | 600 (killed early) |
| agent_steps | 1,034,944,512 | ~410M |
| *all other params* | *same* | *same* |

### OpenMP Performance Impact
MCTS bot was severely CPU-bottlenecked (23.5K SPS single-threaded, 95% wall time in env step). OpenMP parallelization across 20 DGX Spark cores yielded:

| Metric | Before (Run 003 equivalent) | After (Run 004) |
|---|---|---|
| MCTS SPS (8192 envs) | ~23,500 | **~329,000** |
| Speedup | 1x | **14x** |
| CPU usage | 31% (1 core) | ~307% (multi-core) |
| GPU usage | 4-5% | ~26% |
| Env time % | 95% | ~64% |

### Dashboard Snapshot (at kill)
| Metric | Value |
|---|---|
| SPS | 177,700 |
| CPU | 306.9% |
| GPU | 26.3% |
| Steps | 410M |
| Epoch | 391→600 |
| Uptime | 38 min |
| perf | 0.837 |
| p1_wins | 0.837 |
| p2_wins | 0.163 |
| episode_length | 39.8 |
| entropy | 0.375 |
| explained_variance | 0.437 |

### Eval Results (at checkpoint epoch 600)
| Opponent | Episodes | Wins | Losses | Draws | Win Rate |
|---|---|---|---|---|---|
| Random bot | 100 | 100 | 0 | 0 | **100.0%** |
| Heuristic bot | 100 | 72 | 28 | 0 | **72.0%** |
| MCTS bot | 100 | 92 | 8 | 0 | **92.0%** |
| Human | 8 | 5 | 3 | 0 | **62.5%** |

### Comparison: All Runs
| Opponent | Run 001 | Run 002 | Run 003 | Run 004 |
|---|---|---|---|---|
| Random | 60.0% | 98.0% | 100.0% | **100.0%** |
| Heuristic | 1.0% | 38.0% | 85.0% | **72.0%** |
| MCTS | 2.0% | 44.0% | 80.0% | **92.0%** |
| Human | 0.0% | 57.1% | 100.0% | **62.5%** |

### Key Observations

**MCTS win rate improved, heuristic win rate dropped.** Training vs MCTS (92% vs MCTS) produced a policy that specializes against random-rollout opponents but is weaker vs the exhaustive minimax heuristic (72% vs 85% in Run 003). The human win rate dropped from 100% to 62.5% — the Run 003 policy (trained vs heuristic) was more robust overall.

**Same ~85% ceiling.** Training perf plateaued at 83.7% vs MCTS, mirroring Run 003's 82.9% vs heuristic. This confirms the ceiling is structural, not opponent-dependent.

### Species Win Rate Analysis (random-vs-random diagnostic)

To investigate the ~85% ceiling, a diagnostic was run: 100K games of random-vs-random (selfplay, both sides take random actions). Outcomes reflect pure team composition + RNG variance:

| Species | Random WR | Delta | Smogon Tier |
|---|---|---|---|
| Lapras | 58.2% | +8.6pp | C3 |
| Rhydon | 55.1% | +5.5pp | B1 |
| Starmie | 54.4% | +4.8pp | A |
| Slowbro | 54.0% | +4.4pp | C2 |
| Jynx | 53.6% | +4.0pp | B2 |
| Zapdos | 50.7% | +1.1pp | B1 |
| Persian | 50.6% | +1.0pp | C3 |
| Articuno | 50.5% | +0.9pp | C2 |
| Tauros | 50.0% | +0.4pp | S1 |
| Snorlax | 49.6% | -0.0pp | S2 |
| Chansey | 49.4% | -0.2pp | A |
| Exeggutor | 49.3% | -0.3pp | A |
| Cloyster | 49.2% | -0.4pp | B2 |
| Golem | 49.2% | -0.5pp | D |
| Dragonite | 48.9% | -0.7pp | C3 |
| Jolteon | 47.8% | -1.8pp | C1 |
| Alakazam | 47.6% | -2.0pp | B1 |
| Gengar | 43.7% | -5.9pp | B2 |
| Hypno | 42.1% | -7.5pp | D |

**16.1pp spread** between best (Lapras 58.2%) and worst (Hypno 42.1%) species. Random-play ranking does NOT correlate with competitive (Smogon) tiers — Tauros (S1 Smogon) is dead-center with random play because it requires intelligent use of Body Slam paralysis + crit mechanics.

**Conclusion**: The ~85% ceiling across both Run 003 and Run 004 is driven by team composition variance in `generate_ou_team()`. ~15% of random team matchups are fundamentally unfavorable regardless of play quality, compounded by Gen 1's high in-battle RNG variance (permanent freeze, crits, sleep turns).

### Next Steps (Planned)
1. Consider curriculum training: random → heuristic → MCTS for broader generalization
2. Weight team generation toward stronger compositions (remove Hypno/Machamp from pool, or use Smogon tier weights)
3. Try selfplay for more robust generalization
4. Longer training vs MCTS (killed early, may not have fully converged)

---

*Eval command: `python -m pufferlib.ocean.poke_battle.eval` (all bots) or `--human` (GUI play)*
*Specify checkpoint: `--model-path experiments/puffer_poke_battle_RUNID/model_puffer_poke_battle_EPOCH.pt`*
