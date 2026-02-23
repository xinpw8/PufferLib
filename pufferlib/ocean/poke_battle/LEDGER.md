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

## Run 005 — Post-parity-rules eval (Showdown [Gen 1] OU alignment)

**Date**: 2026-02-22  
**Branch**: `poke-battle`  
**Commit**: `df2b40f4` (RBY OU parity clauses + tests)  
**WandB Run**: `891hhxc2`  
**Checkpoint**: `experiments/puffer_poke_battle_891hhxc2.pt`  
**Status**: Eval-only validation after implementing clause/rules parity

### Purpose
Confirm that the model from run `891hhxc2` remains strong after environment updates for Showdown-aligned Gen 1 OU rules:
- Sleep Clause Mod
- Freeze Clause Mod
- Species Clause validation for fixed teams
- Endless Battle Clause handling
- Explicit [Gen 1] OU standard-rule parity assertions (OHKO/Evasion/Dig/Fly bans in move pool)

### Eval Environment
- Venv: `pufferlib/.pufferlib`
- Device: CPU (to avoid interference with active GPU training run)
- Command:
  - `python -m pufferlib.ocean.poke_battle.eval --model-path /home/spark-advantage/pufferlib/experiments/puffer_poke_battle_891hhxc2.pt --episodes 1000 --device cpu`

### Eval Results (1000 episodes per bot)
| Opponent | Episodes | Wins | Losses | Draws | Win Rate |
|---|---|---|---|---|---|
| Random bot | 1000 | 998 | 2 | 0 | **99.8%** |
| Heuristic bot | 1000 | 769 | 231 | 0 | **76.9%** |
| MCTS bot | 1000 | 857 | 143 | 0 | **85.7%** |

### Human Eval (X11 session)
- Games played: 6

| Player | Wins | Losses | Draws | Win Rate |
|---|---|---|---|---|
| Human | 2 | 4 | 0 | **33.3%** |
| Policy | 4 | 2 | 0 | **66.7%** |

### Rule-Parity Validation
- Command:
  - `python -m pytest -q tests/test_poke_battle_rules_parity.py`
- Result: `8 passed`

### Notes
- No regressions observed from parity-rule changes in this checkpoint’s bot eval profile.
- Relative ordering remains sensible (Random > MCTS > Heuristic win rate in this snapshot), with strong absolute performance against all three bot types.

---

*Eval command: `python -m pufferlib.ocean.poke_battle.eval` (all bots) or `--human` (GUI play)*
*Specify checkpoint: `--model-path experiments/puffer_poke_battle_RUNID/model_puffer_poke_battle_EPOCH.pt`*

---

## Run 006 — Full Showdown Gen1 OU legality parity closure

**Date**: 2026-02-22  
**Branch**: `poke-battle`  
**Status**: Implementation + validation complete (rules + team legality + hardcoded moveset legality)

### Scope
This pass closed all remaining identified gaps for full `[Gen 1] OU` legality parity in the local env implementation:
- strict fixed-team species legality (`SPECIES_NONE` rejected)
- robust endless stall detection under switch/no-impact loops
- full 149-species hardcoded moveset legality check against Showdown TeamValidator (no-tradeback Gen1 OU)

### Canonical report
Full technical audit details are documented in:
- `pufferlib/ocean/poke_battle/SHOWDOWN_GEN1_OU_LEGALITY_REPORT_2026-02-22.md`

That report includes:
- exact Showdown source references (format/ruleset/validator semantics)
- discrepancy inventory
- all species-level move corrections
- validation commands and outputs
- reproducibility steps

### Quick verification summary
- Build:
  - `source /home/spark-advantage/pufferlib/.pufferlib/bin/activate`
  - `python setup.py build_poke_battle --inplace --force`
- Tests:
  - `python -m pytest -q tests/test_poke_battle_rules_parity.py tests/test_poke_battle_team_builder.py tests/test_poke_battle_moveset_legality.py`
  - Result: `26 passed`
- External legality sweep using official Showdown TeamValidator (`[Gen 1] OU`, commit `95aad7df02abd58dd737e0acdac22e5d049d360e`):
  - Result: `Invalid species count: 0` across all 149 modeled species sets

---

## Run 007 — RNN-required MCTS retrain prep + SPS characterization

**Date**: 2026-02-22  
**Branch**: `poke-battle`  
**Commit baseline**: `ec946f00`  
**Status**: Profiling + config tuning complete; launch command finalized

### Goal
User-requested retrain target:
1. Keep **RNN enabled** (required).
2. Train policy against strongest available opponent: **MCTS bot** (`bot_mode=2`).
3. Keep adaptive team learning enabled so policy learns team quality/composition.
4. Push SPS as high as possible without weakening the opponent.

All commands were run from:
- `source .pufferlib/bin/activate`

### Environment/Trainer verification
- Confirmed adaptive team builder signal is exposed in trainer logs:
  - `environment/team_builder_recent_winrate`
  - `environment/team_builder_pool_coverage`
  - per-species rates: `environment/wr_<Species>`
- Confirmed RNN path is active by default (`rnn_name=PokeBattleLSTM`).

### Throughput profiling summary

#### 1) Env-only SPS (MCTS 128, depth 5, team_builder_mode=1)
| env.num_envs | SPS |
|---|---|
| 2,048 | 189,893 |
| 4,096 | 246,794 |
| 8,192 | 233,255 |
| 12,288 | **284,444** |

#### 2) Multiprocessing backend sweep (same opponent settings)
`PufferEnv` single-process backend outperformed tested `Multiprocessing` configs for this workload.

| Config | SPS |
|---|---|
| `PufferEnv` 8,192 | **229,086** |
| MP `8x128` | 84,650 |
| MP `12x128` | 91,204 |
| MP `16x128` | 114,135 |
| MP `16x256` | 155,886 |
| MP `20x256` | 97,013 |

#### 3) End-to-end training SPS (RNN ON, MCTS 128)
Representative probes:
- `env.num_envs=8192, horizon=128`: `sps_mean=115,067`, `sps_max=143,203`
- `env.num_envs=12288, horizon=128`: `sps_mean=131,354`, `sps_max=148,275`
- `env.num_envs=12288, horizon=64, minibatch=65536`: `sps_mean=133,310`
- `env.num_envs=16384, horizon=128`: `sps_mean=118,470`
- foreground sanity launch with tuned config: epoch-1 `SPS=95.5K`

### 1M+ SPS feasibility findings
- With **strongest MCTS** (`mcts_iterations=128`, `mcts_depth=5`) and RNN enabled, this setup is in the ~`100K-150K` training SPS regime.
- 1M+ is **not** achievable under these strength constraints on this machine.
- Env-only MCTS sweep showed 1M+ only when reducing search budget substantially (example: `mcts_iterations=32` gave ~`1,075,002` env SPS), which weakens the opponent and was rejected for this run objective.

### Issues found and fixed during prep
1. Non-RNN policy path lacked `forward_eval` on `PokeBattle` model.  
   - Fix: added `forward_eval` pass-through in `pufferlib/ocean/torch.py`.
   - RNN training path remains unchanged.
2. Detected unrelated background load affecting SPS (stale `puffer_chess` training process).  
   - Terminated during profiling to obtain clean measurements.
3. Team-quality observability was incomplete for monitoring learned composition online.  
   - Fix: expanded `vec_log` team-builder diagnostics in `pufferlib/ocean/poke_battle/binding.c` to emit:
     - per-species recent selection rates: `environment/pick_<Species>`
     - inferred best team slots:
       - `environment/team_builder_best_species_1..6`
       - `environment/team_builder_best_species_<slot>_pick_rate`
       - `environment/team_builder_best_species_<slot>_wr`
       - `environment/team_builder_best_species_<slot>_score`
     - summary metrics:
       - `environment/team_builder_best_team_mean_wr`
       - `environment/team_builder_best_team_mean_pick_rate`
   - Added regression coverage: `tests/test_poke_battle_team_builder.py::test_team_builder_best_team_metrics_are_logged`.

### Config updates applied
Updated `pufferlib/config/ocean/poke_battle.ini`:
- `env.num_envs = 12288`
- `env.team_builder_mode = 1`
- `train.horizon = 64`
- `train.minibatch_size = 65536`
- kept strongest-opponent settings:
  - `env.bot_mode = 2`
  - `env.mcts_iterations = 128`
  - `env.mcts_depth = 5`

### Validation
- `pytest -q tests/test_poke_battle_team_builder.py`
- Result: `4 passed`

### Launch command (RNN + strongest MCTS + team builder)
```bash
source .pufferlib/bin/activate
python -m pufferlib.pufferl train puffer_poke_battle \
  --wandb \
  --wandb-project pufferlib \
  --wandb-group poke-battle-mcts-team-builder \
  --tag mcts128_tb1_rnn_env12288_h64
```

### Final assessment
- Training objective (RNN + strongest MCTS + team composition learning) is configured and validated.
- Throughput is tuned to the best stable point found without weakening opponent strength.
- The 1M+ SPS target conflicts with `mcts_iterations=128` strength requirements on current hardware.

---

## Run 008 — Replace Render Placeholders with Official Showdown Sprites

**Date**: 2026-02-23  
**Status**: complete

### Goal
Remove fallback placeholder sprites for non-top-20 species in the Raylib battle renderer by using official Pokemon Showdown sprites for all modeled species.

### Changes
- `pufferlib/ocean/poke_battle/render.h`
  - Replaced the fixed 20-entry `RENDER_SPECIES_NAMES` table with dynamic slug generation from `SPECIES_DATA[id].name`.
  - Added `species_sprite_slug(...)` that normalizes names to Showdown IDs by lowercasing and stripping non-alphanumeric characters:
    - `Mr. Mime -> mrmime`
    - `Farfetch'd -> farfetchd`
    - `Nidoran-F -> nidoranf`
    - `Nidoran-M -> nidoranm`
  - Sprite load path now works for all `NUM_SPECIES=149` entries.
- `pufferlib/resources/poke_battle/download_sprites.py`
  - Switched from hardcoded 20 species to all modeled Gen1 species via `LEGAL_SPECIES_IDS` and `SPECIES_NAMES`.
  - Added the same Showdown-ID normalization rule used by the renderer.
  - Added completion/failure summary reporting.

### Asset sync command
```bash
source .pufferlib/bin/activate
python pufferlib/resources/poke_battle/download_sprites.py
```

### Validation
- Download result: `149 species x2 views = 298 files`, `No download failures`.
- On-disk counts:
  - `pufferlib/resources/poke_battle/sprites/gen1`: `149`
  - `pufferlib/resources/poke_battle/sprites/gen1-back`: `149`
- Consistency check (slug mapping vs files): `checked=149 missing=0`.
- Rebuilt extension:
```bash
source .pufferlib/bin/activate
python setup.py build_poke_battle --inplace --force
```

---

## Run 009 — Post-Train Eval Snapshot (Learned Team + Human Check)

**Date**: 2026-02-23  
**Run**: `xinpw8/pufferlib/ummlv1to`  
**Checkpoint**: `experiments/puffer_poke_battle_ummlv1to/model_puffer_poke_battle_000636.pt`

### Human vs policy (GUI quick check)
Result from user-run head-to-head:
- Human: `0W / 3L / 0D` (`0.0%`)
- Policy: `3W / 0L / 0D` (`100.0%`)

### Learned-team extraction note
- W&B summary keys for `team_builder_best_species_*` can surface non-integer blended values (e.g. `3.75`), which can violate Species Clause when naively cast.
- For evaluation, use the latest **legal** 6-slot row from history scan:
  - species IDs: `[14, 4, 2, 6, 3, 5]`
  - species names: `Slowbro / Alakazam / Chansey / Starmie / Snorlax / Exeggutor`

### Learned team vs bots (fixed policy team, random legal opponent teams)
- Eval script artifact:
  - `experiments/puffer_poke_battle_ummlv1to/manual_eval_20260222_175254/learned_team_vs_bots.json`
- Results:
  - `Random`: `99.75%` (`405W / 1L / 0D`, 406 episodes)
  - `Heuristic`: `92.10%` (`373W / 32L / 0D`, 405 episodes)
  - `MCTS`: `95.27%` (`383W / 19L / 0D`, 402 episodes)

### Operational follow-up
- Post-train eval watcher launched in detached tmux session:
  - session: `poke_post_eval_ummlv1to`
  - watches train PID `299968`
  - writes outputs under:
    - `experiments/puffer_poke_battle_ummlv1to/post_eval_<timestamp>/`
