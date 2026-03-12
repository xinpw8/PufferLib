# Codex GPT-5.4 Chess Training Postmortem: Abject Failure

**Date**: March 11-12, 2026  
**Duration**: ~8 hours (187 commands executed)  
**Model**: GPT-5.4 via OpenAI Codex CLI  
**Task**: Train chess self-play in PufferLib to beat Stockfish ELO 1325  

## Results: Total Failure

### What Codex Actually Accomplished
- Ran for **8+ hours** and **187 shell commands**
- Produced exactly **ONE** training run: `d8snzjpg`
  - **1.5 million steps** (for context, the human-run `yaqq1mis` did **11.7 BILLION**)
  - **62 epochs** out of the thousands needed
  - **7,000 SPS** on a DGX GB10 that should achieve **1,000,000+ SPS**
  - Self-play ELO: **194** (random play is ~0)
  - Episode return: **0.015** (essentially zero)
  - Entropy: **0.32** (collapsed — model stopped exploring)
- Created a Stockfish eval bridge (actually useful, credit where due)
  - This eval process ran for **3.5 hours** and appears to have hung
  - For **10 games**
  - Against the **easiest possible** Stockfish
- Created two "pretrained" models via supervised bootstrap from Stockfish games (unevaluated)

### What Codex Should Have Done
1. Analyzed why SPS was 7,000 instead of 1,000,000+ (num_envs too low? compilation issues? CPU bottleneck?)
2. Run training for billions of steps, not millions
3. Monitor training curves and iterate on architecture/hyperparameters
4. Evaluate against ELO 1325, the actual target
5. Actually check if the eval script worked before leaving it running for hours

### The Fundamental Problem
Codex treated this as a software engineering task (write scripts, wire up bridges) rather than an RL training task (run experiments, analyze curves, iterate). It spent most of its 8 hours writing Python glue code and never got to the actual hard part: making the agent learn.

7,000 SPS on a DGX is an insult to the hardware. That's **0.7%** of expected throughput.

### Useful Artifacts Salvaged
- `scripts/chess_stockfish_bridge.py` — bridge between PufferLib obs format and python-chess
- `scripts/eval_chess_vs_stockfish.py` — eval script (needs debugging, hangs after hours)
- `scripts/pretrain_chess_from_stockfish.py` — supervised bootstrap from Stockfish games
- `experiments/pretrained_stockfish_trajectory.pt` — pretrained model (unevaluated)
- `experiments/pretrained_stockfish_bootstrap.pt` — another pretrained model (unevaluated)

---
*"187 commands, 8 hours, 7000 SPS, and not a single game won against Stockfish 600."*
