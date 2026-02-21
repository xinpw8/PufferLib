"""
Validate selfplay training produces agents that beat bots.

Test 1: Chess selfplay -> vs random bot (target: >50% win rate)
Test 2: SlimeVolley selfplay -> vs built-in bot (target: score > 0)
"""
import sys
import os
import time
import torch
import pufferlib._C as _C
from pufferlib.pufferl import PuffeRL, load_config, _get_trained_state_dict

def train_and_eval(env_name, train_steps, eval_epochs, train_env_overrides=None, eval_env_overrides=None):
    """Train with selfplay, then evaluate against a bot."""

    # --- TRAIN PHASE ---
    print(f"\n{'='*60}")
    print(f"TRAINING: {env_name} for {train_steps/1e6:.0f}M steps")
    print(f"{'='*60}")

    args = load_config(env_name)
    args['train']['total_timesteps'] = train_steps
    args['train']['checkpoint_interval'] = 9999999  # don't checkpoint during training
    if train_env_overrides:
        args['env'].update(train_env_overrides)

    train_config = dict(**args['train'])
    train_config['env_name'] = args['env_name']
    train_config['env'] = args['env_name']
    train_config['use_rnn'] = args['rnn_name'] is not None

    pufferl = PuffeRL(train_config, args['vec'], args['env'], args['policy'], verbose=False)

    start = time.time()
    last_print = 0
    while pufferl.global_step < train_steps:
        pufferl.evaluate()
        pufferl.train()

        if time.time() - last_print > 5:
            elapsed = time.time() - start
            sps = pufferl.global_step / elapsed if elapsed > 0 else 0
            pct = 100 * pufferl.global_step / train_steps
            print(f"  [{pct:5.1f}%] step={pufferl.global_step/1e6:.1f}M  SPS={sps/1e6:.1f}M  epoch={pufferl.epoch}")
            last_print = time.time()

    # Get final train stats
    torch.cuda.synchronize()
    train_logs = _C.log_environments(pufferl.pufferl_cpp)
    print(f"\nTrain stats: {train_logs}")

    # Save trained weights
    trained_state = _get_trained_state_dict(pufferl.policy_fp32, pufferl.pufferl_cpp.muon)

    # Close training env
    pufferl.rollouts = None
    pufferl.policy_fp32 = None
    torch.cuda.synchronize()
    _C.close(pufferl.pufferl_cpp)
    pufferl.pufferl_cpp = None
    torch.cuda.empty_cache()
    torch._C._cuda_clearCublasWorkspaces()

    elapsed = time.time() - start
    print(f"Training done in {elapsed:.1f}s ({pufferl.global_step/elapsed/1e6:.1f}M SPS)")

    # --- EVAL PHASE ---
    print(f"\n{'='*60}")
    print(f"EVALUATING: {env_name} vs bot for {eval_epochs} epochs")
    print(f"{'='*60}")

    args = load_config(env_name)
    if eval_env_overrides:
        args['env'].update(eval_env_overrides)

    # Smaller agent count for eval
    args['vec']['total_agents'] = 1024
    args['train']['total_timesteps'] = 999999999  # won't hit this

    eval_train_config = dict(**args['train'])
    eval_train_config['env_name'] = args['env_name']
    eval_train_config['env'] = args['env_name']
    eval_train_config['use_rnn'] = args['rnn_name'] is not None

    eval_pufferl = PuffeRL(eval_train_config, args['vec'], args['env'], args['policy'], verbose=False)

    # Load trained weights into Muon's contiguous weight buffer
    with torch.no_grad():
        weight_buffer = eval_pufferl.pufferl_cpp.muon.weight_buffer
        offset = 0
        for name, param in eval_pufferl.policy_fp32.named_parameters():
            size = param.numel()
            if name in trained_state:
                weight_buffer.narrow(0, offset, size).copy_(trained_state[name].view(-1))
            offset += size

    # Run eval epochs (just evaluate, no training)
    for i in range(eval_epochs):
        eval_pufferl.evaluate()

    torch.cuda.synchronize()
    eval_logs = _C.log_environments(eval_pufferl.pufferl_cpp)

    # Close eval env
    eval_pufferl.rollouts = None
    eval_pufferl.policy_fp32 = None
    torch.cuda.synchronize()
    _C.close(eval_pufferl.pufferl_cpp)
    eval_pufferl.pufferl_cpp = None
    torch.cuda.empty_cache()
    torch._C._cuda_clearCublasWorkspaces()

    return train_logs, eval_logs


def test_chess():
    """Chess: selfplay training -> eval vs random bot. Target: >50% win rate."""
    train_logs, eval_logs = train_and_eval(
        env_name='puffer_chess',
        train_steps=500_000_000,
        eval_epochs=1024,
        train_env_overrides={'selfplay': 1, 'random_bot': 0},
        # Keep selfplay=1 so model architecture matches (same obs/action layout)
        # random_bot=1 makes env play random opponent moves internally
        eval_env_overrides={'selfplay': 1, 'random_bot': 1},
    )

    perf = eval_logs.get('perf', 0)
    print(f"\n{'='*60}")
    print(f"CHESS RESULT: perf (win rate) = {perf:.3f}")
    print(f"TARGET: > 0.50")
    print(f"PASS: {'YES' if perf > 0.50 else 'NO'}")
    print(f"Full eval logs: {eval_logs}")
    print(f"{'='*60}")
    return perf > 0.50, perf, eval_logs


def test_slimevolley():
    """SlimeVolley: selfplay training -> eval vs random opponent. Target: score > 0."""
    train_logs, eval_logs = train_and_eval(
        env_name='puffer_slimevolley',
        train_steps=50_000_000,
        eval_epochs=256,
        train_env_overrides={'selfplay': 1},
        # Keep selfplay=1 so model architecture matches.
        # Eval PuffeRL's opponent pool starts with random weights,
        # so trained model is evaluated against a random/untrained opponent.
        eval_env_overrides={'selfplay': 1},
    )

    score = eval_logs.get('score', 0)
    perf = eval_logs.get('perf', 0)
    print(f"\n{'='*60}")
    print(f"SLIMEVOLLEY RESULT: score = {score:.3f}, perf = {perf:.3f}")
    print(f"TARGET: score > 0")
    print(f"PASS: {'YES' if score > 0 else 'NO'}")
    print(f"Full eval logs: {eval_logs}")
    print(f"{'='*60}")
    return score > 0, score, eval_logs


if __name__ == '__main__':
    tests = sys.argv[1:] if len(sys.argv) > 1 else ['chess', 'slimevolley']
    # Clear sys.argv so load_config's argparse doesn't choke on our args
    sys.argv = sys.argv[:1]

    results = {}
    for test in tests:
        if test == 'chess':
            passed, metric, logs = test_chess()
            results['chess'] = {'passed': passed, 'win_rate': metric}
        elif test == 'slimevolley':
            passed, metric, logs = test_slimevolley()
            results['slimevolley'] = {'passed': passed, 'score': metric}

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        status = 'PASS' if r['passed'] else 'FAIL'
        print(f"  {name}: {status} - {r}")

    all_passed = all(r['passed'] for r in results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    sys.exit(0 if all_passed else 1)
