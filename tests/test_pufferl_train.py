import json
from queue import SimpleQueue
from types import SimpleNamespace

import pytest

from pufferlib import pufferl


def test_train_worker_args_scale_total_timesteps_per_rank():
    args = {
        "world_size": 4,
        "train": {"total_timesteps": 1_000_000},
    }

    worker_args = pufferl._train_worker_args(args, rank=2, gpu_id=7)

    assert "local_total_timesteps" not in args["train"]
    assert "global_total_timesteps" not in args["train"]
    assert worker_args["train"]["total_timesteps"] == 1_000_000
    assert worker_args["train"]["global_total_timesteps"] == 1_000_000
    assert worker_args["train"]["local_total_timesteps"] == 250_000
    assert pufferl._local_total_timesteps(worker_args) == 250_000
    assert worker_args["rank"] == 2
    assert worker_args["gpu_id"] == 7


def test_train_worker_args_leave_single_gpu_budget_unchanged():
    args = {
        "world_size": 1,
        "train": {"total_timesteps": 123.5},
    }

    worker_args = pufferl._train_worker_args(args, rank=0, gpu_id=0)

    assert worker_args["train"]["total_timesteps"] == 123.5
    assert worker_args["train"]["global_total_timesteps"] == 123.5
    assert worker_args["train"]["local_total_timesteps"] == 123.5
    assert pufferl._local_total_timesteps(worker_args) == 123.5


class _TrainingBackend:
    def __init__(self, *, target_present=False, episode_count=None):
        self.target_present = target_present
        self.episode_count = episode_count
        self.saved = []
        self.closed = False
        self.train_calls = 0
        self.rollout_calls = 0
        self.state = SimpleNamespace(
            global_step=0, last_log_time=0, num_params=lambda: 16)

    def create_pufferl(self, args):
        return self.state

    def rollouts(self, state):
        self.rollout_calls += 1

    def train(self, state):
        self.train_calls += 1
        state.global_step += 2

    def log(self, state):
        logs = {
            'agent_steps': state.global_step,
            'uptime': float(self.rollout_calls),
            'SPS': 12.0,
            'loss': {'policy': 0.25},
            'env': {'hits': 3.0},
        }
        if self.target_present:
            logs['env']['perf'] = 0.75
        return logs

    def eval_log(self, state):
        logs = {'env': {'hits': 5.0}}
        if self.target_present:
            logs['env']['perf'] = 0.8
        if self.episode_count is not None:
            logs['env']['n'] = self.episode_count
        return logs

    def save_weights(self, state, path):
        self.saved.append(path)

    def close(self, state):
        self.closed = True


@pytest.fixture
def training_case(monkeypatch, tmp_path):
    args = {
        'env_name': 'test_env',
        'rank': 0,
        'gpu_id': 0,
        'run_id': 'test_run',
        'wandb': False,
        'vec': {'total_agents': 2},
        'train': {'total_timesteps': 8, 'horizon': 1},
        'sweep': {'metric': 'perf', 'downsample': 1},
        'checkpoint_dir': str(tmp_path / 'checkpoints'),
        'log_dir': str(tmp_path / 'logs'),
        'checkpoint_interval': 10,
        'eval_episodes': 10,
    }
    monkeypatch.setattr(pufferl, 'print_dashboard', lambda *a, **kw: None)
    monkeypatch.setattr(pufferl.selfplay, 'setup', lambda *a, **kw: None)
    monkeypatch.setattr(pufferl.selfplay, 'sync', lambda *a, **kw: None)
    monkeypatch.setattr(pufferl.selfplay, 'step', lambda *a, **kw: None)

    def run(backend, **kwargs):
        monkeypatch.setattr(pufferl, '_resolve_backend', lambda a: backend)
        pufferl._train('test_env', args, **kwargs)

    return args, tmp_path / 'logs' / 'test_env' / 'test_run.json', run


@pytest.mark.parametrize('downsample', [1, 3])
def test_train_saves_actual_metrics_without_sweep_target(training_case, downsample):
    args, log_path, run = training_case
    args['sweep']['downsample'] = downsample
    backend = _TrainingBackend()

    run(backend)

    metrics = json.loads(log_path.read_text())['metrics']
    assert metrics['agent_steps'][-1] == 8
    assert metrics['SPS'][-1] == 12.0
    assert metrics['loss/policy'][-1] == 0.25
    assert metrics['env/hits'][-1] == 5.0
    assert 'env/perf' not in metrics
    assert 'env/n' not in metrics
    assert backend.train_calls == 4
    assert backend.rollout_calls == 6
    assert len(backend.saved) == 2
    assert backend.closed
    if downsample > 1:
        assert metrics['agent_steps'][0] < metrics['agent_steps'][-1]
        assert metrics['env/hits'][0] == 3.0


def test_missing_sweep_target_reports_unscored_trial_and_keeps_logs(
        training_case, capsys):
    args, log_path, run = training_case
    backend = _TrainingBackend(episode_count=20)
    result_queue = SimpleQueue()

    def early_stop(*args):
        pytest.fail('Early stopping requires a measured target')

    run(backend, sweep_obj=SimpleNamespace(early_stop=early_stop),
        result_queue=result_queue)

    assert result_queue.get_nowait() == (0, [], [], [])
    assert result_queue.empty()
    assert 'env/perf' not in json.loads(log_path.read_text())['metrics']
    assert "sweep metric 'env/perf' was not reported" in capsys.readouterr().out
    assert backend.rollout_calls == 5
    assert backend.closed


def test_measured_sweep_target_preserves_scoring_and_early_stop(training_case):
    args, log_path, run = training_case
    backend = _TrainingBackend(target_present=True, episode_count=20)
    result_queue = SimpleQueue()
    early_stop_calls = []

    def early_stop(logs, target):
        early_stop_calls.append((logs['agent_steps'], target))
        return False

    run(backend, sweep_obj=SimpleNamespace(early_stop=early_stop),
        result_queue=result_queue)

    assert result_queue.get_nowait() == (0, [0.8], [4.0], [8])
    assert early_stop_calls == [(step, 'env/perf') for step in (2, 4, 6, 8)]
    assert json.loads(log_path.read_text())['metrics']['env/perf'] == [0.8]
    assert backend.saved == []
    assert backend.closed


def test_nonowner_rank_does_not_publish_logs_or_sweep_results(training_case):
    args, log_path, run = training_case
    args['rank'] = 1
    result_queue = SimpleQueue()
    backend = _TrainingBackend()

    run(backend, result_queue=result_queue)

    assert not log_path.exists()
    assert result_queue.empty()
    assert backend.saved == []
    assert backend.closed


def test_league_trial_without_local_target_keeps_async_scoring(
        training_case, monkeypatch):
    args, log_path, run = training_case
    args['sweep']['league'] = True
    backend = _TrainingBackend()
    finished = []
    monkeypatch.setattr(pufferl.league, 'finish_trial',
        lambda *a: finished.append(a))

    run(backend, sweep_obj=object())

    assert len(finished) == 1
    assert finished[0][2] == backend.saved[-1]
    assert len(finished[0][3]) == 4
    assert 'env/perf' not in finished[0][4]
    assert backend.rollout_calls == 4
    assert not log_path.exists()
    assert backend.closed


def test_zero_step_run_does_not_fabricate_metrics(training_case):
    args, log_path, run = training_case
    args['train']['total_timesteps'] = 0
    args['sweep']['downsample'] = 3
    backend = _TrainingBackend()

    run(backend)

    assert json.loads(log_path.read_text())['metrics'] == {}
    assert backend.rollout_calls == 0
    assert backend.saved == []
    assert backend.closed
