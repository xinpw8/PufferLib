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
