import pytest

from pufferlib import league
from pufferlib.sweep import Protein


def _protein_config():
    return {
        "method": "Protein",
        "metric": "elo",
        "metric_distribution": "linear",
        "goal": "maximize",
        "downsample": 1,
        "use_gpu": False,
        "prune_pareto": True,
        "max_suggestion_cost": 100,
        "early_stop_quantile": 0.3,
        "gpus": 1,
        "max_runs": 4,
        "train": {
            "total_timesteps": {
                "distribution": "log_normal",
                "min": 1,
                "max": 100,
                "scale": "auto",
            },
        },
    }


def test_protein_refreshes_observations_by_run_id():
    protein = Protein(_protein_config(), use_gpu=False, gp_training_iter=1)
    hypers = {"train": {"total_timesteps": 10}}

    protein.observe(hypers, 10.0, 1.0, run_id="run_a")
    protein.observe(hypers, 20.0, 1.0, run_id="run_b")

    assert len(protein.success_observations) == 2
    assert protein.refresh_observations_by_run_id({"run_a": 30.0}) == 1

    by_run = {obs["run_id"]: obs["output"] for obs in protein.success_observations}
    assert by_run["run_a"] == 30.0
    assert by_run["run_b"] == 20.0


def test_batch_ratings_anchor_zero_and_symmetric():
    players = [
        {"id": league.ANCHOR_ID},
        {"id": "policy_a"},
        {"id": "policy_b"},
    ]
    matches = [
        {"a": league.ANCHOR_ID, "b": "policy_a", "games": 100, "a_score_rate": 0.25},
        {"a": league.ANCHOR_ID, "b": "policy_b", "games": 100, "a_score_rate": 0.75},
    ]

    ratings = league.recompute_ratings(players, matches)

    assert ratings[league.ANCHOR_ID] == 0.0
    assert ratings["policy_a"] > 0
    assert ratings["policy_b"] < 0
    assert ratings["policy_a"] == pytest.approx(-ratings["policy_b"], abs=1e-4)


def test_league_config_allows_architecture_sweep_keys():
    args = {
        "train": {"gpus": 1},
        "sweep": {
            "league": True,
            "metric": "score",
            "downsample": 5,
            "policy": {
                "hidden_size": {
                    "distribution": "uniform_pow2",
                    "min": 32,
                    "max": 1024,
                    "scale": "auto",
                },
                "num_layers": {
                    "distribution": "uniform",
                    "min": 1,
                    "max": 8,
                    "scale": "auto",
                },
            },
        },
        "selfplay": {"enabled": 1},
        "vec": {},
        "env": {"num_agents": 2, "num_bots": 0},
        "policy": {"hidden_size": 256, "num_layers": 3},
    }

    league.validate_and_force_config("robocode", args)

    assert "hidden_size" in args["sweep"]["policy"]
    assert "num_layers" in args["sweep"]["policy"]



def test_league_trials_use_historical_selfplay_only():
    args = {
        "train": {"gpus": 8},
        "sweep": {
            "league": True,
            "metric": "score",
            "downsample": 5,
        },
        "selfplay": {
            "enabled": 1,
            "snapshot_interval": 200_000_000,
            "opp_timeout_steps": 100_000_000,
        },
        "vec": {
            "num_frozen_banks": 2,
            "frozen_bank_pct": 0.125,
        },
        "env": {
            "num_agents": 2,
            "num_bots": 0,
        },
        "policy": {
            "hidden_size": 256,
            "num_layers": 3,
        },
    }

    league.validate_and_force_config("robocode", args)
    league.configure_trial_args(args)

    assert args["train"]["gpus"] == 1
    assert args["sweep"]["metric"] == "elo"
    assert args["selfplay"]["enabled"] == 1
    assert args["selfplay"]["snapshot_interval"] == 200_000_000
    assert args["selfplay"]["opp_timeout_steps"] == 100_000_000
    assert args["vec"]["num_frozen_banks"] == 2
    assert args["vec"]["frozen_bank_pct"] == 0.125
    assert args["vec"]["frozen_bank_hidden_size"] == 256
    assert args["vec"]["frozen_bank_num_layers"] == 3



def test_league_match_reward_conditioning_uses_player_hypers():
    state = {
        "players": [
            {
                "id": "a",
                "kind": "policy",
                "checkpoint_path": "a.bin",
                "elo": 10.0,
                "hypers": {
                    "env": {
                        "reward_melee_damage_inflicted": 0.001,
                        "reward_damage_taken": -0.002,
                        "reward_range_damage_inflicted": 0.003,
                    }
                },
            },
            {
                "id": "b",
                "kind": "policy",
                "checkpoint_path": "b.bin",
                "elo": 20.0,
                "hypers": {
                    "env": {
                        "reward_melee_damage_inflicted": 0.004,
                        "reward_damage_taken": -0.005,
                        "reward_range_damage_inflicted": 0.006,
                    }
                },
            },
        ]
    }

    player_a, player_b = league.opponent_pool(state)
    match_args = {"env": {}}
    league.apply_match_reward_conditioning(match_args, player_a, player_b)

    env = match_args["env"]
    assert env["reward_melee_damage_inflicted_slot_0"] == 0.001
    assert env["reward_damage_taken_slot_0"] == -0.002
    assert env["reward_range_damage_inflicted_slot_0"] == 0.003
    assert env["reward_melee_damage_inflicted_slot_1"] == 0.004
    assert env["reward_damage_taken_slot_1"] == -0.005
    assert env["reward_range_damage_inflicted_slot_1"] == 0.006



def test_league_match_policy_arch_uses_player_hypers():
    match_args = {
        "policy": {"hidden_size": 256, "num_layers": 3},
        "vec": {},
    }
    player_a = {
        "id": "a",
        "arch": {"hidden_size": 128, "num_layers": 2},
        "hypers": {"policy": {"hidden_size": 64, "num_layers": 1}},
    }
    player_b = {
        "id": "b",
        "hypers": {"policy": {"hidden_size": 512, "num_layers": 5}},
    }

    league.apply_match_policy_arch(match_args, player_a, player_b)

    assert match_args["policy"]["hidden_size"] == 128
    assert match_args["policy"]["num_layers"] == 2
    assert match_args["enemy_hidden_size"] == 512
    assert match_args["enemy_num_layers"] == 5
    assert match_args["vec"]["frozen_bank_hidden_size"] == 512
    assert match_args["vec"]["frozen_bank_num_layers"] == 5
