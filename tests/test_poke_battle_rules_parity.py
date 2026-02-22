import numpy as np
import pytest

from pufferlib.ocean.poke_battle.poke_battle import (
    PokeBattle,
    RBY_OU_RULESET,
    SPECIES_ALAKAZAM,
    SPECIES_CHANSEY,
    SPECIES_EXEGGUTOR,
    SPECIES_SNORLAX,
    SPECIES_STARMIE,
    SPECIES_TAUROS,
)


DEFAULT_UNIQUE_TEAM = [
    SPECIES_TAUROS,
    SPECIES_CHANSEY,
    SPECIES_SNORLAX,
    SPECIES_ALAKAZAM,
    SPECIES_EXEGGUTOR,
    SPECIES_STARMIE,
]


def make_env(p1_team, p2_team, **kwargs):
    env = PokeBattle(
        num_envs=1,
        selfplay=1,
        auto_reset=0,
        seed=7,
        p1_team=p1_team,
        p2_team=p2_team,
        **kwargs,
    )
    env.reset(seed=7)
    return env


def step(env, p1_action, p2_action):
    _, _, terminals, _, _ = env.step(np.array([p1_action, p2_action], dtype=np.int32))
    return bool(terminals[0]), env.get_state(0)


def test_sleep_clause_blocks_second_opponent_inflicted_sleep():
    p1 = [SPECIES_EXEGGUTOR, SPECIES_TAUROS, SPECIES_CHANSEY, SPECIES_SNORLAX, SPECIES_ALAKAZAM, SPECIES_STARMIE]
    p2 = [SPECIES_CHANSEY, SPECIES_TAUROS, SPECIES_SNORLAX, SPECIES_ALAKAZAM, SPECIES_EXEGGUTOR, SPECIES_STARMIE]
    env = make_env(p1, p2, force_accuracy=1)
    try:
        _, state = step(env, 1, 2)  # Sleep Powder into Chansey
        assert state["p2"]["team"][0]["status_name"] == "slp"
        assert state["p2"]["team"][0]["sleep_source_side"] == 0

        _, state = step(env, 0, 5)  # switch to Tauros
        assert state["p2"]["active_idx"] == 1

        _, state = step(env, 1, 0)  # Sleep Powder again should be blocked
        assert state["p2"]["team"][1]["status_name"] != "slp"
        assert state["p2"]["team"][0]["status_name"] == "slp"
    finally:
        env.close()


def test_sleep_clause_does_not_block_after_self_inflicted_rest_sleep():
    p1 = [SPECIES_EXEGGUTOR, SPECIES_TAUROS, SPECIES_CHANSEY, SPECIES_SNORLAX, SPECIES_ALAKAZAM, SPECIES_STARMIE]
    p2 = [SPECIES_SNORLAX, SPECIES_TAUROS, SPECIES_CHANSEY, SPECIES_ALAKAZAM, SPECIES_EXEGGUTOR, SPECIES_STARMIE]
    env = make_env(p1, p2, force_accuracy=1)
    try:
        _, state = step(env, 0, 3)  # Psychic damages Snorlax, then Snorlax Rests
        assert state["p2"]["team"][0]["status_name"] == "slp"
        assert state["p2"]["team"][0]["sleep_source_side"] == 1

        _, state = step(env, 0, 5)  # switch sleeping Snorlax out
        assert state["p2"]["active_idx"] == 1

        _, state = step(env, 1, 0)  # Sleep Powder should still work on Tauros
        assert state["p2"]["team"][1]["status_name"] == "slp"
        assert state["p2"]["team"][1]["sleep_source_side"] == 0
    finally:
        env.close()


def test_freeze_clause_blocks_second_opponent_inflicted_freeze():
    p1 = [SPECIES_CHANSEY, SPECIES_TAUROS, SPECIES_SNORLAX, SPECIES_ALAKAZAM, SPECIES_EXEGGUTOR, SPECIES_STARMIE]
    p2 = [SPECIES_CHANSEY, SPECIES_TAUROS, SPECIES_SNORLAX, SPECIES_ALAKAZAM, SPECIES_EXEGGUTOR, SPECIES_STARMIE]
    env = make_env(p1, p2, force_accuracy=1, force_secondary=1)
    try:
        _, state = step(env, 1, 2)  # Ice Beam freezes Chansey
        assert state["p2"]["team"][0]["status_name"] == "frz"
        assert state["p2"]["team"][0]["freeze_source_side"] == 0

        _, state = step(env, 2, 5)  # switch to Tauros
        assert state["p2"]["active_idx"] == 1

        _, state = step(env, 1, 0)  # Second freeze attempt should be blocked
        assert state["p2"]["team"][1]["status_name"] != "frz"
        assert state["p2"]["team"][0]["status_name"] == "frz"
    finally:
        env.close()


def test_freeze_clause_allows_new_freeze_after_first_frozen_target_faints():
    p1 = [SPECIES_CHANSEY, SPECIES_TAUROS, SPECIES_SNORLAX, SPECIES_ALAKAZAM, SPECIES_EXEGGUTOR, SPECIES_STARMIE]
    p2 = [SPECIES_ALAKAZAM, SPECIES_TAUROS, SPECIES_CHANSEY, SPECIES_SNORLAX, SPECIES_EXEGGUTOR, SPECIES_STARMIE]
    env = make_env(p1, p2, force_accuracy=1, force_secondary=1)
    try:
        _, state = step(env, 1, 3)  # Freeze Alakazam
        assert state["p2"]["team"][0]["status_name"] == "frz"

        for _ in range(10):
            _, state = step(env, 3, 0)  # Seismic Toss until Alakazam faints
            if state["p2"]["team"][0]["is_alive"] == 0:
                break

        assert state["p2"]["team"][0]["is_alive"] == 0

        if state["mode"] == 2:
            _, state = step(env, 0, 5)  # forced switch to Tauros
        assert state["p2"]["active_idx"] == 1

        _, state = step(env, 1, 0)  # Ice Beam can freeze again now
        assert state["p2"]["team"][1]["status_name"] == "frz"
        assert state["p2"]["team"][1]["freeze_source_side"] == 0
    finally:
        env.close()


def test_species_clause_rejects_duplicate_fixed_team():
    bad_team = [
        SPECIES_TAUROS,
        SPECIES_TAUROS,
        SPECIES_CHANSEY,
        SPECIES_SNORLAX,
        SPECIES_ALAKAZAM,
        SPECIES_EXEGGUTOR,
    ]
    with pytest.raises(ValueError, match="Species Clause"):
        env = PokeBattle(
            num_envs=1,
            selfplay=1,
            auto_reset=0,
            seed=1,
            p1_team=bad_team,
            p2_team=DEFAULT_UNIQUE_TEAM,
        )
        env.close()


def test_species_clause_accepts_unique_team_and_reports_ruleset():
    env = make_env(DEFAULT_UNIQUE_TEAM, DEFAULT_UNIQUE_TEAM)
    try:
        state = env.get_state(0)
        assert state["ruleset"] == RBY_OU_RULESET
        assert state["species_clause"] == 1
        assert state["sleep_clause_mod"] == 1
        assert state["freeze_clause_mod"] == 1
        assert state["ohko_clause"] == 1
        assert state["evasion_moves_clause"] == 1
        assert state["endless_battle_clause"] == 1
    finally:
        env.close()


def test_endless_battle_clause_terminates_stale_battle():
    p1 = [SPECIES_EXEGGUTOR, SPECIES_TAUROS, SPECIES_CHANSEY, SPECIES_SNORLAX, SPECIES_ALAKAZAM, SPECIES_STARMIE]
    p2 = [SPECIES_EXEGGUTOR, SPECIES_TAUROS, SPECIES_CHANSEY, SPECIES_SNORLAX, SPECIES_ALAKAZAM, SPECIES_STARMIE]
    env = make_env(p1, p2, enforce_endless_clause=1, force_accuracy=1)
    try:
        done = False
        state = env.get_state(0)
        for _ in range(50):
            done, state = step(env, 2, 2)  # Stun Spore into already paralyzed targets
            if done:
                break
        assert done
        assert state["stale_turns"] >= 32
    finally:
        env.close()


def test_endless_battle_clause_can_be_disabled_for_debugging():
    p1 = [SPECIES_EXEGGUTOR, SPECIES_TAUROS, SPECIES_CHANSEY, SPECIES_SNORLAX, SPECIES_ALAKAZAM, SPECIES_STARMIE]
    p2 = [SPECIES_EXEGGUTOR, SPECIES_TAUROS, SPECIES_CHANSEY, SPECIES_SNORLAX, SPECIES_ALAKAZAM, SPECIES_STARMIE]
    env = make_env(p1, p2, enforce_endless_clause=0, force_accuracy=1)
    try:
        done = False
        for _ in range(50):
            done, _ = step(env, 2, 2)
            if done:
                break
        assert not done
    finally:
        env.close()
