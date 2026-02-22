import numpy as np

from pufferlib.ocean.poke_battle.poke_battle import LEGAL_SPECIES_IDS, PokeBattle


def _play_one_episode(env, seed, rng):
    env.reset(seed=seed)
    latest_log = None
    for _ in range(700):
        action = np.array([int(rng.integers(0, 10))], dtype=np.int32)
        _, _, terminals, _, info = env.step(action)
        if info:
            latest_log = info[-1]
        if bool(terminals[0]):
            return True, latest_log
    return False, latest_log


def test_legal_species_pool_is_comprehensive_for_modeled_ou():
    assert LEGAL_SPECIES_IDS == tuple(range(1, 150))
    assert len(set(LEGAL_SPECIES_IDS)) == len(LEGAL_SPECIES_IDS)


def test_adaptive_team_builder_covers_all_legal_species():
    env = PokeBattle(
        num_envs=1,
        selfplay=0,
        auto_reset=0,
        seed=5,
        team_builder_mode=1,
    )
    seen = set()
    try:
        # Adaptive builder enforces one rotating anchor species per episode.
        for ep in range(170):
            env.reset(seed=1000 + ep)
            state = env.get_state(0)
            for mon in state["p1"]["team"]:
                seen.add(int(mon["species_id"]))
        assert seen == set(LEGAL_SPECIES_IDS)
    finally:
        env.close()


def test_team_builder_metrics_update_after_episode():
    env = PokeBattle(
        num_envs=1,
        selfplay=0,
        auto_reset=0,
        seed=13,
        bot_mode=0,
        team_builder_mode=1,
    )
    rng = np.random.default_rng(7)
    try:
        done, _ = _play_one_episode(env, seed=123, rng=rng)
        assert done
        state = env.get_state(0)
        assert 0.0 <= state["team_builder_recent_winrate"] <= 1.0
        assert state["team_builder_unique_species_seen"] > 0
        assert state["team_builder_pool_coverage"] > 0.0
    finally:
        env.close()


def test_switching_team_builder_mode_resets_search_state():
    env = PokeBattle(
        num_envs=1,
        selfplay=0,
        auto_reset=0,
        seed=21,
        team_builder_mode=1,
    )
    rng = np.random.default_rng(11)
    try:
        done, _ = _play_one_episode(env, seed=222, rng=rng)
        assert done
        state_before = env.get_state(0)
        assert state_before["team_builder_unique_species_seen"] > 0

        env.put_state(0, team_builder_mode=0)
        env.put_state(0, team_builder_mode=1)
        state_after = env.get_state(0)
        assert state_after["team_builder_mode"] == 1
        assert state_after["team_builder_unique_species_seen"] == 0
        assert abs(state_after["team_builder_recent_winrate"] - 0.5) < 1e-6
    finally:
        env.close()


def test_team_builder_best_team_metrics_are_logged():
    env = PokeBattle(
        num_envs=1,
        selfplay=0,
        auto_reset=0,
        seed=29,
        bot_mode=0,
        team_builder_mode=1,
        log_interval=1,
    )
    rng = np.random.default_rng(19)
    latest_log = None
    try:
        for ep in range(6):
            done, episode_log = _play_one_episode(env, seed=900 + ep, rng=rng)
            assert done
            if episode_log:
                latest_log = episode_log

        assert latest_log is not None
        assert "team_builder_best_team_mean_wr" in latest_log
        assert "team_builder_best_team_mean_pick_rate" in latest_log
        assert any(key.startswith("pick_") for key in latest_log)

        picked = []
        for slot in range(1, 7):
            sid_key = f"team_builder_best_species_{slot}"
            pick_key = f"team_builder_best_species_{slot}_pick_rate"
            wr_key = f"team_builder_best_species_{slot}_wr"
            score_key = f"team_builder_best_species_{slot}_score"

            assert sid_key in latest_log
            assert pick_key in latest_log
            assert wr_key in latest_log
            assert score_key in latest_log

            sid = int(round(float(latest_log[sid_key])))
            assert sid in LEGAL_SPECIES_IDS
            picked.append(sid)

            assert 0.0 <= float(latest_log[pick_key]) <= 1.0
            assert 0.0 <= float(latest_log[wr_key]) <= 1.0

        assert len(set(picked)) == 6
    finally:
        env.close()
