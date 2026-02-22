import pytest

from pufferlib.ocean.poke_battle.poke_battle import PokeBattle, SPECIES_NAMES


NAME_TO_ID = {name: sid for sid, name in SPECIES_NAMES.items() if sid > 0}


def _team_with_target(target_species_id):
    team = [target_species_id]
    for sid in range(1, 150):
        if sid == target_species_id:
            continue
        team.append(sid)
        if len(team) == 6:
            break
    return team


def test_all_modeled_species_have_no_duplicate_non_none_moves():
    env = PokeBattle(
        num_envs=1,
        selfplay=1,
        auto_reset=0,
        seed=19,
    )
    try:
        for species_id in range(1, 150):
            team = _team_with_target(species_id)
            env.put_state(0, p1_team=team, p2_team=team)
            env.reset(seed=1000 + species_id)
            state = env.get_state(0)
            moves = [m["name"] for m in state["p1"]["team"][0]["moves"] if m["name"] != "None"]
            assert len(moves) == len(set(moves)), f"duplicate move on species id {species_id}: {moves}"
    finally:
        env.close()


@pytest.mark.parametrize(
    ("species_name", "forbidden_move"),
    [
        ("Weedle", "Tackle"),
        ("Pidgey", "Tackle"),
        ("Pidgeotto", "Tackle"),
        ("Pidgeot", "Tackle"),
        ("Vulpix", "Hypnosis"),
        ("Meowth", "Hypnosis"),
        ("Psyduck", "Hypnosis"),
        ("Ponyta", "Hypnosis"),
        ("Exeggcute", "Mega Drain"),
        ("Rhyhorn", "Blizzard"),
        ("Mr. Mime", "Hypnosis"),
    ],
)
def test_known_tradeback_illegal_moves_are_not_in_hardcoded_sets(species_name, forbidden_move):
    species_id = NAME_TO_ID[species_name]
    team = _team_with_target(species_id)
    env = PokeBattle(
        num_envs=1,
        selfplay=1,
        auto_reset=0,
        seed=29,
        p1_team=team,
        p2_team=team,
    )
    try:
        env.reset(seed=29)
        state = env.get_state(0)
        moves = [m["name"] for m in state["p1"]["team"][0]["moves"]]
        assert forbidden_move not in moves
    finally:
        env.close()
