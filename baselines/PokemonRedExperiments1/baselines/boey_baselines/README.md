# BOEY's Baselines
This baseline can beat Gym 2 within 150m steps, take down Route 24/25 & Sea Cottage and backtrack to continue to Route 5 in 300m steps!

*The training is still ongoing, and the agent is on its way to beat Gym 3. More updates will be coming soon!

## Customized Network Architecture
<img src="../../assets/Pokemon-Network-Architecture-Boey.png?raw=true">

(Right click and open image in a new tab to see the full size)

- Modular features processing. (See [OpenAI Five](https://openai.com/blog/openai-five/) for more details)
- This implementation allow the agent to access to all the required information and prevented the [Curse of dimensionaility](https://en.wikipedia.org/wiki/Curse_of_dimensionality) issue.
- Without this network achitecture, the agent fails to learn anything when given the same observation.

## Observation space
- 6 Party pokemons' HP, level, status, moves, PP, and types
- 6 Opponent's pokemon's (including wild) HP, level, status, moves, PP, and types
- 20 items in the bag
- Last 10 events
- 3 Frames stacked in channel dimension
- Scalar / One-hot states (see `get_all_raw_obs` in `boey_baselines/red_gym_env.py`)

## Rewards
- 1 for event gained
- 0.5 for each level
- 5 for badge
- 2 for each new PokemonCenter visited (set new respawn point, see `get_visited_pokecenter_reward` in `boey_baselines/red_gym_env.py`)
- 0.005 for each new coordinate visited (visited will be reset on every new event gained, <b>this is crucial for backtracking</b>)
- 0.1 for maximum opponent pokemon level past 5
- -0.1 for each death
- 1 for each HM obtained
- 2 for each HM move in the party

## Env improvements
- Env reset when less than 1.0 reward is gained in 10,240 steps
- `restricted_start_menu` option available, when `start` button is enabled, the agent can choose the options of `Pokemon` and `Items`. (See `get_menu_restricted_action` in `boey_baselines/red_gym_env.py`)
- `seen_coords` is emptied when new event is gained, previous rewards is stored.
