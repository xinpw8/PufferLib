# TrashPickup Environment

The `TrashPickup` environment is a simple yet powerful multi-agent environment for reinforcement learning (RL) research, especially tailored for investigating agent coordination and cooperation strategies. This environment allows for flexible configuration, scaling, and benchmarking in multi-agent setups, making it ideal for testing scalable algorithms and exploring emergent behaviors in cooperative settings.

![Trash Pickup](./trash_pickup_vid.gif)

## Overview

In the `TrashPickup` environment, multiple agents are placed in a grid world containing randomly scattered trash items and bins. Agents are rewarded for picking up trash and depositing it into bins. To maximize their rewards, agents must balance efficient movement, avoid redundant actions, and, ideally, coordinate with each other to minimize time and effort.

Key challenges and objectives in this environment include:
- **Coordination:** Ensuring multiple agents don’t waste time by attempting to pick up the same trash item.
- **Teamwork:** Agents can maximize their rewards by strategically pushing bins closer to trash piles or other agents, minimizing movement costs.
- **Efficient Planning:** Agents are encouraged to "self-assign" sections of the map, spreading out across the grid to efficiently cover all trash areas.

## Features

### 1. **Discrete Action and Observation Spaces**
   - **Action Space:** Each agent has a discrete set of actions: `UP`, `DOWN`, `LEFT`, `RIGHT`.
   - **Observation Space:** Each agent observes its position, whether it’s carrying trash, and the entire grid state (i.e., locations of other agents, trash, and bins).
   
### 2. **Multi-Agent Coordination**
   - This environment is ideal for studying cooperative behavior and emergent multi-agent strategies.
   - Researchers can examine whether agents learn to spread out efficiently, avoid conflict over resources, and cooperate to accomplish shared goals.

### 3. **Configurable Environment Size and Complexity**
   - The environment’s parameters (grid size, number of agents, number of trash items, and bins) are easily adjustable, allowing users to scale up complexity for more demanding multi-agent experiments.
   - This scalability makes it a useful benchmark for testing multi-agent reinforcement learning algorithms that need to perform efficiently at larger scales.
   -  Configurable options are:
        - grid_size: Size of the grid world (e.g., 10 for a 10x10 grid).
        - num_agents: Number of agents in the environment.
        - num_trash: Number of trash items scattered in the grid.
        - num_bins: Number of trash bins in the grid.
        - max_steps: Maximum number of steps in an episode before it ends.

### 4. **Target Audience**
   - **RL Researchers and Developers:** Ideal for those focused on multi-agent reinforcement learning, cooperative tasks, and emergent behavior.
   - **Educators and Students:** Suitable for learning about multi-agent systems in a manageable and understandable environment.
   - **Algorithm Developers:** A lightweight environment that supports rapid iteration and testing for new multi-agent algorithms.

### 5. **Fast and Lightweight**
   - `TrashPickup` is designed to be computationally efficient, allowing fast training and testing even on large-scale configurations.

## Example Behaviors to Investigate

The environment's simplicity makes it a great playground for analyzing complex behaviors, such as:
- **Coordination vs. Greed:** Do agents learn to coordinate or act in a greedy manner, competing over the same resources?
- **Area Division:** Can agents learn to implicitly "self-assign" sections of the grid, focusing on different regions to maximize coverage and avoid duplication?
- **Push Mechanics for Efficiency:** Do agents learn to use the pushing mechanic effectively, moving bins closer to trash for greater efficiency?

## Cython Implementation

Included in this repository is a Cython version of `TrashPickup` (files: `trash_pickup.c`, `trash_pickup.h`, `cy_trash_pickup.pyx`, and `cy_environment.py`). Although slightly slower than the native Python implementation, this version is provided as a reference example for developers interested in implementing multi-agent environments with Python-C bindings via Cython.

## License

This environment is open-source and available for educational and research purposes. Contributions are welcome.