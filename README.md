# Drone Pick-and-Place Environment

A reinforcement learning environment for training drones to perform package delivery tasks, built for the REVEL x LycheeAI Isaac Sim Hackathon.
# Python version: 3.12.7 (runs on multiple python versions)

Train a drone to:
1. **Take off** and locate a package (DHL package)
2. **Fly to** and **pick up** the package
3. **Carry** the package to a goal location
4. **Drop** the package at the goal location

### Installation

1. **Install dependencies**
```bash
pip install -e .
```
2. **Build the environment**
```bash
python setup.py build_ext --inplace --force
```
That's it! You're ready to train or evaluate.

## 📊 Usage

### Train a new model
```bash
puffer train puffer_drone_pickplace --wandb
```
### Evaluate a trained model
```bash
puffer eval puffer_drone_pickplace --load-model-path /path/to/your/model.pt
```
Example with existing model:
```bash
puffer eval puffer_drone_pickplace --load-model-path /puffertank/hackathon/isaac-sim-hackathon/pufferlib/experiments/puffer_drone_pickplace_zz8xivzz/model_puffer_drone_pickplace_001200.pt
```
### Human play with WASD and Space to pick up (jank, possibly env bugs, possibly rendering bugs, need to diagnose)
scripts/build_ocean.sh drone_pickplace 
./drone_pickplace

### Export to Isaac Sim (ONNX format)
```bash
python export_onnx.py path/to/checkpoint.pt --output-dir onnx_export
```
Example:
```bash
python export_onnx.py experiments/puffer_drone_pickplace_zz8xivzz/model_puffer_drone_pickplace_001200.pt
```
This creates:
- `onnx_export/drone_policy.onnx` - The ONNX model for Isaac Sim
- `onnx_export/model_config.json` - Configuration file with observation/action specs
- `onnx_export/isaac_sim_policy.py` - Python wrapper for easy integration

## 🎮 Environment Details
### Observation Space (45 continuous values)
- **Drone state** (14): position, velocity, quaternion, angular velocity, gripper state
- **Object states** (21): positions, velocities, and status for packages
- **Target zones** (8): positions and occupancy status
- **Task info** (2): time remaining, task progress
### Action Space (10 discrete actions)
- `0-3`: Move Forward/Backward/Left/Right
- `4-5`: Move Up/Down
- `6-7`: Rotate Left/Right
- `8-9`: Gripper Open/Close
### Rewards
- **Sparse rewards** reward for distance-based progress towards object, then towards drop zone
- **+1.0** for successful package pickup
- **+1.0** for successful package delivery
- **time penalty** for time wasting

## Original PufferLib README

![figure](https://pufferai.github.io/source/resource/header.png)

[![PyPI version](https://badge.fury.io/py/pufferlib.svg)](https://badge.fury.io/py/pufferlib)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pufferlib)
![Github Actions](https://github.com/PufferAI/PufferLib/actions/workflows/install.yml/badge.svg)
[![](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez5341)](https://twitter.com/jsuarez5341)

PufferLib is the reinforcement learning library I wish existed during my PhD. It started as a compatibility layer to make working with complex environments a breeze. Now, it's a high-performance toolkit for research and industry with optimized parallel simulation, environments that run and train at 1M+ steps/second, and tons of quality of life improvements for practitioners. All our tools are free and open source. We also offer priority service for companies, startups, and labs!

![Trailer](https://github.com/PufferAI/puffer.ai/blob/main/docs/assets/puffer_2.gif?raw=true)

All of our documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support -- post here before opening issues. We're always looking for new contributors, too!

## Star to puff up the project!

<a href="https://star-history.com/#pufferai/pufferlib&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
 </picture>
</a>
