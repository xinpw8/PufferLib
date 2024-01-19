![figure](https://pufferai.github.io/source/resource/header.png)

[![PyPI version](https://badge.fury.io/py/pufferlib.svg)](https://badge.fury.io/py/pufferlib)
[![](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez5341)](https://twitter.com/jsuarez5341)

You have an environment, a PyTorch model, and a reinforcement learning framework that are designed to work together but don’t. PufferLib is a wrapper layer that makes RL on complex game environments as simple as RL on Atari. You write a native PyTorch network and a short binding for your environment; PufferLib takes care of the rest.

All of our [Documentation](https://pufferai.github.io "PufferLib Documentation") is hosted by github.io. @jsuarez5341 on [Discord](https://discord.gg/spT4huaGYV) for support -- post here before opening issues. I am also looking for contributors interested in adding bindings for other environments and RL frameworks.

--------------------------------------------------------------------------------------------------------
xinpw8 Unofficial Pokegym Documentation below
-Train RL agents to play Pokemon Red-
original author: https://github.com/PWhiddy/PokemonRedExperiments
ported to pufferlib and pokegym by @jsuarez5341

mv PufferLib pufferlib

*MUST HAVE TEXT FILE running_experiment.txt in pufferlib/experiments for logging/folder organization to work!*
*MUST HAVE pokemon_red.gb ROM FILE in pufferlib/*

Experiments And Files From Training Save To:

Directory: pufferlib/experiments/{session_uuid8}
Saved Models: pufferlib/experiments/{session_uuid8}/model_stepcount.pt
All Party Pokemon/Moves/Levels/Incidence From All Environments: pufferlib/experiments/{session_uuid8}/pokes_log.txt
CSV For Making Cool Postprocessed Heatmaps: pufferlib/experiments/{session_uuid8}/steps_map.csv
Name of Current Experiment (File Must Exist and Be Editable): pufferlib/experiments/{session_uuid8}/running_experiment.txt
Environment Folders: pufferlib/experiments/{session_uuid8}/sessions


AI starts from a state post-Oak-parcel-fetch-quest. It goes to Cerulean City and beats Misty, gets Bill, beat SS Anne, gets Cut, but doesn't teach/teaches w/ Bulbasaur but doesn't use.
Clone my repositories:

github.com/xinpw8/PufferLib.git -b 0.6 
github.com/xinpw8/pokegym.git -b damnitleankeididitforyou

Clone both into a main folder, [mv PufferLib pufferlib], then, in pufferlib folder, 

pip install -e ".[cleanrl,pokemon_red]"
 then cd .. to main folder again and 
pip install -e pokegym

pip list should show both pointing to your local folders always.
 then you need to put the pokemonred.gb rom file into pufferlib folder and that's it. Environment file to edit is in pokegym/pokegym. To run, cd pufferlib, in demo.py, ctrl+f xinpw8 and change to your wandb account name, then 

./run.sh

 Default max global steps (and most config settings) is in config.yaml, set to 100m.

 Current features are:
 -Save/load states (makes it progress rapidly through game)
 -LSTM (makes it able to actually do stuff)
 -Explore (buffered view around agent of PyBoy emulator window; helps exploration)
 -Explore NPCs function does Bill

 Currently getting Cut but AI is not teaching/using.