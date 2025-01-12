# README
Please star PufferLib on GitHub: it really makes a difference!
https://github.com/pufferai/pufferlib

by Daniel Addis, 2024
https://github.com/xinpw8

## Puffer Ocean Enduro
This project contains a performant reinforcement-learning environment inspired by the classic Atari 2600 game Enduro. It uses C, Cython, and Python to provide an interactive RL environment, trainable with PufferLib.

## Building & Setup
1. Install dependencies. All commands should be run from the `PufferLib` top directory.
```sh
pip install -e .'[cleanrl]'
```
Instructions on https://puffer.ai/docs.html

2. Compilation is run when pufferlib is pip installed. After making changes to `cy_enduro.pyx`, `enduro.c`, or `enduro.h`, recompile with:
    ```sh
    python setup.py build_ext --inplace
    ```

3. To locally compile the C environment for testing (without Cython), run:
    ```sh
    scripts/build_ocean.sh enduro local
    ```
    This builds a runnable `enduro` module in the PufferLib top directory, which you can run with:
    ```sh
    ./enduro
    ```
    Hold Shift to take control from the agent.

## Training
To train using the demo script with wandb logs, run:
```sh
python demo.py --env puffer_enduro --mode train --track
```
Model files are saved at intervals specified in `config/ocean/enduro.ini` to the `experiments/` directory.

## Evaluation
To evaluate a local checkpoint, run:
```sh
python demo.py --env puffer_enduro --mode eval --eval-model-path your_model.pt
```
