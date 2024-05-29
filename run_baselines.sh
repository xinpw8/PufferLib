#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <group>"
    exit 1
fi

GROUP=$1
COMMON="python demo.py --mode train --baseline --env"

case $GROUP in
    ocean)
        $COMMON squared
        $COMMON password
        $COMMON stochastic
        $COMMON memory
        $COMMON multiagent
        $COMMON spaces
        $COMMON bandit
        ;;
    procgen)
        $COMMON bigfish
        $COMMON ninja
        $COMMON miner
        ;;
    atari)
        $COMMON breakout --vec multiprocessing
        $COMMON pong --vec multiprocessing
        $COMMON beam-rider --vec multiprocessing
        $COMMON enduro --vec multiprocessing
        $COMMON qbert --vec multiprocessing
        $COMMON space-invaders --vec multiprocessing
        $COMMON seaquest --vec multiprocessing
        ;;
    pokemon)
        $COMMON pokemon --vec multiprocessing --wandb-entity xinpw8
        ;;
    crafter)
        $COMMON crafter --vec multiprocessing
        ;;
    nethack)
        $COMMON nethack --vec multiprocessing
        ;;
    nmmo)
        $COMMON nmmo --vec multiprocessing
        ;;
    pokedebug)
        $COMMON pokedebug --vec serial --wandb-entity xinpw8
        ;;
    *)
        echo "Invalid group. Please specify 'ocean', 'procgen', 'atari', 'pokemon', 'crafter', 'nethack', 'nmmo', or 'pokedebug'."
        exit 1
        ;;
esac
