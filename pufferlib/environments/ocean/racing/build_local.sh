#!/bin/bash

clang -Wall -Wuninitialized -Wmisleading-indentation -fsanitize=address \
    -ferror-limit=3 -g -o racinggame racing.c \
    -I./raylib-5.0_linux_amd64/include/ \
    -L./raylib-5.0_linux_amd64/lib/ -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 \
    -DPLATFORM_DESKTOP
