#!/bin/bash

# Usage: ./build_env.sh enduro_clone [local|web]

ENV=$1
MODE=${2:-local}

SRC_DIR="pufferlib/environments/ocean/$ENV"
OUTPUT_DIR="."
WEB_OUTPUT_DIR="build_web/$ENV"
RESOURCES_DIR="resources"

# Create build output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$WEB_OUTPUT_DIR"

if [ "$MODE" = "local" ]; then
    echo "Building $ENV for local testing..."
    if [[ "$ENV" == "pong" ]]; then
        gcc -g -O2 -Wall \
            -I./raylib/include \
            -I./pufferlib \
            "$SRC_DIR/$ENV.c" -o "$OUTPUT_DIR/$ENV" \
            ./raylib/lib/libraylib.a -lm -lpthread \
            -fsanitize=address,undefined,bounds,pointer-overflow,leak
    elif [[ "$ENV" == "enduro_clone" ]]; then
        gcc -g -O2 -Wall \
            -I/usr/include/SDL2 \
            -D_REENTRANT \
            "$SRC_DIR/$ENV.c" -o "$OUTPUT_DIR/$ENV" \
            -L/usr/lib/x86_64-linux-gnu -lSDL2 -lSDL2_image \
            -lm -lpthread \
            -fsanitize=address,undefined,bounds,pointer-overflow,leak
    fi
    echo "Built to: $OUTPUT_DIR/$ENV"
elif [ "$MODE" = "web" ]; then
    echo "Building $ENV for web deployment..."

    PRELOAD=""
    if [ -d "$RESOURCES_DIR" ]; then
        PRELOAD="--preload-file $RESOURCES_DIR@resources/"
    fi

    echo "Preloading resources from $RESOURCES_DIR"

    if [[ "$ENV" == "pong" ]]; then
        emcc \
            -o "$WEB_OUTPUT_DIR/game.html" \
            "$SRC_DIR/$ENV.c" \
            -Os \
            -Wall \
            ./raylib_wasm/lib/libraylib.a \
            -I./raylib_wasm/include \
            -L. \
            -L./raylib_wasm/lib \
            -s ASSERTIONS=2 \
            -gsource-map \
            -s USE_GLFW=3 \
            -s USE_WEBGL2=1 \
            -s ASYNCIFY \
            -s FILESYSTEM \
            -s FORCE_FILESYSTEM=1 \
            --shell-file ./minshell.html \
            -DPLATFORM_WEB \
            -DGRAPHICS_API_OPENGL_ES3 $PRELOAD
    elif [[ "$ENV" == "enduro_clone" ]]; then
        echo "Web build for SDL2 not configured, as SDL2 is not typically used for web via emscripten."
    fi
    echo "Web build completed: $WEB_OUTPUT_DIR/game.html"
else
    echo "Invalid mode specified. Use 'local' or 'web'."
    exit 1
fi
