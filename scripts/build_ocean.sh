#!/bin/bash

# Usage: ./build_env.sh pong [local|fast|web]

ENV=$1
MODE=${2:-local}
PLATFORM="$(uname -s)"
# Determine repository root based on script location so the script can be
# executed from anywhere.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC_DIR="$ROOT_DIR/pufferlib/ocean/$ENV"
WEB_OUTPUT_DIR="$ROOT_DIR/build_web/$ENV"
RAYLIB_NAME='raylib-5.5_macos'
BOX2D_NAME='box2d-macos-arm64'
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    BOX2D_NAME='box2d-linux-amd64'
fi
if [ "$MODE" = "web" ]; then
    RAYLIB_NAME='raylib-5.5_webassembly'
    BOX2D_NAME='box2d-web'
fi

LINK_ARCHIVES="./$RAYLIB_NAME/lib/libraylib.a"
if [ "$ENV" = "impulse_wars" ]; then
    LINK_ARCHIVES="$LINK_ARCHIVES ./$BOX2D_NAME/libbox2d.a"
fi

# Create build output directory
mkdir -p "$WEB_OUTPUT_DIR"

if [ "$MODE" = "web" ]; then
    echo "Building $ENV for web deployment..."
    emcc \
        -o "$WEB_OUTPUT_DIR/game.html" \
        "$SRC_DIR/$ENV.c" \
        -O3 \
        -Wall \
        "$ROOT_DIR/$RAYLIB_NAME/lib/libraylib.a" \
        -I"$ROOT_DIR/$RAYLIB_NAME/include" \
        -I"$ROOT_DIR/pufferlib" \
        -I"$ROOT_DIR/pufferlib/extensions" \
        -I"$ROOT_DIR/pufferlib/pufferlib/extensions" \
        -L"$ROOT_DIR" \
        -L"$ROOT_DIR/$RAYLIB_NAME/lib" \
        -sASSERTIONS=2 \
        -gsource-map \
        -s USE_GLFW=3 \
        -s USE_WEBGL2=1 \
        -s ASYNCIFY \
        -sFILESYSTEM \
        -s FORCE_FILESYSTEM=1 \
        -s ERROR_ON_UNDEFINED_SYMBOLS=0 \
        -s WARN_ON_UNDEFINED_SYMBOLS=0 \
        --shell-file "$SCRIPT_DIR/minshell.html" \
        -sINITIAL_MEMORY=512MB \
        -sALLOW_MEMORY_GROWTH \
        -sSTACK_SIZE=512KB \
        -DNDEBUG \
        -sTOTAL_STACK=512KB \
        -DPLATFORM_WEB \
        -D_TIME_BITS=64 \
        -DGRAPHICS_API_OPENGL_ES3 \
        --preload-file pufferlib/resources/$1@resources/$1 \
        --preload-file pufferlib/resources/shared@resources/shared 
    echo "Web build completed: $WEB_OUTPUT_DIR/game.html"
    exit 0
fi

echo "${FLAGS[@]}"

if [ "$ENV" = "chess" ]; then
    echo "Building chess binary for local testing…"

    echo "ROOT_DIR: $ROOT_DIR"
    echo "SRC_DIR: $SRC_DIR"
    echo "Attempting to compile with:"
    echo "clang++ -std=c++17 -g -O2 -o chess \\"
    echo "    -I\"$ROOT_DIR/pufferlib/extensions\" \\"
    echo "    \"$SRC_DIR/chess.cpp\" \\"
    echo "    -lpthread -lm"

    clang++ -std=c++17 -g -O2 -o chess \
        -I"$ROOT_DIR/pufferlib/extensions" \
        "$SRC_DIR/chess.cpp" \
        -lpthread -lm

    echo "Binary built: chess"
    exit 0
fi

# Chess demo compilation removed; chess-specific code is now fully isolated inside the ENV=="chess" block above.

FLAGS=(
    -Wall
    -I./$RAYLIB_NAME/include 
    -I./pufferlib/extensions \
    -I./pufferlib/pufferlib/extensions
    "$SRC_DIR/$ENV.c" -o "$ENV"
    $LINK_ARCHIVES
    -lm
    -lpthread
    -ferror-limit=3
    -DPLATFORM_DESKTOP
)

# Provide the legacy include path expected by some test files
OPEN_SPIEL_PREFIX="$SRC_DIR/open_spiel"
if [ ! -d "$OPEN_SPIEL_PREFIX" ]; then
    mkdir -p "$OPEN_SPIEL_PREFIX"
fi
if [ ! -e "$OPEN_SPIEL_PREFIX/abseil-cpp" ]; then
    ln -s ../abseil-cpp "$OPEN_SPIEL_PREFIX/abseil-cpp"
fi

if [ "$PLATFORM" = "Darwin" ]; then
    FLAGS+=(
        -framework Cocoa
        -framework IOKit
        -framework CoreVideo
    )
fi

echo ${FLAGS[@]}

if [ "$MODE" = "local" ]; then
    echo "Building $ENV for local testing..."
    if [ "$PLATFORM" = "Linux" ]; then
        # These important debug flags don't work on macos
        FLAGS+=(
            -fsanitize=address,undefined,bounds,pointer-overflow,leak
            -fno-omit-frame-pointer
        )
    fi  
    clang -g -O0 ${FLAGS[@]}
elif [ "$MODE" = "fast" ]; then
    echo "Building optimized $ENV for local testing..."
    clang -pg -O2 -DNDEBUG ${FLAGS[@]}
    echo "Built to: $ENV"
else
    echo "Invalid mode specified: local|fast|web"
    exit 1
fi
