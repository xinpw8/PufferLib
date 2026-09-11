#!/bin/bash
set -e

# Usage:
#   ./build.sh breakout              # Build _C.so with breakout statically linked
#   ./build.sh breakout --float      # float32 precision (required for --slowly)
#   ./build.sh breakout --cpu        # CPU fallback, torch only
#   ./build.sh breakout --debug      # Debug build
#   ./build.sh breakout --local      # Standalone executable (debug, sanitizers)
#   ./build.sh breakout --fast       # Standalone executable (optimized)
#   ./build.sh breakout --web        # Emscripten web build
#   ./build.sh breakout --profile    # Kernel profiling binary
#   ./build.sh all                   # Build all envs with default and --float

if [ -z "$1" ]; then
    echo "Usage: ./build.sh ENV_NAME [--float] [--debug] [--local|--fast|--web|--profile|--cpu|--all]"
    exit 1
fi
ENV=$1
shift

for arg in "$@"; do
    case $arg in
        --float) PRECISION="-DPRECISION_FLOAT" ;;
        --debug) DEBUG=1 ;;
        --local) MODE=local ;;
        --fast)  MODE=fast ;;
        --web)   MODE=web ;;
        --profile) MODE=profile ;;
        --cpu)   MODE=cpu; PRECISION="-DPRECISION_FLOAT" ;;
        *) echo "Error: unknown argument '$arg'" && exit 1 ;;
    esac
done

if [ "$ENV" = "all" ]; then
    FAILED=""
    for env_dir in ocean/*/; do
        env=$(basename "$env_dir")
        if bash "$0" "$env" && bash "$0" "$env" --float; then
            echo "OK: $env"
        else
            echo "FAIL: $env"
            FAILED="$FAILED\n  $env"
        fi
    done

    if [ -n "$FAILED" ]; then
        echo -e "\nFailed builds:$FAILED"
        exit 1
    fi
    exit 0
fi

# Linux/mac
PLATFORM="$(uname -s)"
MACHINE="$(uname -m)"
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    OMP_LIB=-lomp5
    SANITIZE_FLAGS=(-fsanitize=address,undefined,bounds,pointer-overflow,leak -fno-omit-frame-pointer)
    STANDALONE_LDFLAGS=(-lGL -ldl -lrt -lX11)
    PROFILE_GRAPHICS_LDFLAGS=(-lGL)
    SHARED_LDFLAGS=(-Bsymbolic-functions)
else
    RAYLIB_NAME='raylib-5.5_macos'
    OMP_LIB=-lomp
    SANITIZE_FLAGS=()
    STANDALONE_LDFLAGS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
    PROFILE_GRAPHICS_LDFLAGS=(-framework OpenGL)
    SHARED_LDFLAGS=(-framework Cocoa -framework OpenGL -framework IOKit -undefined dynamic_lookup)
fi

CLANG_WARN=(
    -Wall
    -ferror-limit=3
    -Werror=incompatible-pointer-types
    -Werror=return-type
    -Wno-error=incompatible-pointer-types-discards-qualifiers
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-error=array-parameter
)

download() {
    local name=$1 url=$2
    [ -d "$name" ] && return
    echo "Downloading $name..."
    case "$url" in
        *.zip) curl -sL "$url" -o "$name.zip" && unzip -q "$name.zip" && rm "$name.zip" ;;
        *)     curl -sL "$url" -o "$name.tar.gz" && tar xf "$name.tar.gz" && rm "$name.tar.gz" ;;
    esac
}

RAYLIB_URL="https://github.com/raysan5/raylib/releases/download/5.5"
if [ "$ENV" = "rek" ] || [ "$ENV" = "rek_g1" ] || [ "$ENV" = "rek_match" ] || [ "$ENV" = "rek_sandbox" ] || [ "$ENV" = "rek_fight" ]; then
    RAYLIB_A=""
    RAYLIB_IFLAGS=()
    INCLUDES=(-I./src -I./vendor)
    LINK_ARCHIVES=()
    PROFILE_GRAPHICS_LDFLAGS=()
elif [ "$MODE" = "web" ]; then
    RAYLIB_NAME='raylib-5.5_webassembly'
    download "$RAYLIB_NAME" "$RAYLIB_URL/$RAYLIB_NAME.zip"
    RAYLIB_A="$RAYLIB_NAME/lib/libraylib.a"
    RAYLIB_IFLAGS=(-I./$RAYLIB_NAME/include)
    INCLUDES=(-I./$RAYLIB_NAME/include -I./src -I./vendor)
    LINK_ARCHIVES=("$RAYLIB_A")
else
    download "$RAYLIB_NAME" "$RAYLIB_URL/$RAYLIB_NAME.tar.gz"
    RAYLIB_A="$RAYLIB_NAME/lib/libraylib.a"
    RAYLIB_IFLAGS=(-I./$RAYLIB_NAME/include)
    INCLUDES=(-I./$RAYLIB_NAME/include -I./src -I./vendor)
    LINK_ARCHIVES=("$RAYLIB_A")
fi
EXTRA_SRC=""
EXTRA_LDFLAGS=()
NATIVE_SOURCES=()

if [ "$ENV" = "constellation" ]; then
    SRC_DIR="constellation"
    EXTRA_SRC="vendor/cJSON.c"
    OUTPUT_NAME="seethestars"
elif [ "$ENV" = "trailer" ]; then
    SRC_DIR="trailer"
    OUTPUT_NAME="trailer/trailer"
elif [ "$ENV" = "impulse_wars" ]; then
    SRC_DIR="ocean/$ENV"
    if [ "$MODE" = "web" ]; then BOX2D_NAME='box2d-web'
    elif [ "$PLATFORM" = "Linux" ]; then BOX2D_NAME='box2d-linux-amd64'
    else BOX2D_NAME='box2d-macos-arm64'
    fi
    BOX2D_URL="https://github.com/capnspacehook/box2d/releases/latest/download"
    download "$BOX2D_NAME" "$BOX2D_URL/$BOX2D_NAME.tar.gz"
    INCLUDES+=(-I./$BOX2D_NAME/include -I./$BOX2D_NAME/src)
    LINK_ARCHIVES+=("./$BOX2D_NAME/libbox2d.a")
elif [ "$ENV" = "nethack" ]; then
    SRC_DIR="ocean/$ENV"
    NLE_DIR="vendor/nle"
    NLE_REPO="https://github.com/liujonathan24/NetHack.git"
    if [ ! -d "$NLE_DIR/src" ]; then
        echo "Cloning modified NLE from $NLE_REPO ..."
        git clone --depth 1 "$NLE_REPO" "$NLE_DIR"
    fi
    NETHACK_LIB_DIR="$(pwd)/$NLE_DIR/src/build"
    if [ ! -f "$NETHACK_LIB_DIR/libnethack.so" ]; then
        echo "Building libnethack.so ..."
        make -C "$NETHACK_LIB_DIR" nethack -j$(nproc)
    fi
    INCLUDES+=(-I./$NLE_DIR/include)
    EXTRA_LDFLAGS+=(-L"$NETHACK_LIB_DIR" -lnethack -Wl,-rpath,"$NETHACK_LIB_DIR" -ldl)
elif [ "$ENV" = "rek" ] || [ "$ENV" = "rek_g1" ] || [ "$ENV" = "rek_match" ] || [ "$ENV" = "rek_sandbox" ] || [ "$ENV" = "rek_fight" ]; then
    SRC_DIR="ocean/$ENV"
    # The recovered plant has no renderer and does not link Raylib.
    if [ "$MODE" = "web" ] || [ "$MODE" = "local" ] || [ "$MODE" = "fast" ]; then
        if [ "$ENV" = "rek_sandbox" ]; then
            echo "Error: use ocean/rek_sandbox/test_rek_sandbox.c for standalone checks"
        elif [ "$ENV" = "rek_match" ]; then
            echo "Error: use ocean/rek_match/test_rek_match.c for standalone checks"
        elif [ "$ENV" = "rek_fight" ]; then
            echo "Error: use ocean/rek_fight/test_rek_fight.c for standalone checks"
        else
            echo "Error: use ocean/rek/test_rek.c for standalone REK plant checks"
        fi
        exit 1
    fi
    MUJOCO_HOME=${MUJOCO_HOME:?set MUJOCO_HOME to a directory containing include/mujoco/mujoco.h}
    MUJOCO_LIB=${MUJOCO_LIB:?set MUJOCO_LIB to the exact native libmujoco shared library}
    [ -f "$MUJOCO_HOME/include/mujoco/mujoco.h" ] \
        || { echo "Error: MuJoCo header missing below MUJOCO_HOME"; exit 1; }
    [ -f "$MUJOCO_LIB" ] \
        || { echo "Error: MUJOCO_LIB is not a file"; exit 1; }
    INCLUDES+=(-I"$MUJOCO_HOME/include")
    EXTRA_LDFLAGS+=("$MUJOCO_LIB" -Wl,-rpath,"$(dirname "$MUJOCO_LIB")")
    if [ "$ENV" = "rek_g1" ]; then
        ONNXRUNTIME_HOME=${ONNXRUNTIME_HOME:?set ONNXRUNTIME_HOME to a directory containing include/onnxruntime_c_api.h}
        ONNXRUNTIME_LIB=${ONNXRUNTIME_LIB:?set ONNXRUNTIME_LIB to the exact native libonnxruntime shared library}
        REK_G1_PUBLIC_MODEL_BUNDLE=${REK_G1_PUBLIC_MODEL_BUNDLE:?set REK_G1_PUBLIC_MODEL_BUNDLE to the pinned public-family source model directory}
        REK_G1_EXPLICIT_BATCH_MANIFEST=${REK_G1_EXPLICIT_BATCH_MANIFEST:?set REK_G1_EXPLICIT_BATCH_MANIFEST to the validated exact-batch manifest}
        [ -f "$ONNXRUNTIME_HOME/include/onnxruntime_c_api.h" ] \
            || { echo "Error: ONNX Runtime C API header missing below ONNXRUNTIME_HOME"; exit 1; }
        [ -f "$ONNXRUNTIME_LIB" ] \
            || { echo "Error: ONNXRUNTIME_LIB is not a file"; exit 1; }
        INCLUDES+=(-I"$ONNXRUNTIME_HOME/include")
        EXTRA_LDFLAGS+=("$ONNXRUNTIME_LIB" -Wl,-rpath,"$(dirname "$ONNXRUNTIME_LIB")" -lcrypto -lm)
    fi
elif [ -d "ocean/$ENV" ]; then
    SRC_DIR="ocean/$ENV"
else
    echo "Error: environment '$ENV' not found" && exit 1
fi

OUTPUT_NAME=${OUTPUT_NAME:-$ENV}

# Standalone environment build
# -mavx2 enables AVX2 intrinsics (__m256, _mm256_*) which drive.h and
# src/bf16.h use directly. x86_64 only — strip if porting to ARM/Apple Silicon.
SIMD_FLAGS=()
if [ "$MACHINE" = "x86_64" ] || [ "$MACHINE" = "amd64" ]; then
    SIMD_FLAGS=(-mavx2 -mfma)
fi
if [ -n "$DEBUG" ] || [ "$MODE" = "local" ]; then
    CLANG_OPT=(-g -O0 "${CLANG_WARN[@]}" "${SANITIZE_FLAGS[@]}" "${SIMD_FLAGS[@]}")
    NVCC_OPT="-O0 -g"
    LINK_OPT="-g"
else
    CLANG_OPT=(-O2 -DNDEBUG "${CLANG_WARN[@]}" "${SIMD_FLAGS[@]}")
    NVCC_OPT="-O2 --threads 0"
    LINK_OPT="-O2"
fi
if [ "$MODE" = "local" ] || [ "$MODE" = "fast" ]; then
    FLAGS=(
        "${INCLUDES[@]}"
        "$SRC_DIR/$ENV.c" $EXTRA_SRC -o "$OUTPUT_NAME"
        "${LINK_ARCHIVES[@]}"
        "${EXTRA_LDFLAGS[@]}"
        "${STANDALONE_LDFLAGS[@]}"
        -lm -lpthread -fopenmp
        -DPLATFORM_DESKTOP
    )
    echo "Compiling $ENV..."
    ${CC:-clang} "${CLANG_OPT[@]}" "${FLAGS[@]}"
    echo "Built: ./$OUTPUT_NAME"
    exit 0
elif [ "$MODE" = "web" ]; then
    mkdir -p "build/web/$ENV"
    echo "Compiling $ENV for web..."
    emcc \
        -o "build/web/$ENV/game.html" \
        "$SRC_DIR/$ENV.c" $EXTRA_SRC \
        -O3 -Wall \
        "${LINK_ARCHIVES[@]}" \
        "${INCLUDES[@]}" \
        -L. -L./$RAYLIB_NAME/lib \
        -sASSERTIONS=2 -gsource-map \
        -sUSE_GLFW=3 -sUSE_WEBGL2=1 -sASYNCIFY -sFILESYSTEM -sFORCE_FILESYSTEM=1 \
        --shell-file vendor/minshell.html \
        -sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=512KB \
        -DNDEBUG -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3 \
        --preload-file resources/$ENV@resources/$ENV \
        --preload-file resources/shared@resources/shared
    echo "Built: build/web/$ENV/game.html"
    exit 0
fi

# Find cuDNN path
CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname "$(which nvcc)")")}}
CUDNN_IFLAG=""
CUDNN_LFLAG=""
for dir in /usr/local/cuda/include /usr/include; do
    if [ -f "$dir/cudnn.h" ]; then
        CUDNN_IFLAG="-I$dir"
        break
    fi
done
for dir in /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu; do
    if [ -f "$dir/libcudnn.so" ]; then
        CUDNN_LFLAG="-L$dir"
        break
    fi
done
if [ -z "$CUDNN_IFLAG" ]; then
    CUDNN_IFLAG=$(python -c "import nvidia.cudnn, os; print('-I' + os.path.join(nvidia.cudnn.__path__[0], 'include'))" 2>/dev/null || echo "")
fi
if [ -z "$CUDNN_LFLAG" ]; then
    CUDNN_LFLAG=$(python -c "import nvidia.cudnn, os; print('-L' + os.path.join(nvidia.cudnn.__path__[0], 'lib'))" 2>/dev/null || echo "")
fi

# NCCL include/lib fallback (mirrors the cuDNN fallback above).
# Needed when NCCL is provided by the nvidia-nccl-cu12 wheel in the active venv.
NCCL_IFLAG=""
NCCL_LFLAG=""
for dir in /usr/include /usr/local/cuda/include; do
    if [ -f "$dir/nccl.h" ]; then NCCL_IFLAG="-I$dir"; break; fi
done
for dir in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64; do
    if [ -f "$dir/libnccl.so" ] || [ -f "$dir/libnccl.so.2" ]; then NCCL_LFLAG="-L$dir"; break; fi
done
if [ -z "$NCCL_IFLAG" ]; then
    NCCL_IFLAG=$(python -c "import nvidia.nccl, os; print('-I' + os.path.join(nvidia.nccl.__path__[0], 'include'))" 2>/dev/null || echo "")
fi
if [ -z "$NCCL_LFLAG" ]; then
    NCCL_LFLAG=$(python -c "import nvidia.nccl, os; print('-L' + os.path.join(nvidia.nccl.__path__[0], 'lib'))" 2>/dev/null || echo "")
fi

WHEEL_RPATH_FLAGS=()
for lib_flag in "$CUDNN_LFLAG" "$NCCL_LFLAG"; do
    if [[ "$lib_flag" == -L* ]]; then
        WHEEL_RPATH_FLAGS+=("-Wl,-rpath,${lib_flag#-L}")
    fi
done

export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
export CCACHE_BASEDIR="$(pwd)"
export CCACHE_COMPILERCHECK=content
NVCC="ccache $CUDA_HOME/bin/nvcc"
CC="${CC:-$(command -v ccache >/dev/null && echo 'ccache clang' || echo 'clang')}"
ARCH=${NVCC_ARCH:-native}

PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$(python -c "import pybind11; print(pybind11.get_include())")
NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTPUT="pufferlib/_C${EXT_SUFFIX}"

BINDING_SRC="$SRC_DIR/binding.c"
mkdir -p build
STATIC_OBJ_DIR="build/static_${ENV}"
STATIC_LIB="build/libstatic_${ENV}.a"

if [ ! -f "$BINDING_SRC" ]; then
    echo "Error: $BINDING_SRC not found"
    exit 1
fi

NATIVE_SOURCES+=("$BINDING_SRC")
NATIVE_SOURCE_MANIFEST="$SRC_DIR/native_sources.txt"
if [ -f "$NATIVE_SOURCE_MANIFEST" ]; then
    while IFS= read -r native_source || [ -n "$native_source" ]; do
        native_source=${native_source%$'\r'}
        case "$native_source" in
            ""|\#*) continue ;;
        esac
        if [[ ! "$native_source" =~ ^[A-Za-z0-9_-]+\.c$ ]]; then
            echo "Error: invalid native source entry '$native_source' in $NATIVE_SOURCE_MANIFEST"
            exit 1
        fi
        native_source="$SRC_DIR/$native_source"
        [ -f "$native_source" ] \
            || { echo "Error: native source from manifest is missing: $native_source"; exit 1; }
        NATIVE_SOURCES+=("$native_source")
    done < "$NATIVE_SOURCE_MANIFEST"
fi

echo "Compiling static library for $ENV..."
mkdir -p "$STATIC_OBJ_DIR"
if [ "$ENV" = "rek_g1" ]; then
    REK_G1_MODEL_GATE_PYTHON=${REK_G1_MODEL_GATE_PYTHON:-python}
    command -v "$REK_G1_MODEL_GATE_PYTHON" >/dev/null 2>&1 \
        || { echo "Error: REK_G1_MODEL_GATE_PYTHON is not executable"; exit 1; }
    "$REK_G1_MODEL_GATE_PYTHON" \
        "$SRC_DIR/generate_g1_model_identity_header.py" \
        --source-bundle "$REK_G1_PUBLIC_MODEL_BUNDLE" \
        --manifest "$REK_G1_EXPLICIT_BATCH_MANIFEST" \
        --out "$STATIC_OBJ_DIR/g1_model_identity_generated.h"
    INCLUDES+=(-I"$STATIC_OBJ_DIR")
fi
STATIC_OBJECTS=()
native_source_index=0
for native_source in "${NATIVE_SOURCES[@]}"; do
    native_object="$STATIC_OBJ_DIR/$(printf '%03d' "$native_source_index")_$(basename "${native_source%.c}").o"
    ${CC:-clang} -c "${CLANG_OPT[@]}" $EXTRA_CFLAGS \
        -I. -Isrc -I$SRC_DIR -Ivendor \
        "${INCLUDES[@]}" \
        -I./$RAYLIB_NAME/include -I$CUDA_HOME/include \
        -DPLATFORM_DESKTOP \
        -fno-semantic-interposition -fvisibility=hidden \
        -fPIC -fopenmp \
        "$native_source" -o "$native_object"
    STATIC_OBJECTS+=("$native_object")
    native_source_index=$((native_source_index + 1))
done
STATIC_LIB_TMP="$STATIC_LIB.tmp.$$"
ar rcs "$STATIC_LIB_TMP" "${STATIC_OBJECTS[@]}"
mv -f "$STATIC_LIB_TMP" "$STATIC_LIB"

# Brittle hack: have to extract the tensor type from the static lib to build trainer
OBS_TENSOR_T=$(awk '/^#define OBS_TENSOR_T/{print $3}' "$BINDING_SRC")
if [ -z "$OBS_TENSOR_T" ]; then
    echo "Error: Could not find OBS_TENSOR_T in $BINDING_SRC"
    exit 1
fi

if [ -z "$MODE" ]; then
    echo "Compiling CUDA ($ARCH) training backend..."
    $NVCC -c -arch=$ARCH -Xcompiler -fPIC \
        -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1 \
        -Xcompiler=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
        -Xcompiler=-DPLATFORM_DESKTOP \
        -std=c++17 \
        -I. -Isrc \
        -I$PYTHON_INCLUDE -I$PYBIND_INCLUDE -I$NUMPY_INCLUDE \
        -I$CUDA_HOME/include $CUDNN_IFLAG $NCCL_IFLAG "${RAYLIB_IFLAGS[@]}" \
        -Xcompiler=-fopenmp \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        $PRECISION $NVCC_OPT \
        src/bindings.cu -o build/bindings.o

    LINK_CMD=(
        ${CXX:-g++} -shared -fPIC -fopenmp
        build/bindings.o "$STATIC_LIB" "${LINK_ARCHIVES[@]}"
        -L$CUDA_HOME/lib64 $CUDNN_LFLAG $NCCL_LFLAG
        "${WHEEL_RPATH_FLAGS[@]}"
        "${EXTRA_LDFLAGS[@]}"
        -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -lcudnn
        $OMP_LIB $LINK_OPT
        "${SHARED_LDFLAGS[@]}"
        -o "$OUTPUT"
    )
    "${LINK_CMD[@]}"
    echo "Built: $OUTPUT"

elif [ "$MODE" = "cpu" ]; then
    echo "Compiling CPU training backend..."
    ${CXX:-g++} -c -fPIC -fopenmp \
        -D_GLIBCXX_USE_CXX11_ABI=1 \
        -DPLATFORM_DESKTOP \
        -std=c++17 \
        -I. -Isrc \
        -I$PYTHON_INCLUDE -I$PYBIND_INCLUDE \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        $PRECISION $LINK_OPT \
        src/bindings_cpu.cpp -o build/bindings_cpu.o
    LINK_CMD=(
        ${CXX:-g++} -shared -fPIC -fopenmp
        build/bindings_cpu.o "$STATIC_LIB" "${LINK_ARCHIVES[@]}"
        "${EXTRA_LDFLAGS[@]}"
        -lm -lpthread $OMP_LIB $LINK_OPT
        "${SHARED_LDFLAGS[@]}"
        -o "$OUTPUT"
    )
    "${LINK_CMD[@]}"
    echo "Built: $OUTPUT"

elif [ "$MODE" = "profile" ]; then
    echo "Compiling profile binary ($ARCH)..."
    $NVCC $NVCC_OPT -arch=$ARCH -std=c++17 \
        -I. -Isrc -I$SRC_DIR -Ivendor \
        -I$CUDA_HOME/include $CUDNN_IFLAG $NCCL_IFLAG -I$RAYLIB_NAME/include \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        -Xcompiler=-DPLATFORM_DESKTOP \
        $PRECISION \
        -Xcompiler=-fopenmp \
        tests/profile_kernels.cu vendor/ini.c \
        "$STATIC_LIB" "${LINK_ARCHIVES[@]}" \
        "${EXTRA_LDFLAGS[@]}" \
        -lnccl -lnvidia-ml -lcublas -lcurand -lcudnn \
        "${PROFILE_GRAPHICS_LDFLAGS[@]}" -lm -lpthread $OMP_LIB \
        -o profile
    echo "Built: ./profile"
fi
