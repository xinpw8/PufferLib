# Native SONIC and robot-state validation

Validated on Spark with native C++17/CUDA executables on 2026-09-13. No Python
interpreter, Torch runtime, or learned trainer policy was used by these tests.
Production controller dependencies are CUDA, cuBLAS, OpenSSL libcrypto, and the
C++ runtime. ONNX Runtime is a separate CPU oracle dependency in the controller
test executable only.

| Test | Result | Observed maximum error | Elapsed |
| --- | --- | --- | --- |
| SONIC batch 16 | Pass | Encoder 0; decoder 8.82148743e-6 | 0.68 s |
| SONIC batch 1024 | Strict comparison fails | One of 65,536 encoder tokens differs by 0.0625; decoder 1.33514404e-5 | 1.05 s |
| Robot state and drives | Pass | 414,212 values checked; history/actions/drive fixtures exact; encoder features within 1e-7 | 0.31 s |

The batch-1024 encoder mismatch is at row 768, token 62: native CUDA 0.4375,
CPU oracle 0.375. It is a one-bin quantizer difference on the existing synthetic
probe. Both separate bias addition and bias supplied through GEMM produced the
same mismatch. The strict test exits with status 1. This result does not establish
bitwise equivalence or REK parity. The unchanged decoder tolerance is 2e-5.

At both batch sizes, packed/full encoder input paths agree exactly, and CUDA
graph replay agrees exactly with eager native execution. Native loading verified
14,415,581 float32 coefficients, totaling 57,662,324 bytes. Inspection rejects an
incorrect explicit batch and truncated protobuf length. The controller reads the
existing exported ONNX files, validates graph I/O and operator counts, and loads
the G1 coefficients once. Its SHA-256 getters report the exact bytes loaded.

`test_robot_state.cu` includes the unchanged `gear_sonic_native_batch.c` as its
CPU observation/history/action-transform oracle. It directly calls assembly
stages. Its encoder/decoder inference stubs are unreachable and execute no
policies. The test covers eight rows, 18 ticks, suspension, selected-row history
reset, three graph replays, clipping, joint limits, retained filter values,
even/odd substeps, retained reset targets, dampened forces, and reset isolation.

The text files in `validation/` preserve the captured native test output. No
test binaries or model files are included in this source directory. The test
scratch directory on Spark is
`/home/spark-advantage/rek-training/native5-controller-build-20260913`.

## Reproduction

Run on Spark. `SRC` is the staged `ocean/rek_g1/native5` source directory and
`OUT` is an existing build directory. The ORT headers used in this test are the
unmodified `onnxruntime_c_api.h` and `onnxruntime_ep_c_api.h` from Microsoft's
`v1.24.4` release; `ORT_INC` identifies their directory. `ORT_LIB` identifies a
directory exposing the matching native `libonnxruntime.so.1` and full library.
These commands do not invoke Python.

```sh
nvcc -std=c++17 -O3 -lineinfo -arch=sm_121 -Xcompiler=-Wall,-Wextra \
  -c "$SRC/sonic_controller.cu" -o "$OUT/sonic_controller.o"
nvcc -std=c++17 -O3 -lineinfo -arch=sm_121 -Xcompiler=-Wall,-Wextra \
  -c "$SRC/robot_state.cu" -o "$OUT/robot_state.o"

nvcc -std=c++17 -O3 -lineinfo -arch=sm_121 \
  -I"$ORT_INC" "$SRC/test_sonic_controller.cu" "$OUT/sonic_controller.o" \
  -Xlinker="$ORT_LIB/libonnxruntime.so.1.24.4" \
  -Xlinker=-rpath -Xlinker="$ORT_LIB" -lcublas -lcrypto \
  -o "$OUT/test_sonic_controller"
nvcc -std=c++17 -O3 -lineinfo -arch=sm_121 \
  -Xcompiler=-Wall,-Wextra,-ffp-contract=off -I"$ORT_INC" -I"$SRC/.." \
  "$SRC/test_robot_state.cu" "$OUT/robot_state.o" \
  -o "$OUT/test_robot_state"

"$OUT/test_sonic_controller" 16 \
  /home/spark-advantage/rek-training/gpu-runtime-20260910/controller-batch16-aa4b4d5c-20260910/model_encoder.batch16.onnx \
  /home/spark-advantage/rek-training/gpu-runtime-20260910/controller-batch16-aa4b4d5c-20260910/model_decoder.batch16.onnx
"$OUT/test_sonic_controller" 1024 \
  /home/spark-advantage/rek-training/gpu-runtime-20260910/controller-batch1024-20260910/model_encoder.batch1024.onnx \
  /home/spark-advantage/rek-training/gpu-runtime-20260910/controller-batch1024-20260910/model_decoder.batch1024.onnx
"$OUT/test_robot_state"
```

Add `--inspect-only` to either controller command to read and validate its
model tensors without initializing CUDA or running inference. All
`sonic_controller_*` functions returning `int` use 1 for success, 0 for failure.
The robot-state functions return `cudaError_t`, with `cudaSuccess` equal to 0.
