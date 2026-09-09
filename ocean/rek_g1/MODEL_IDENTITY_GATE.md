# Exact-batch model identity gate

The native `rek_g1` extension accepts only the encoder and decoder identified by
one build-validated `rek.g1_gear_sonic_explicit_batch.v1` manifest. The manifest
must describe the exact robot batch compiled into the extension. Its source
graphs must match the two source SHA-256 values pinned in
`gear_sonic_batch_model.py`. Equivalence fields stored in the manifest are
descriptive evidence and cannot authorize a build by themselves.

Before emitting the C identity header, the generator performs two independent
checks for both encoder and decoder:

1. It runs `rewrite_model` against the pinned source graph in a temporary
   directory and requires the regenerated ONNX SHA-256 to equal the supplied
   exact-batch graph SHA-256.
2. It runs `verify_equivalence` against the source and supplied graph using
   `CPUExecutionProvider`, seed `20260831`, deterministic finite fixtures, and
   zero tolerance. Both outputs must be finite `float32`, have the expected
   shape, contain distinct batch rows, and be bitwise equal as little-endian
   binary32 tensors.

The temporary independently rewritten graphs are removed before the header is
written. A shape-compatible graph, forged manifest equivalence fields, or a
graph that matches only its claimed hashes cannot pass these checks.

Set these build-only paths before `./build.sh rek_g1`:

```sh
export REK_G1_PUBLIC_MODEL_BUNDLE=/path/to/pinned/source/models
export REK_G1_EXPLICIT_BATCH_MANIFEST=/path/to/exact-batch/explicit_batch_manifest.json
export REK_G1_MODEL_GATE_PYTHON=/path/to/python-with-onnx-and-onnxruntime
```

The model-gate interpreter must have `numpy`, `onnx`, and `onnxruntime`
available. It defaults to `python`. An explicit interpreter lets the extension
continue building against a separate active Python environment. Missing
execution support rejects the build.

The build generates a private header below `build/static_rek_g1/`. It embeds
only the manifest SHA-256, source and generated graph SHA-256 values, generated
graph byte counts, batch count, and classification flags. It does not copy ONNX
payloads, retain the temporary rewrite, or embed machine-local paths. The
generator's JSON result also reports the freshly measured equivalence record
and independent rewrite SHA-256 for audit use.

At startup, `REK_G1_ENCODER_ONNX` and `REK_G1_DECODER_ONNX` are opened once with
`O_NOFOLLOW`. The binding reads and hashes the retained regular-file descriptor
bytes, requires the compiled byte counts and SHA-256 values, and constructs both
ONNX Runtime sessions directly from those verified byte arrays. The arrays are
released only after session construction returns. The runtime robot count must
equal the compiled exact batch. A path replacement cannot substitute different
session bytes between verification and loading. A different graph
serialization, swapped encoder and decoder, symlink, truncated file, or batch
mismatch is rejected.

This gate establishes identity with one equivalence-validated public-family
candidate bundle only. `rek_parity_claim`, `current_steam_authority`, and
`training_enabled` remain false. It does not establish current Steam/server
model identity, trajectory parity, or readiness for training.
