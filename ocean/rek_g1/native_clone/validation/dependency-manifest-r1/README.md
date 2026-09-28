# Native clone dependency manifest

This release binds build-r6, source-r6, run-r4, app-r3, and tools-r5 on the existing Spark. The binary SHA256 is 90a004a6aa89a72d13e1c44abe10f2adea70059c746075b062bedb442ed36b73.

DEPENDENCIES.json records 586 exact paths, byte sizes and SHA256 values. All required paths existed and every recorded input pin matched. The path-size sum is 1,243,442,075 bytes, which includes shared-library aliases and categories that overlap; it is not a required download size. No large assets or system libraries were copied for this inventory.

Runtime dependencies include the 50,100,513-byte encoder and 40,900,688-byte decoder, physics/render models, 86 motion/model asset files, 22 foot-feature files, 46 cached Warp PTX modules (55,736,056 bytes), conditional PTX, MuJoCo 3.7.0, CUDA 13 libraries, EGL/GLVND vendor libraries, and the Node viewer. These are exact existing paths, including absolute paths inside the kernel catalog. Removing the Warp cache or moving dependencies requires a deliberate path migration.

The build uses eight freshly built object files plus 27 hash-pinned historical objects. The manifest maps every reused object to its recorded translation-unit source and preserves the historical build-script/manifest hashes. The small source-r6 snapshot contains the units rebuilt for this release, not all historical object sources. Historical source paths are listed separately. No clean-room rebuild, transitive-header equivalence, cross-platform portability, or redistribution-license claim is made.

Recorded platform: Linux aarch64, CUDA nvcc13.0.88 targeting sm_121, g++13.3.0, Node18.19.1, NVIDIA driver580.95.05. The host GPU driver and system/toolkit installation remain external prerequisites.

To recheck the existing-host files without starting the worker or GPU, run:

```sh
python3 verify_dependencies.py DEPENDENCIES.json
```

To rebuild the same selected source using the already pinned historical objects, choose a new output directory and run on Spark:

```sh
REK_CLONE_SOURCE_DIR=/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/source-r6 \
REK_CLONE_BUILD_DIR=/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/build-reproduction-new \
bash /home/spark-advantage/rek-training/rek-native-clone-20260927-r1/tools-r5/build-native.sh
```

That build command is recorded from the actual frozen tooling; it was not rerun by the inventory task. It refuses an existing output directory and verifies reused object pins. Root owns the launch instructions and integrated GPU acceptance evidence. The manifest itself records only read-only dependency verification.
