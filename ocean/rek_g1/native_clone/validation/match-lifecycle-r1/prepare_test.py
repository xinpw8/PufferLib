from pathlib import Path
import shutil,hashlib,json
root=Path(__file__).resolve().parent
base=Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile\ocean\rek_g1')
deps=root/'dependencies';deps.mkdir(exist_ok=True)
names=[p.name for p in base.glob('*.h')]+['g1_fight_state.c','g1_fall_state.c','g1_combat_tick.c','g1_hit_detector.c','test_g1_fight_state.c','test_g1_combat_tick.c']
for name in names: shutil.copyfile(base/name,deps/name)
stub=root/'stubs';stub.mkdir(exist_ok=True)
(stub/'cuda_runtime_api.h').write_text('#pragma once\n#include <stddef.h>\ntypedef int cudaError_t;typedef void* cudaStream_t;\n')
(stub/'cuda_runtime.h').write_text('#pragma once\n#include "cuda_runtime_api.h"\n#define __global__\n#define __device__\n#define __constant__\nstruct Dim {unsigned x=1;}; static Dim blockIdx{0},blockDim{1},threadIdx{0},gridDim{1};\n')
source=(root/'source/g1_native_combat_cuda.cu').read_text()
kernel=source[:source.index('extern "C" size_t rek_g1_cuda_native_combat_state_size')]
(root/'kernels_cpu.inc').write_text(kernel)
pins={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['source','original','dependencies'] for p in (root/folder).iterdir() if p.is_file()}
(root/'SOURCE-PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
print(len(pins),'pinned files; CUDA bodies copied exactly through namespace end for CPU execution')
