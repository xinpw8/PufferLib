import ctypes
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = Path(__file__).with_name("test_osrs_entity_encoders_cuda.cu")
ITEM_HEADER = ROOT / "ocean/osrs/osrs_item_obs_generated.h"
ITEM_SOURCE = ITEM_HEADER.read_text()
BATCH = 9
HIDDEN = 40
BOTTLENECK = 16
INVENTORY_START = 52
INVENTORY_COUNT = 28
EQUIPMENT_START = 80
EQUIPMENT_COUNT = 11


@dataclass(frozen=True)
class EncoderContract:
    name: str
    kind: int
    obs_size: int
    npc_start: int
    npc_count: int
    npc_obs_features: int
    npc_features: int
    type_count: int
    type_code_scale: int


CONTRACTS = (
    EncoderContract(
        name="colosseum",
        kind=0,
        obs_size=904,
        npc_start=101,
        npc_count=24,
        npc_obs_features=23,
        npc_features=34,
        type_count=12,
        type_code_scale=1,
    ),
    EncoderContract(
        name="inferno",
        kind=1,
        obs_size=530,
        npc_start=124,
        npc_count=14,
        npc_obs_features=13,
        npc_features=26,
        type_count=14,
        type_code_scale=16,
    ),
)


def _generated_integer(name: str) -> int:
    match = re.search(rf"^#define\s+{name}\s+(\d+)\s*$", ITEM_SOURCE, re.MULTILINE)
    assert match, f"generated item-table constant {name} is missing"
    return int(match.group(1))


def _load_item_table() -> np.ndarray:
    columns = _generated_integer("OSRS_ITEM_OBS_TABLE_COLS")
    expected_rows = _generated_integer("OSRS_ITEM_OBS_TABLE_ROWS")
    rows = []
    for line in ITEM_SOURCE.splitlines():
        if not line.lstrip().startswith("X("):
            continue
        values = re.findall(
            r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?",
            line,
            re.IGNORECASE,
        )
        rows.append([float(value) for value in values[-columns:]])
        if len(rows) == expected_rows:
            break
    table = np.asarray(rows, dtype=np.float32)
    assert table.shape == (expected_rows, columns)
    return table


def _make_observations(
    contract: EncoderContract,
    table_rows: int,
) -> np.ndarray:
    linear = np.arange(
        BATCH * contract.obs_size,
        dtype=np.float32,
    ).reshape(BATCH, contract.obs_size)
    observations = np.sin(linear * np.float32(0.013)) * np.float32(0.01)

    npc_block = contract.npc_count * contract.npc_obs_features
    observations[:, contract.npc_start : contract.npc_start + npc_block] = 0
    npc_cells = observations[
        :,
        contract.npc_start : contract.npc_start + npc_block,
    ].reshape(BATCH, contract.npc_count, contract.npc_obs_features)
    for npc_type in range(contract.type_count):
        flat_index = npc_type + 1
        batch, cell = divmod(flat_index, contract.npc_count)
        npc_cells[batch, cell, 0] = (npc_type + 1) / contract.type_code_scale
        npc_cells[batch, cell, 1:] = np.linspace(
            0.01 * (npc_type + 1),
            0.01 * (npc_type + contract.npc_obs_features - 1),
            contract.npc_obs_features - 1,
            dtype=np.float32,
        )

    code_scale = _generated_integer("OSRS_ITEM_OBS_CODE_SCALE")
    observations[:, INVENTORY_START : INVENTORY_START + INVENTORY_COUNT] = 0
    inventory_cells = observations[
        :, INVENTORY_START : INVENTORY_START + INVENTORY_COUNT
    ].reshape(BATCH, INVENTORY_COUNT)
    assert table_rows <= inventory_cells.size
    for code in range(table_rows):
        batch, cell = divmod(code, INVENTORY_COUNT)
        inventory_cells[batch, cell] = code / code_scale

    equipment_cells = observations[
        :, EQUIPMENT_START : EQUIPMENT_START + EQUIPMENT_COUNT
    ].reshape(BATCH, EQUIPMENT_COUNT)
    for batch in range(BATCH):
        for cell in range(EQUIPMENT_COUNT):
            code = 1 + (batch * EQUIPMENT_COUNT + cell) % (table_rows - 1)
            equipment_cells[batch, cell] = code / code_scale
    return observations


def _materialize_items(
    observations: np.ndarray,
    start: int,
    count: int,
    table: np.ndarray,
):
    source = observations[:, start : start + count].reshape(BATCH, count)
    code_scale = _generated_integer("OSRS_ITEM_OBS_CODE_SCALE")
    codes = np.rint(source * code_scale).astype(np.int64)
    assert np.all((codes >= 0) & (codes < table.shape[0]))
    return table[codes]


def _materialize(
    contract: EncoderContract,
    observations: np.ndarray,
    table: np.ndarray,
):
    npc_source = observations[
        :,
        contract.npc_start : contract.npc_start
        + contract.npc_count * contract.npc_obs_features,
    ].reshape(BATCH, contract.npc_count, contract.npc_obs_features)
    npc_records = np.zeros(
        (BATCH, contract.npc_count, contract.npc_features),
        dtype=np.float32,
    )
    npc_codes = np.rint(
        npc_source[:, :, 0] * contract.type_code_scale,
    ).astype(np.int64)
    for npc_type in range(contract.type_count):
        npc_records[:, :, npc_type] = npc_codes == npc_type + 1
    npc_records[:, :, contract.type_count :] = npc_source[:, :, 1:]

    inventory_records = _materialize_items(
        observations,
        INVENTORY_START,
        INVENTORY_COUNT,
        table,
    )
    equipment_records = _materialize_items(
        observations,
        EQUIPMENT_START,
        EQUIPMENT_COUNT,
        table,
    )
    return npc_records, inventory_records, equipment_records


def _torch_reference(
    torch,
    functional,
    contract,
    observations,
    table,
    weights,
):
    npc_records, inventory_records, equipment_records = _materialize(
        contract, observations.detach().cpu().numpy(), table
    )
    npc = torch.from_numpy(npc_records).to(observations.device)
    inventory = torch.from_numpy(inventory_records).to(observations.device)
    equipment = torch.from_numpy(equipment_records).to(observations.device)
    (
        global_w,
        inventory_l1_w,
        inventory_l2_w,
        equipment_l1_w,
        equipment_l2_w,
        npc_l1_w,
        npc_l2_w,
    ) = weights

    def branch(records, l1_weight, l2_weight, active_width):
        hidden = functional.gelu(
            functional.linear(records, l1_weight),
            approximate="tanh",
        )
        values = functional.linear(hidden, l2_weight)
        active = records[:, :, :active_width].sum(dim=2) > 0
        pooled = values.masked_fill(
            ~active.unsqueeze(2),
            -torch.inf,
        ).amax(dim=1)
        return torch.where(
            active.any(dim=1, keepdim=True),
            pooled,
            torch.zeros_like(pooled),
        )

    return (
        functional.linear(observations, global_w)
        + branch(inventory, inventory_l1_w, inventory_l2_w, 1)
        + branch(equipment, equipment_l1_w, equipment_l2_w, 1)
        + branch(npc, npc_l1_w, npc_l2_w, contract.type_count)
    )


def _configure_library(path: Path):
    library = ctypes.CDLL(path)
    pointer = ctypes.c_void_p
    library.osrs_entity_test_init.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.osrs_entity_test_set_weights.argtypes = [pointer] * 7
    library.osrs_entity_test_forward.argtypes = [
        pointer,
        pointer,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.osrs_entity_test_backward.argtypes = [pointer, ctypes.c_int, ctypes.c_int]
    library.osrs_entity_test_get_grad.argtypes = [ctypes.c_int, pointer]
    return library


@pytest.fixture(scope="session")
def cuda_library(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    raylib_include = next(ROOT.glob("raylib-*/include"))
    raylib_archive = raylib_include.parent / "lib/libraylib.a"
    pytest.importorskip("torch")
    library_path = (
        tmp_path_factory.mktemp("osrs_entity_encoder") / "osrs_entity_encoder.so"
    )
    subprocess.run(
        [
            nvcc,
            "-shared",
            "-o",
            str(library_path),
            str(CUDA_SOURCE),
            "-I",
            str(ROOT / "src"),
            "-I",
            str(raylib_include),
            "-Xlinker",
            "--no-as-needed",
            "-lcublas",
            "-lcudnn",
            "-lcurand",
            "-lnccl",
            "-lnvidia-ml",
            "-lcusolver",
            str(raylib_archive),
            "-lGL",
            "-lm",
            "-lpthread",
            "-ldl",
            "-lrt",
            "--compiler-options",
            "-fPIC",
            "-Xcompiler",
            "-O2",
        ],
        cwd=ROOT,
        check=True,
    )
    return _configure_library(library_path)


def _pointer(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda contract: contract.name)
def test_cuda_forward_and_all_weight_gradients(cuda_library, contract):
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    cuda_library.osrs_entity_test_init(
        contract.kind, BATCH, contract.obs_size, HIDDEN
    )

    table = _load_item_table()
    observations_np = _make_observations(contract, table.shape[0])
    observations = torch.from_numpy(observations_np).cuda()
    generator = torch.Generator(device="cuda").manual_seed(6100 + contract.kind)
    shapes = (
        (HIDDEN, contract.obs_size),
        (BOTTLENECK, table.shape[1]),
        (HIDDEN, BOTTLENECK),
        (BOTTLENECK, table.shape[1]),
        (HIDDEN, BOTTLENECK),
        (BOTTLENECK, contract.npc_features),
        (HIDDEN, BOTTLENECK),
    )
    weights = [
        (torch.randn(shape, generator=generator, device="cuda") * 0.05).requires_grad_()
        for shape in shapes
    ]
    cuda_library.osrs_entity_test_set_weights(*[_pointer(weight) for weight in weights])

    cuda_output = torch.empty(BATCH, HIDDEN, device="cuda")
    cuda_library.osrs_entity_test_forward(
        _pointer(cuda_output),
        _pointer(observations),
        BATCH,
        contract.obs_size,
    )
    reference_output = _torch_reference(
        torch,
        functional,
        contract,
        observations,
        table,
        weights,
    )
    torch.testing.assert_close(cuda_output, reference_output, atol=3e-4, rtol=3e-4)

    output_gradient = torch.randn(
        (BATCH, HIDDEN),
        generator=generator,
        device="cuda",
    )
    reference_output.backward(output_gradient)
    cuda_library.osrs_entity_test_backward(_pointer(output_gradient), BATCH, HIDDEN)

    for index, weight in enumerate(weights):
        cuda_gradient = torch.empty_like(weight)
        cuda_library.osrs_entity_test_get_grad(index, _pointer(cuda_gradient))
        torch.testing.assert_close(cuda_gradient, weight.grad, atol=5e-3, rtol=5e-3)
