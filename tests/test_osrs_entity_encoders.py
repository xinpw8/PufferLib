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
ITEM_TABLE = ROOT / "ocean/osrs/osrs_item_obs_table.inc"
BATCH = 9
HIDDEN = 40
BOTTLENECK = 16


@dataclass(frozen=True)
class EncoderContract:
    name: str
    kind: int
    source: Path
    prefix: str
    obs_size: int
    npc_start: int
    npc_count: int
    npc_obs_features: int
    npc_features: int
    type_count: int
    type_code_scale: int
    inventory_start: int
    inventory_count: int
    inventory_obs_features: int
    inventory_features: int
    inventory_overlays: bool


CONTRACTS = (
    EncoderContract(
        "colosseum", 0, ROOT / "ocean/osrs_colosseum/osrs_colosseum.cu", "COLO_ENT",
        934, 130, 24, 23, 34, 12, 1, 36, 28, 3, 15, True,
    ),
    EncoderContract(
        "inferno", 1, ROOT / "ocean/osrs_inferno/osrs_inferno.cu", "INF_ENT",
        498, 54, 14, 13, 26, 14, 16, 460, 28, 1, 15, False,
    ),
)


def _integer_constant(source: str, name: str):
    match = re.search(rf"\b{name}\s*=\s*(\d+)\s*;", source)
    return int(match.group(1)) if match else None


def _generated_integer(name: str) -> int:
    source = ITEM_HEADER.read_text()
    match = re.search(rf"^#define\s+{name}\s+(\d+)\s*$", source, re.MULTILINE)
    assert match, f"generated item-table constant {name} is missing"
    return int(match.group(1))


def _load_item_table() -> np.ndarray:
    columns = _generated_integer("OSRS_ITEM_OBS_TABLE_COLS")
    rows = []
    for line in ITEM_TABLE.read_text().splitlines():
        if not line.startswith("{"):
            continue
        values = re.findall(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", line, re.IGNORECASE)
        rows.append([float(value) for value in values])
    table = np.asarray(rows, dtype=np.float32)
    assert table.shape == (_generated_integer("OSRS_ITEM_OBS_TABLE_ROWS"), columns)
    return table


def _source_contract(contract: EncoderContract) -> dict[str, int | None]:
    source = contract.source.read_text()
    prefix = contract.prefix
    values = {
        "npc_start": _integer_constant(source, f"{prefix}_NPC_START"),
        "npc_count": _integer_constant(source, f"{prefix}_NUM_NPCS"),
        "npc_features": _integer_constant(source, f"{prefix}_FEATS"),
        "type_count": _integer_constant(source, f"{prefix}_TYPE_ONEHOT"),
        "inventory_obs_features": _integer_constant(source, f"{prefix}_INV_OBS_FEATS"),
        "inventory_start": _integer_constant(source, f"{prefix}_INV_START"),
        "inventory_count": _integer_constant(source, f"{prefix}_INV_NUM_CELLS"),
        "inventory_features": _integer_constant(source, f"{prefix}_INV_FEATS"),
    }
    if contract.kind == 1:
        values["obs_size"] = _integer_constant(source, "INF_ENT_OBS_SIZE")
        values["npc_obs_features"] = _integer_constant(
            source, "INF_ENT_OBS_FEATS",
        )
        values["type_code_scale"] = _integer_constant(
            source, "INF_ENT_TYPE_CODE_SCALE",
        )
    return values


def _expected_source_contract(contract: EncoderContract) -> dict[str, int]:
    values = {
        "npc_start": contract.npc_start,
        "npc_count": contract.npc_count,
        "npc_features": contract.npc_features,
        "type_count": contract.type_count,
        "inventory_start": contract.inventory_start,
        "inventory_count": contract.inventory_count,
        "inventory_obs_features": contract.inventory_obs_features,
        "inventory_features": contract.inventory_features,
    }
    if contract.kind == 1:
        values["npc_obs_features"] = contract.npc_obs_features
        values["obs_size"] = contract.obs_size
        values["type_code_scale"] = contract.type_code_scale
    return values


def _make_observations(contract: EncoderContract, table_rows: int) -> np.ndarray:
    linear = np.arange(BATCH * contract.obs_size, dtype=np.float32).reshape(BATCH, contract.obs_size)
    observations = np.sin(linear * np.float32(0.013)) * np.float32(0.01)

    npc_block = contract.npc_count * contract.npc_obs_features
    observations[:, contract.npc_start:contract.npc_start + npc_block] = 0
    npc_cells = observations[:, contract.npc_start:contract.npc_start + npc_block].reshape(
        BATCH, contract.npc_count, contract.npc_obs_features,
    )
    for npc_type in range(contract.type_count):
        flat_index = npc_type + 1
        batch, cell = divmod(flat_index, contract.npc_count)
        npc_cells[batch, cell, 0] = (npc_type + 1) / contract.type_code_scale
        npc_cells[batch, cell, 1:] = np.linspace(
            0.01 * (npc_type + 1), 0.01 * (npc_type + contract.npc_obs_features - 1),
            contract.npc_obs_features - 1, dtype=np.float32,
        )

    inventory_block = contract.inventory_count * contract.inventory_obs_features
    observations[:, contract.inventory_start:contract.inventory_start + inventory_block] = 0
    inventory_cells = observations[
        :, contract.inventory_start:contract.inventory_start + inventory_block
    ].reshape(BATCH, contract.inventory_count, contract.inventory_obs_features)
    code_scale = _generated_integer("OSRS_ITEM_OBS_CODE_SCALE")
    assert table_rows <= inventory_cells.shape[0] * inventory_cells.shape[1]
    for code in range(table_rows):
        batch, cell = divmod(code, contract.inventory_count)
        inventory_cells[batch, cell, 0] = code / code_scale
        if contract.inventory_overlays and code:
            inventory_cells[batch, cell, 1] = float(code & 1)
            inventory_cells[batch, cell, 2] = (code % 17) / 20.0
    return observations


def _materialize(contract: EncoderContract, observations: np.ndarray, table: np.ndarray):
    npc_source = observations[
        :, contract.npc_start:
        contract.npc_start + contract.npc_count * contract.npc_obs_features
    ].reshape(BATCH, contract.npc_count, contract.npc_obs_features)
    npc_records = np.zeros(
        (BATCH, contract.npc_count, contract.npc_features), dtype=np.float32,
    )
    npc_codes = np.rint(npc_source[:, :, 0] * contract.type_code_scale).astype(np.int64)
    for npc_type in range(contract.type_count):
        npc_records[:, :, npc_type] = npc_codes == npc_type + 1
    npc_records[:, :, contract.type_count:] = npc_source[:, :, 1:]

    inventory_source = observations[
        :, contract.inventory_start:
        contract.inventory_start + contract.inventory_count * contract.inventory_obs_features
    ].reshape(BATCH, contract.inventory_count, contract.inventory_obs_features)
    code_scale = _generated_integer("OSRS_ITEM_OBS_CODE_SCALE")
    inventory_codes = np.rint(inventory_source[:, :, 0] * code_scale).astype(np.int64)
    assert np.all((inventory_codes >= 0) & (inventory_codes < table.shape[0]))
    inventory_records = table[inventory_codes].copy()
    if contract.inventory_overlays:
        equipped = _generated_integer("OSRS_ITEM_OBS_OVERLAY_EQUIPPED")
        hp_heal = _generated_integer("OSRS_ITEM_OBS_OVERLAY_HP_HEAL")
        inventory_records[:, :, equipped] = inventory_source[:, :, 1]
        gear = (inventory_records[:, :, 3] != 0) | (inventory_records[:, :, 4] != 0)
        inventory_records[:, :, hp_heal] = np.where(
            gear, inventory_records[:, :, hp_heal], inventory_source[:, :, 2],
        )
    return npc_records, inventory_records, npc_codes, inventory_codes


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda contract: contract.name)
def test_cpu_materialization_contract(contract):

    table = _load_item_table()
    observations = _make_observations(contract, table.shape[0])
    npc_records, inventory_records, npc_codes, inventory_codes = _materialize(
        contract, observations, table,
    )
    assert npc_records.shape == (BATCH, contract.npc_count, contract.npc_features)
    assert inventory_records.shape == (BATCH, contract.inventory_count, contract.inventory_features)
    assert set(npc_codes.ravel()) == set(range(contract.type_count + 1))
    assert set(inventory_codes.ravel()) == set(range(table.shape[0]))
    assert np.count_nonzero(npc_records[npc_codes == 0]) == 0
    assert np.count_nonzero(inventory_records[inventory_codes == 0]) == 0
    assert BATCH * contract.npc_count * contract.npc_features > 256
    assert BATCH * contract.npc_count * contract.npc_features % 256 != 0
    assert BATCH * contract.inventory_count * contract.inventory_features > 256
    assert BATCH * contract.inventory_count * contract.inventory_features % 256 != 0
    actual = _source_contract(contract)
    expected = _expected_source_contract(contract)
    stale = {name: (actual[name], value) for name, value in expected.items() if actual[name] != value}
    assert not stale, f"{contract.name} entity encoder contract is stale: {stale}"


def _torch_reference(torch, functional, contract, observations, table, weights):
    npc_records, inventory_records, _, _ = _materialize(
        contract, observations.detach().cpu().numpy(), table,
    )
    npc = torch.from_numpy(npc_records).to(observations.device)
    inventory = torch.from_numpy(inventory_records).to(observations.device)
    global_w, entity_l1_w, entity_l2_w, inventory_l1_w, inventory_l2_w = weights

    def branch(records, l1_weight, l2_weight, active_width):
        hidden = functional.gelu(functional.linear(records, l1_weight), approximate="tanh")
        values = functional.linear(hidden, l2_weight)
        active = records[:, :, :active_width].sum(dim=2) > 0
        pooled = values.masked_fill(~active.unsqueeze(2), -torch.inf).amax(dim=1)
        return torch.where(active.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))

    return (
        functional.linear(observations, global_w)
        + branch(npc, entity_l1_w, entity_l2_w, contract.type_count)
        + branch(inventory, inventory_l1_w, inventory_l2_w, 1)
    )


def _configure_library(path: Path):
    library = ctypes.CDLL(path)
    pointer = ctypes.c_void_p
    library.osrs_entity_test_contract.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    library.osrs_entity_test_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    library.osrs_entity_test_init.restype = ctypes.c_int
    library.osrs_entity_test_set_weights.argtypes = [pointer] * 5
    library.osrs_entity_test_forward.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int]
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
    library_path = tmp_path_factory.mktemp("osrs_entity_encoder") / "osrs_entity_encoder.so"
    subprocess.run(
        [
            nvcc, "-shared", "-o", str(library_path), str(CUDA_SOURCE),
            "-I", str(ROOT / "src"), "-I", str(raylib_include),
            "-Xlinker", "--no-as-needed",
            "-lcublas", "-lcudnn", "-lcurand", "-lnccl", "-lnvidia-ml", "-lcusolver",
            str(raylib_archive),
            "-lGL", "-lm", "-lpthread", "-ldl", "-lrt",
            "--compiler-options", "-fPIC", "-Xcompiler", "-O2",
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

    reported = (ctypes.c_int * 8)()
    cuda_library.osrs_entity_test_contract(contract.kind, reported)
    assert list(reported) == [
        contract.obs_size, contract.npc_start, contract.npc_count, contract.npc_features,
        contract.type_count, contract.inventory_start, contract.inventory_count,
        contract.inventory_features,
    ]
    assert cuda_library.osrs_entity_test_init(
        contract.kind, BATCH, contract.obs_size, HIDDEN,
    ) == 0

    table = _load_item_table()
    observations_np = _make_observations(contract, table.shape[0])
    observations = torch.from_numpy(observations_np).cuda()
    generator = torch.Generator(device="cuda").manual_seed(6100 + contract.kind)
    shapes = (
        (HIDDEN, contract.obs_size),
        (BOTTLENECK, contract.npc_features),
        (HIDDEN, BOTTLENECK),
        (BOTTLENECK, contract.inventory_features),
        (HIDDEN, BOTTLENECK),
    )
    weights = [
        (torch.randn(shape, generator=generator, device="cuda") * 0.05).requires_grad_()
        for shape in shapes
    ]
    cuda_library.osrs_entity_test_set_weights(*[_pointer(weight) for weight in weights])

    cuda_output = torch.empty(BATCH, HIDDEN, device="cuda")
    cuda_library.osrs_entity_test_forward(
        _pointer(cuda_output), _pointer(observations), BATCH, contract.obs_size,
    )
    reference_output = _torch_reference(
        torch, functional, contract, observations, table, weights,
    )
    torch.testing.assert_close(cuda_output, reference_output, atol=3e-4, rtol=3e-4)

    output_gradient = torch.randn(
        (BATCH, HIDDEN), generator=generator, device="cuda",
    )
    reference_output.backward(output_gradient)
    cuda_library.osrs_entity_test_backward(_pointer(output_gradient), BATCH, HIDDEN)
    torch.cuda.synchronize()

    for index, weight in enumerate(weights):
        cuda_gradient = torch.empty_like(weight)
        cuda_library.osrs_entity_test_get_grad(index, _pointer(cuda_gradient))
        torch.cuda.synchronize()
        torch.testing.assert_close(cuda_gradient, weight.grad, atol=5e-3, rtol=5e-3)
