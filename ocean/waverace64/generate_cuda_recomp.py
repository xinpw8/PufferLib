#!/usr/bin/env python3
"""Generate the device-callable Wave Race race-frame recompilation closure."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys


FUNCTION_START = re.compile(
    r"RECOMP_FUNC\s+void\s+(\w+)\s*\([^)]*\)\s*\{")
DIRECT_CALL = re.compile(r"\b(\w+)\s*\(rdram\s*,\s*ctx\s*\)")
LOOKUP_CALL = re.compile(
    r"LOOKUP_FUNC\(([^)]+)\)\s*\(rdram\s*,\s*ctx\s*\)")
TABLE_ENTRY = re.compile(
    r"\{0x([0-9A-Fa-f]+)u,\s*(-?\d+),\s*(\w+)\}")
JUMP_ADDEND = re.compile(
    r"^(\s*)gpr\s+(jr_addend_[0-9A-Fa-f]+)\s*=\s*([^;]+);",
    re.MULTILINE,
)

DEFAULT_ROOTS = (
    "func_800922E4",
    # The race root updates physics, water, checkpoints and official finish
    # data. This state-only overlay function commits the resulting terminal
    # game-state transition before the adapter reads it. The surrounding
    # func_80092CF0 display chain is intentionally excluded.
    "func_i1_802C5DF4",
)

# This generator is deliberately pinned to the same cartridge revision as the
# native recompile. Indirect targets live in immutable cartridge function
# records, so the cartridge is an input to the static call-graph proof rather
# than a source of runtime coverage samples.
PINNED_ROM_SHA256 = (
    "f35d2423ebcb86eaf86fa935b613c753"
    "2b123a7bc50fb74996984c3b02fc3999"
)

MAIN_SEGMENT_VRAM_START = 0x80046850
MAIN_SEGMENT_ROM_START = 0x00001050
CODES_SEGMENT_VRAM_START = 0x801DAFA0
CODES_SEGMENT_ROM_START = 0x000A95D0

# func_8009C2CC indexes eight-byte animation records beginning at these fake
# symbols. Together they span one contiguous, immutable 44-record table. The
# bounds come from the US Rev 1 main-segment data layout, not from an observed
# execution trace.
ANIMATION_RECORDS_VRAM_START = 0x800E60DC
ANIMATION_RECORDS_VRAM_END = 0x800E623C
ANIMATION_RECORD_SIZE = 8
ANIMATION_RECORD_BASES = (
    0x800E60DC,
    0x800E60E4,
    0x800E60EC,
    0x800E60F4,
    0x800E60FC,
    0x800E6104,
    0x800E6114,
    0x800E6124,
    0x800E618C,
    0x800E61A4,
    0x800E61CC,
    0x800E61E4,
    0x800E61FC,
    0x800E6214,
    0x800E622C,
)

# Every value ever passed to func_801DFD94 from the retained closure. The first
# eleven are the finite branches in func_801DDAB8; the final six are the finite
# branches in func_801DDEDC. Each is a cartridge-resident array of 12-byte
# records terminated by -1 or -2 in word zero. Word one is the optional
# callback consumed by func_801DFCB8.
SCRIPT_RECORD_ROOTS = (
    0x80223A30,
    0x80223D38,
    0x80224944,
    0x80224B14,
    0x80223F7C,
    0x80224060,
    0x802241E0,
    0x802242C0,
    0x80224430,
    0x802245B4,
    0x802246B4,
    0x80224D9C,
    0x80224F98,
    0x8022512C,
    0x802252B4,
    0x802254F4,
    0x80225804,
)
SCRIPT_RECORD_SIZE = 12
SCRIPT_RECORD_LIMIT = 512
SCRIPT_DATA_VRAM_END = 0x80225900

# These are the only variable LOOKUP_FUNC sites reachable after closing over
# all cartridge records and literal callbacks. Generation fails if a future
# runtime introduces another site, so an unknown pointer source cannot silently
# weaken the proof.
PROVED_VARIABLE_LOOKUPS = (
    "func_8009A460",
    "func_8009C2CC",
    "func_801DD6B4",
    "func_801DFCB8",
)

REGISTER_WRITE = re.compile(r"ctx->r(\d+)\s*=")
REGISTER_LUI = re.compile(
    r"ctx->r(\d+)\s*=\s*S32\(0X([0-9A-F]+)\s*<<\s*16\);",
    re.IGNORECASE,
)
REGISTER_ADD_SELF = re.compile(
    r"ctx->r(\d+)\s*=\s*ADD32\(ctx->r\1,\s*([-+]?0X[0-9A-F]+)\);",
    re.IGNORECASE,
)


def extract_functions(directory: pathlib.Path):
    functions = {}
    for path in sorted(directory.glob("funcs_*.c")):
        source = path.read_text(encoding="utf-8")
        for match in FUNCTION_START.finditer(source):
            depth = 1
            cursor = match.end()
            while cursor < len(source) and depth:
                char = source[cursor]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                cursor += 1
            if depth != 0:
                raise RuntimeError(f"unterminated function {match.group(1)} in {path}")
            functions[match.group(1)] = (path.name, source[match.start():cursor])
    return functions


def extract_function_table(path: pathlib.Path):
    entries = {}
    for address, overlay, name in TABLE_ENTRY.findall(path.read_text(encoding="utf-8")):
        entries.setdefault(int(address, 16), []).append((int(overlay), name))
    return entries


def resolve_target(entries, address: int, overlay: int):
    candidates = entries.get(address, ())
    for candidate_overlay, name in candidates:
        if candidate_overlay == overlay:
            return name
    for candidate_overlay, name in candidates:
        if candidate_overlay < 0:
            return name
    return candidates[0][1] if candidates else None


def read_pinned_rom(path: pathlib.Path):
    try:
        data = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"cannot read pinned US Rev 1 ROM {path}: {error}") from error
    digest = hashlib.sha256(data).hexdigest()
    if digest != PINNED_ROM_SHA256:
        raise RuntimeError(
            "CUDA closure requires the byte-exact Wave Race 64 USA Rev 1 ROM; "
            f"expected SHA-256 {PINNED_ROM_SHA256}, got {digest}")
    return data, digest


def rom_u32(rom: bytes, offset: int):
    if offset < 0 or offset + 4 > len(rom):
        raise RuntimeError(f"ROM read outside cartridge at 0x{offset:X}")
    return int.from_bytes(rom[offset:offset + 4], "big")


def main_segment_rom_offset(vram: int):
    return MAIN_SEGMENT_ROM_START + vram - MAIN_SEGMENT_VRAM_START


def codes_segment_rom_offset(vram: int):
    return CODES_SEGMENT_ROM_START + vram - CODES_SEGMENT_VRAM_START


def require_function_address(entries, address: int, overlay: int, source: str):
    target = resolve_target(entries, address, overlay)
    if not target:
        raise RuntimeError(
            f"{source} contains unmapped function address 0x{address:08X}")
    return target


def cartridge_animation_targets(rom: bytes, entries, overlay: int):
    targets = set()
    records = 0
    for vram in range(
            ANIMATION_RECORDS_VRAM_START,
            ANIMATION_RECORDS_VRAM_END,
            ANIMATION_RECORD_SIZE):
        address = rom_u32(rom, main_segment_rom_offset(vram))
        require_function_address(entries, address, overlay,
                                 f"animation record 0x{vram:08X}")
        targets.add(address)
        records += 1
    expected = ((ANIMATION_RECORDS_VRAM_END - ANIMATION_RECORDS_VRAM_START)
                // ANIMATION_RECORD_SIZE)
    if records != expected or records != 44:
        raise RuntimeError(f"expected 44 animation records, found {records}")
    return targets, records


def validate_animation_record_bases(functions):
    body = functions["func_8009C2CC"][1]
    constants = set()
    for register in range(32):
        constants |= register_constant_pairs(body, register)
    actual = {
        address for address in constants
        if ANIMATION_RECORDS_VRAM_START <= address < ANIMATION_RECORDS_VRAM_END
    }
    expected = set(ANIMATION_RECORD_BASES)
    if actual != expected:
        raise RuntimeError(
            "func_8009C2CC animation-record bases changed; expected "
            f"{', '.join(f'0x{x:08X}' for x in sorted(expected))}; got "
            f"{', '.join(f'0x{x:08X}' for x in sorted(actual))}")


def cartridge_script_targets(rom: bytes, entries, overlay: int):
    targets = set()
    record_count = 0
    for root in SCRIPT_RECORD_ROOTS:
        base = codes_segment_rom_offset(root)
        for index in range(SCRIPT_RECORD_LIMIT):
            offset = base + index * SCRIPT_RECORD_SIZE
            duration = rom_u32(rom, offset)
            callback = rom_u32(rom, offset + 4)
            record_count += 1
            if callback:
                require_function_address(
                    entries, callback, overlay,
                    f"script record 0x{root + index * SCRIPT_RECORD_SIZE:08X}")
                targets.add(callback)
            if duration in (0xFFFFFFFF, 0xFFFFFFFE):
                break
        else:
            raise RuntimeError(
                f"unterminated script records at 0x{root:08X} "
                f"(>{SCRIPT_RECORD_LIMIT} records)")
    return targets, record_count


def validate_script_record_roots(functions):
    constants = set()
    for name in ("func_801DDAB8", "func_801DDEDC"):
        body = functions[name][1]
        for register in range(32):
            constants |= register_constant_pairs(body, register)
    actual = {
        address for address in constants
        if SCRIPT_RECORD_ROOTS[0] <= address < SCRIPT_DATA_VRAM_END
    }
    expected = set(SCRIPT_RECORD_ROOTS)
    if actual != expected:
        raise RuntimeError(
            "script roots passed to func_801DFD94 changed; expected "
            f"{', '.join(f'0x{x:08X}' for x in sorted(expected))}; got "
            f"{', '.join(f'0x{x:08X}' for x in sorted(actual))}")


def register_constant_pairs(body: str, register: int):
    """Return every LUI/add-immediate constant built in one generated body."""
    high = None
    values = set()
    for line in body.splitlines():
        write = REGISTER_WRITE.search(line)
        if not write or int(write.group(1)) != register:
            continue
        lui = REGISTER_LUI.search(line)
        if lui:
            high = int(lui.group(2), 16) << 16
            continue
        add = REGISTER_ADD_SELF.search(line)
        if add and high is not None:
            values.add((high + int(add.group(2), 0)) & 0xFFFFFFFF)
            continue
        high = None
    return values


def literal_dispatch_targets(functions, entries, overlay: int):
    """Prove the finite jalr targets constructed by func_801DD6B4."""
    body = functions["func_801DD6B4"][1]
    candidates = register_constant_pairs(body, 3)
    targets = {address for address in candidates if address in entries}
    for address in targets:
        require_function_address(entries, address, overlay,
                                 "func_801DD6B4 literal target")
    if len(targets) != 9:
        formatted = ", ".join(f"0x{x:08X}" for x in sorted(targets))
        raise RuntimeError(
            "expected nine finite func_801DD6B4 targets, found "
            f"{len(targets)}: {formatted}")
    return targets


def constant_argument_before(body: str, end: int, register: int, caller: str):
    """Resolve a literal function-pointer argument at one generated callsite."""
    high = None
    addend = None
    for line in reversed(body[:end].splitlines()):
        write = REGISTER_WRITE.search(line)
        if not write or int(write.group(1)) != register:
            continue
        if addend is None:
            add = REGISTER_ADD_SELF.search(line)
            if add:
                addend = int(add.group(2), 0)
                continue
            lui = REGISTER_LUI.search(line)
            if lui:
                return (int(lui.group(2), 16) << 16) & 0xFFFFFFFF
            break
        lui = REGISTER_LUI.search(line)
        if lui:
            high = int(lui.group(2), 16) << 16
            break
        break
    if high is None or addend is None:
        raise RuntimeError(
            f"cannot prove func_8009A460 function pointer in {caller}")
    return (high + addend) & 0xFFFFFFFF


def helper_callback_targets(functions, entries, closure, overlay: int):
    targets = set()
    marker = re.compile(r"\bfunc_8009A460\s*\(rdram\s*,\s*ctx\s*\);")
    for caller in sorted(closure):
        body = functions[caller][1]
        for call in marker.finditer(body):
            address = constant_argument_before(body, call.start(), 4, caller)
            require_function_address(
                entries, address, overlay,
                f"func_8009A460 argument in {caller}")
            targets.add(address)
    return targets


def variable_lookup_callers(functions, closure):
    callers = set()
    for name in closure:
        for expression in LOOKUP_CALL.findall(functions[name][1]):
            expression = expression.strip()
            if not re.fullmatch(r"0[xX][0-9A-Fa-f]+", expression):
                callers.add(name)
    return callers


def section_addresses(path: pathlib.Path):
    source = path.read_text(encoding="utf-8")
    values = [int(value, 16) for value in re.findall(r"\(int32_t\)0x([0-9A-Fa-f]+)u", source)]
    if len(values) != 82:
        raise RuntimeError(f"expected 82 section addresses, found {len(values)}")
    return values


def make_graph(functions, entries, overlay: int):
    names = set(functions)
    graph = {}
    for name, (_, body) in functions.items():
        calls = (set(DIRECT_CALL.findall(body)) & names) - {name}
        for expression in LOOKUP_CALL.findall(body):
            expression = expression.strip()
            if re.fullmatch(r"0[xX][0-9A-Fa-f]+", expression):
                target = resolve_target(entries, int(expression, 0), overlay)
                if target in names:
                    calls.add(target)
        graph[name] = calls
    return graph


def transitive_closure(graph, roots):
    closure = set(roots)
    pending = list(roots)
    while pending:
        name = pending.pop()
        for target in graph.get(name, ()):
            if target not in closure:
                closure.add(target)
                pending.append(target)
    return closure


def hoist_jump_addends(body: str):
    declarations = []

    def replace(match):
        declarations.append(match.group(2))
        return f"{match.group(1)}{match.group(2)} = {match.group(3)};"

    body = JUMP_ADDEND.sub(replace, body)
    if declarations:
        opening = body.index("{") + 1
        decl = "\n    gpr " + ", ".join(declarations) + ";"
        body = body[:opening] + decl + body[opening:]
    return body


def transform_body(body: str, entries, overlay: int):
    body = body.replace("RECOMP_FUNC void", "__device__ __noinline__ void", 1)
    body = body.replace("get_cop1_cs()", "wr64_device_get_cop1_cs(ctx)")
    body = body.replace("set_cop1_cs(", "wr64_device_set_cop1_cs(ctx, ")

    def replace_lookup(match):
        expression = match.group(1).strip()
        if re.fullmatch(r"0[xX][0-9A-Fa-f]+", expression):
            target = resolve_target(entries, int(expression, 0), overlay)
            if target:
                return f"{target}(rdram, ctx)"
        return f"wr64_device_lookup((int32_t)({expression}), rdram, ctx)"

    body = LOOKUP_CALL.sub(replace_lookup, body)
    return hoist_jump_addends(body)


def generate(args):
    runtime = args.runtime.resolve()
    rom, rom_digest = read_pinned_rom(args.rom.resolve())
    functions = extract_functions(runtime / "RecompiledFuncs")
    entries = extract_function_table(runtime / "runtime" / "func_table.c")
    validate_animation_record_bases(functions)
    validate_script_record_roots(functions)
    graph = make_graph(functions, entries, args.overlay)

    roots = list(args.root or DEFAULT_ROOTS)
    missing = sorted(set(roots) - set(functions))
    if missing:
        raise RuntimeError(f"missing generated roots: {', '.join(missing)}")

    animation_targets, animation_records = cartridge_animation_targets(
        rom, entries, args.overlay)
    script_targets, script_records = cartridge_script_targets(
        rom, entries, args.overlay)
    literal_targets = literal_dispatch_targets(functions, entries, args.overlay)
    indirect_targets = animation_targets | script_targets | literal_targets

    helper_targets = set()
    while True:
        indirect_roots = []
        for address in sorted(indirect_targets):
            target = require_function_address(
                entries, address, args.overlay, "indirect closure")
            if target not in functions:
                raise RuntimeError(
                    f"indirect target 0x{address:08X} ({target}) has no "
                    "generated function body")
            indirect_roots.append(target)
        closure = transitive_closure(graph, roots + indirect_roots)
        discovered = helper_callback_targets(
            functions, entries, closure, args.overlay)
        new_targets = discovered - indirect_targets
        helper_targets |= discovered
        if not new_targets:
            break
        indirect_targets |= new_targets

    actual_variable_callers = variable_lookup_callers(functions, closure)
    expected_variable_callers = set(PROVED_VARIABLE_LOOKUPS)
    if actual_variable_callers != expected_variable_callers:
        missing_callers = sorted(expected_variable_callers - actual_variable_callers)
        unknown_callers = sorted(actual_variable_callers - expected_variable_callers)
        details = []
        if missing_callers:
            details.append("missing expected sites: " + ", ".join(missing_callers))
        if unknown_callers:
            details.append("unproved sites: " + ", ".join(unknown_callers))
        raise RuntimeError("variable indirect-call proof changed; " + "; ".join(details))

    dispatch_names = {
        require_function_address(entries, address, args.overlay,
                                 "indirect dispatch")
        for address in indirect_targets
    }

    external_calls = set()
    for name in closure:
        for target in DIRECT_CALL.findall(functions[name][1]):
            if target not in functions:
                external_calls.add(target)

    selected_addresses = {}
    for address, candidates in entries.items():
        target = resolve_target(entries, address, args.overlay)
        if target in closure or target in external_calls or target in dispatch_names:
            selected_addresses[address] = target

    addresses = section_addresses(runtime / "runtime" / "section_table.c")
    lines = [
        "// Generated by ocean/waverace64/generate_cuda_recomp.py.",
        f"// Runtime: {runtime}",
        f"// Cartridge SHA-256: {rom_digest}",
        f"// Overlay: {args.overlay}",
        f"// Functions: {len(closure)}",
        ("// Exhaustive variable targets: "
         f"{len(indirect_targets)} "
         f"({animation_records} animation records/{len(animation_targets)} unique, "
         f"{script_records} script records/{len(script_targets)} unique, "
         f"{len(literal_targets)} finite literals, "
         f"{len(helper_targets)} helper callbacks)"),
        "",
        "static __device__ __constant__ int32_t wr64_device_section_addresses[82] = {",
    ]
    for index in range(0, len(addresses), 6):
        chunk = addresses[index:index + 6]
        lines.append("    " + ", ".join(f"(int32_t)0x{x:08X}u" for x in chunk) + ",")
    lines.extend(["};", ""])

    for name in sorted(closure):
        lines.append(f"__device__ void {name}(uint8_t* rdram, recomp_context* ctx);")
    for name in sorted(external_calls | dispatch_names):
        if name not in closure:
            lines.append(f"__device__ void {name}(uint8_t* rdram, recomp_context* ctx);")
    lines.append("")

    for name in sorted(closure):
        filename, body = functions[name]
        lines.append(f"// Source: {filename}")
        lines.append(transform_body(body, entries, args.overlay))
        lines.append("")

    lines.append("__device__ void wr64_device_lookup(")
    lines.append("        int32_t vram, uint8_t* rdram, recomp_context* ctx) {")
    lines.append("    switch ((uint32_t)vram) {")
    for address, target in sorted(selected_addresses.items()):
        lines.append(f"        case 0x{address:08X}u: {target}(rdram, ctx); return;")
    lines.extend([
        "        default:",
        "            ctx->machine->error = 1;",
        "            ctx->machine->indirect_target = (uint32_t)vram;",
        "            return;",
        "    }",
        "}",
        "",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"generated {len(closure)} functions, "
        f"{len(external_calls)} external shims, "
        f"{len(selected_addresses)} dispatch targets, "
        f"{len(indirect_targets)} exhaustive variable targets -> {args.output}")


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=pathlib.Path)
    parser.add_argument(
        "--rom", required=True, type=pathlib.Path,
        help="byte-exact Wave Race 64 USA Rev 1 cartridge image")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--overlay", type=int, default=1)
    parser.add_argument("--root", action="append")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        generate(parse_args(sys.argv[1:]))
    except Exception as error:
        print(f"generate_cuda_recomp.py: {error}", file=sys.stderr)
        raise
