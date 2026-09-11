#!/usr/bin/env python3
"""Generate a dynamics-only MJCF from measured REK Unity/MuJoCo assets.

The input is produced by ``mujoco_asset_probe.py``.  This tool reconstructs
only serialized MuJoCo components.  It does not infer game rules, rewards,
network authority, policy observations, or the runtime PD rewrite performed by
``REKApp.Robot``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HIERARCHICAL_CLASSES = {
    "MjBody",
    "MjFreeJoint",
    "MjGeom",
    "MjHingeJoint",
    "MjInertial",
}

SHAPE_NAMES = {
    0: "sphere",
    1: "capsule",
    2: "ellipsoid",
    3: "cylinder",
    4: "box",
    5: "plane",
    6: "mesh",
    7: "hfield",
}

INTEGRATOR_NAMES = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}
CONE_NAMES = {0: "pyramidal", 1: "elliptic"}
JACOBIAN_NAMES = {0: "dense", 1: "sparse", 2: "auto"}
SOLVER_NAMES = {0: "PGS", 1: "CG", 2: "Newton"}

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w, x, y, z
Matrix = tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class WorldTransform:
    position: Vec3
    rotation: Quat
    lossy_scale: Vec3
    shear: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: float | int) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite MJCF value: {value}")
    if value == 0:
        value = 0.0
    return format(value, ".17g")


def vector_text(values: Iterable[float]) -> str:
    return " ".join(number(value) for value in values)


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def vec(record: dict[str, Any]) -> Vec3:
    return (float(record["x"]), float(record["y"]), float(record["z"]))


def quat(record: dict[str, Any]) -> Quat:
    return quat_normalize((
        float(record["w"]),
        float(record["x"]),
        float(record["y"]),
        float(record["z"]),
    ))


def quat_normalize(value: Quat) -> Quat:
    norm = math.sqrt(sum(component * component for component in value))
    if norm <= 1e-12:
        raise ValueError("zero-length quaternion")
    return tuple(component / norm for component in value)  # type: ignore[return-value]


def quat_multiply(lhs: Quat, rhs: Quat) -> Quat:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return quat_normalize((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ))


def quat_inverse(value: Quat) -> Quat:
    w, x, y, z = value
    return (w, -x, -y, -z)


def quat_rotate(rotation: Quat, value: Vec3) -> Vec3:
    w, x, y, z = rotation
    vx, vy, vz = value
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quat_matrix(rotation: Quat) -> tuple[tuple[float, float, float], ...]:
    w, x, y, z = rotation
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def identity_matrix() -> Matrix:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def trs_matrix(position: Vec3, rotation: Quat, scale: Vec3) -> Matrix:
    basis = quat_matrix(rotation)
    return (
        (basis[0][0] * scale[0], basis[0][1] * scale[1], basis[0][2] * scale[2], position[0]),
        (basis[1][0] * scale[0], basis[1][1] * scale[1], basis[1][2] * scale[2], position[1]),
        (basis[2][0] * scale[0], basis[2][1] * scale[1], basis[2][2] * scale[2], position[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def matrix_multiply(lhs: Matrix, rhs: Matrix) -> Matrix:
    return tuple(
        tuple(sum(lhs[row][index] * rhs[index][column] for index in range(4))
              for column in range(4))
        for row in range(4)
    )


def column(matrix: Matrix, index: int) -> Vec3:
    return (matrix[0][index], matrix[1][index], matrix[2][index])


def norm(value: Vec3) -> float:
    return math.sqrt(sum(component * component for component in value))


def dot(lhs: Vec3, rhs: Vec3) -> float:
    return sum(a * b for a, b in zip(lhs, rhs))


def world_transform(chain: list[dict[str, Any]]) -> WorldTransform:
    matrix = identity_matrix()
    rotation: Quat = (1.0, 0.0, 0.0, 0.0)
    for node in chain:
        local_position = vec(node["local_position"])
        local_rotation = quat(node["local_rotation"])
        local_scale = vec(node["local_scale"])
        matrix = matrix_multiply(matrix, trs_matrix(local_position, local_rotation, local_scale))
        rotation = quat_multiply(rotation, local_rotation)

    axes = [column(matrix, index) for index in range(3)]
    scales = tuple(norm(axis) for axis in axes)
    normalized = [tuple(component / scale for component in axis) if scale else (0.0, 0.0, 0.0)
                  for axis, scale in zip(axes, scales)]
    shear = max(abs(dot(normalized[a], normalized[b])) for a, b in ((0, 1), (0, 2), (1, 2)))
    return WorldTransform(
        position=(matrix[0][3], matrix[1][3], matrix[2][3]),
        rotation=rotation,
        lossy_scale=scales,  # Unity shapes consume transform.lossyScale.
        shear=shear,
    )


def mj_vector(unity: Vec3) -> Vec3:
    return (unity[0], unity[2], unity[1])


def mj_quaternion(unity: Quat) -> Quat:
    w, x, y, z = unity
    return (-w, x, z, y)


def relative_transform(child: WorldTransform, parent: WorldTransform | None) -> tuple[Vec3, Quat]:
    if parent is None:
        return child.position, child.rotation
    inverse = quat_inverse(parent.rotation)
    delta = tuple(a - b for a, b in zip(child.position, parent.position))
    return quat_rotate(inverse, delta), quat_multiply(inverse, child.rotation)


def hierarchy_path(record: dict[str, Any]) -> str:
    return str(record["hierarchy"]["game_object_path"])


def ancestor_path(path: str, candidate: str) -> bool:
    return path.startswith(candidate + "/")


def safe_name(record: dict[str, Any]) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record.get("owner") or record["class"]))
    return f"{base}_{record['path_id']}"


def enabled_record(record: dict[str, Any]) -> bool:
    hierarchy = record.get("hierarchy")
    return bool(record.get("enabled")) and hierarchy is not None and all(
        bool(node.get("active", True)) for node in hierarchy.get("transform_chain", ())
    )


def find_parent_body(
    record: dict[str, Any],
    bodies: list[dict[str, Any]],
) -> dict[str, Any] | None:
    path = hierarchy_path(record)
    candidates = [body for body in bodies
                  if body is not record and ancestor_path(path, hierarchy_path(body))]
    return max(candidates, key=lambda body: len(hierarchy_path(body)), default=None)


def set_transform_attributes(element: ET.Element, child: WorldTransform,
                             parent: WorldTransform | None) -> None:
    position, rotation = relative_transform(child, parent)
    element.set("pos", vector_text(mj_vector(position)))
    element.set("quat", vector_text(mj_quaternion(rotation)))


def set_joint_settings(element: ET.Element, settings: dict[str, Any]) -> None:
    spring = settings["Spring"]
    solver = settings["Solver"]
    element.set("armature", number(settings["Armature"]))
    element.set("springref", number(spring["EquilibriumPose"]))
    element.set("springdamper", vector_text((spring["TimeConstant"], spring["DampingRatio"])))
    element.set("damping", number(spring["Damping"]))
    element.set("stiffness", number(spring["Stiffness"]))
    element.set("solreflimit", vector_text((solver["RefLimit"]["TimeConst"],
                                             solver["RefLimit"]["DampRatio"])))
    element.set("solimplimit", vector_text((
        solver["ImpLimit"]["DMin"], solver["ImpLimit"]["DMax"],
        solver["ImpLimit"]["Width"], solver["ImpLimit"]["Midpoint"],
        solver["ImpLimit"]["Power"],
    )))
    element.set("solreffriction", vector_text((solver["RefFriction"]["TimeConst"],
                                                solver["RefFriction"]["DampRatio"])))
    element.set("solimpfriction", vector_text((
        solver["ImpFriction"]["DMin"], solver["ImpFriction"]["DMax"],
        solver["ImpFriction"]["Width"], solver["ImpFriction"]["Midpoint"],
        solver["ImpFriction"]["Power"],
    )))
    element.set("frictionloss", number(solver["FrictionLoss"]))
    element.set("limited", bool_text(solver["Limited"]))
    element.set("margin", number(solver["Margin"]))


def set_geom_settings(element: ET.Element, settings: dict[str, Any]) -> None:
    filtering = settings["Filtering"]
    solver = settings["Solver"]
    friction = settings["Friction"]
    fluid = settings["FluidCoefficients"]
    element.set("priority", str(int(settings["Priority"])))
    element.set("contype", str(int(filtering["Contype"])))
    element.set("conaffinity", str(int(filtering["Conaffinity"])))
    element.set("group", str(int(filtering["Group"])))
    element.set("condim", str(int(solver["ConDim"])))
    element.set("solmix", number(solver["SolMix"]))
    element.set("solref", vector_text((solver["SolRef"]["TimeConst"],
                                        solver["SolRef"]["DampRatio"])))
    element.set("solimp", vector_text((solver["SolImp"]["DMin"],
                                        solver["SolImp"]["DMax"],
                                        solver["SolImp"]["Width"])))
    element.set("margin", number(solver["Margin"]))
    element.set("gap", number(solver["Gap"]))
    element.set("friction", vector_text((friction["Sliding"], friction["Torsional"],
                                          friction["Rolling"])))
    element.set("fluidshape", {0: "none", 1: "ellipsoid"}[int(settings["FluidShapeType"])] )
    element.set("fluidcoef", vector_text((fluid["BluntDrag"], fluid["SlenderDrag"],
                                           fluid["AngularDrag"], fluid["KuttaLift"],
                                           fluid["MagnusLift"])))


def shape_size(values: dict[str, Any], transform: WorldTransform) -> tuple[str, Vec3 | tuple[float, ...]]:
    shape_type = int(values["ShapeType"])
    shape_name = SHAPE_NAMES.get(shape_type)
    if shape_name is None:
        raise ValueError(f"unsupported shape type: {shape_type}")
    sx, sy, sz = transform.lossy_scale
    if shape_type == 0:
        return shape_name, (float(values["Sphere"]["Radius"]) * sx,)
    if shape_type == 1:
        return shape_name, (
            float(values["Capsule"]["Radius"]) * sx,
            float(values["Capsule"]["HalfHeight"]) * sy,
        )
    if shape_type == 2:
        radiuses = vec(values["Ellipsoid"]["Radiuses"])
        return shape_name, mj_vector(tuple(a * b for a, b in zip(radiuses, (sx, sy, sz))))
    if shape_type == 3:
        return shape_name, (
            float(values["Cylinder"]["Radius"]) * sx,
            float(values["Cylinder"]["HalfHeight"]) * sy,
        )
    if shape_type == 4:
        extents = vec(values["Box"]["Extents"])
        return shape_name, mj_vector(tuple(a * b for a, b in zip(extents, (sx, sy, sz))))
    if shape_type == 5:
        extents = values["Plane"]["Extents"]
        return shape_name, (float(extents["x"]) * sx, float(extents["y"]) * sz, 0.1)
    raise ValueError(f"derived MJCF intentionally rejects non-primitive shape {shape_name}")


def make_component_element(
    record: dict[str, Any],
    world: dict[tuple[str, int], WorldTransform],
    parent_body: dict[str, Any] | None,
) -> ET.Element:
    class_name = record["class"]
    values = record["values"]
    key = (record["container"], int(record["path_id"]))
    child_transform = world[key]
    parent_transform = None
    if parent_body is not None:
        parent_key = (parent_body["container"], int(parent_body["path_id"]))
        parent_transform = world[parent_key]

    if class_name == "MjBody":
        element = ET.Element("body", name=safe_name(record))
        set_transform_attributes(element, child_transform, parent_transform)
        element.set("gravcomp", number(values["GravityCompensation"]))
        return element
    if class_name == "MjInertial":
        element = ET.Element("inertial")
        set_transform_attributes(element, child_transform, parent_transform)
        element.set("mass", number(values["Mass"]))
        element.set("diaginertia", vector_text(mj_vector(vec(values["DiagInertia"]))))
        return element
    if class_name == "MjHingeJoint":
        element = ET.Element("joint", name=safe_name(record), type="hinge")
        position, rotation = relative_transform(child_transform, parent_transform)
        axis = quat_rotate(rotation, (1.0, 0.0, 0.0))
        axis_norm = norm(axis)
        axis = tuple(component / axis_norm for component in axis)
        element.set("pos", vector_text(mj_vector(position)))
        element.set("axis", vector_text(mj_vector(axis)))
        set_joint_settings(element, values["Settings"])
        if float(values["RangeLower"]) > float(values["RangeUpper"]):
            raise ValueError(f"reversed joint range: {hierarchy_path(record)}")
        element.set("range", vector_text((values["RangeLower"], values["RangeUpper"])))
        element.set("ref", number(values["Configuration"]))
        return element
    if class_name == "MjFreeJoint":
        return ET.Element("freejoint", name=safe_name(record))
    if class_name == "MjGeom":
        element = ET.Element("geom", name=safe_name(record))
        if float(values["Mass"]) > 0:
            element.set("mass", number(values["Mass"]))
        else:
            element.set("density", number(values["Density"]))
        shape_name, size = shape_size(values, child_transform)
        element.set("type", shape_name)
        element.set("size", vector_text(size))
        set_transform_attributes(element, child_transform, parent_transform)
        set_geom_settings(element, values["Settings"])
        return element
    raise ValueError(f"unsupported hierarchical component: {class_name}")


def add_global_configuration(root: ET.Element, settings: dict[str, Any],
                             static_survey: dict[str, Any]) -> dict[str, Any]:
    time_values = static_survey["settings"]["TimeManager"]["values"]
    physics_values = static_survey["settings"]["PhysicsManager"]["values"]
    timestep = (
        float(time_values["Fixed Timestep.m_Count"])
        * float(time_values["Fixed Timestep.m_Rate.m_Denominator"])
        / float(time_values["Fixed Timestep.m_Rate.m_Numerator"])
    )
    unity_gravity = (
        float(physics_values["m_Gravity.x"]),
        float(physics_values["m_Gravity.y"]),
        float(physics_values["m_Gravity.z"]),
    )

    ET.SubElement(root, "compiler", coordinate="local")
    options = settings["GlobalOptions"]
    option = ET.SubElement(root, "option")
    option.set("timestep", number(timestep))
    option.set("gravity", vector_text(mj_vector(unity_gravity)))
    option.set("impratio", number(options["ImpRatio"]))
    option.set("magnetic", vector_text(vec(options["Magnetic"])))
    option.set("wind", vector_text(vec(options["Wind"])))
    option.set("density", number(options["Density"]))
    option.set("viscosity", number(options["Viscosity"]))
    option.set("o_margin", number(options["OverrideMargin"]))
    option.set("o_solref", vector_text((options["OverrideSolRef"]["TimeConst"],
                                         options["OverrideSolRef"]["DampRatio"])))
    option.set("o_solimp", vector_text((
        options["OverrideSolImp"]["DMin"], options["OverrideSolImp"]["DMax"],
        options["OverrideSolImp"]["Width"], options["OverrideSolImp"]["Midpoint"],
        options["OverrideSolImp"]["Power"],
    )))
    option.set("integrator", INTEGRATOR_NAMES[int(options["Integrator"])])
    option.set("cone", CONE_NAMES[int(options["Cone"])])
    option.set("jacobian", JACOBIAN_NAMES[int(options["Jacobian"])])
    option.set("solver", SOLVER_NAMES[int(options["Solver"])])
    option.set("iterations", str(int(options["Iterations"])))
    option.set("tolerance", number(options["Tolerance"]))
    option.set("noslip_iterations", str(int(options["NoSlipIterations"])))
    option.set("noslip_tolerance", number(options["NoSlipTolerance"]))
    option.set("ccd_iterations", str(int(options["CcdIterations"])))
    option.set("ccd_tolerance", number(options["CcdTolerance"]))
    flag = ET.SubElement(option, "flag")
    for source_name, attribute in (
        ("Constraint", "constraint"), ("Equality", "equality"),
        ("FrictionLoss", "frictionloss"), ("Limit", "limit"),
        ("Contact", "contact"), ("Spring", "spring"), ("Damper", "damper"),
        ("Gravity", "gravity"), ("ClampCtrl", "clampctrl"),
        ("WarmStart", "warmstart"), ("FilterParent", "filterparent"),
        ("Actuation", "actuation"), ("RefSafe", "refsafe"),
        ("Override", "override"), ("Energy", "energy"), ("FwdInv", "fwdinv"),
        ("MultiCCD", "multiccd"),
    ):
        flag.set(attribute, {0: "enable", 1: "disable"}[int(options["Flag"][source_name])])
    ET.SubElement(root, "size", memory=str(settings["GlobalSizes"]["Memory"]))
    custom = ET.SubElement(root, "custom")
    for numeric in settings.get("CustomNumeric", ()):
        ET.SubElement(custom, "numeric", name=str(numeric["Name"]), data=str(numeric["Data"]))
    return {"timestep": timestep, "unity_gravity": unity_gravity,
            "mujoco_gravity": mj_vector(unity_gravity)}


def component_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    chain = record["hierarchy"]["transform_chain"]
    sibling_path = tuple(-1 if node.get("sibling_index") is None else int(node["sibling_index"])
                         for node in chain)
    return sibling_path, len(chain), int(record["path_id"])


def generate(
    probe_path: Path,
    static_survey_path: Path,
    root_name: str,
    container: str | None,
    settings_container: str,
) -> tuple[ET.ElementTree, dict[str, Any]]:
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    static_survey = json.loads(static_survey_path.read_text(encoding="utf-8"))
    if probe.get("schema") != "rek.mujoco_asset_probe.v1":
        raise ValueError("unsupported probe schema")
    if probe["build_fingerprint"] != static_survey["build_fingerprint"]:
        raise ValueError("probe and static survey build fingerprints differ")

    selected = []
    for record in probe["targets"]:
        hierarchy = record.get("hierarchy")
        chain = hierarchy.get("transform_chain") if hierarchy else None
        if not chain or chain[0]["name"] != root_name:
            continue
        if container is not None and record["container"] != container:
            continue
        if record["class"] not in HIERARCHICAL_CLASSES | {"MjActuator"}:
            continue
        if not record["parsed"]:
            raise ValueError(f"unparsed required component: {record['class']}:{record['path_id']}")
        selected.append(record)
    if not selected:
        raise ValueError(f"no MuJoCo components found below root {root_name!r}")

    skipped = [record for record in selected if not enabled_record(record)]
    active = [record for record in selected if enabled_record(record)]
    bodies = sorted((record for record in active if record["class"] == "MjBody"),
                    key=component_sort_key)
    actuators = sorted((record for record in active if record["class"] == "MjActuator"),
                       key=component_sort_key)
    hierarchy_records = sorted((record for record in active
                                if record["class"] in HIERARCHICAL_CLASSES),
                               key=component_sort_key)
    if not bodies:
        raise ValueError("plant has no MjBody")

    world = {
        (record["container"], int(record["path_id"])):
            world_transform(record["hierarchy"]["transform_chain"])
        for record in active
    }
    settings_records = [record for record in probe["targets"]
                        if record["class"] == "MjGlobalSettings"
                        and record["container"] == settings_container and record["parsed"]]
    if len(settings_records) != 1:
        raise ValueError(f"expected one MjGlobalSettings in {settings_container}, got {len(settings_records)}")

    document_root = ET.Element("mujoco", model=f"rek_{safe_name(bodies[0])}")
    global_report = add_global_configuration(
        document_root, settings_records[0]["values"], static_survey)
    worldbody = ET.SubElement(document_root, "worldbody")

    elements: dict[tuple[str, int], ET.Element] = {}
    parent_bodies: dict[tuple[str, int], dict[str, Any] | None] = {}
    for record in hierarchy_records:
        key = (record["container"], int(record["path_id"]))
        parent_body = find_parent_body(record, bodies)
        parent_bodies[key] = parent_body
        element = make_component_element(record, world, parent_body)
        elements[key] = element
        if parent_body is None:
            worldbody.append(element)
        else:
            parent_key = (parent_body["container"], int(parent_body["path_id"]))
            elements[parent_key].append(element)

    actuator_section = ET.SubElement(document_root, "actuator")
    joints_by_pointer = {
        (record["container"], int(record["path_id"])): record
        for record in hierarchy_records if record["class"] == "MjHingeJoint"
    }
    unresolved = []
    for actuator in actuators:
        values = actuator["values"]
        if int(values["Type"]) != 1:
            raise ValueError(f"only measured motor actuators are supported: {hierarchy_path(actuator)}")
        pointer = values["Joint"]
        joint_key = (actuator["container"], int(pointer["m_PathID"]))
        joint = joints_by_pointer.get(joint_key)
        if joint is None:
            unresolved.append({"actuator": safe_name(actuator), "joint_pointer": pointer})
            continue
        common = values["CommonParams"]
        motor = ET.SubElement(actuator_section, "motor", name=safe_name(actuator),
                              joint=safe_name(joint))
        motor.set("ctrllimited", bool_text(common["CtrlLimited"]))
        motor.set("forcelimited", bool_text(common["ForceLimited"]))
        motor.set("ctrlrange", vector_text(sorted((common["CtrlRange"]["x"],
                                                   common["CtrlRange"]["y"]))))
        motor.set("forcerange", vector_text(sorted((common["ForceRange"]["x"],
                                                    common["ForceRange"]["y"]))))
        motor.set("lengthrange", vector_text(sorted((common["LengthRange"]["x"],
                                                     common["LengthRange"]["y"]))))
        motor.set("gear", vector_text(common["Gear"]))
    if unresolved:
        raise ValueError(f"unresolved actuator joints: {unresolved}")

    counts = {class_name: sum(record["class"] == class_name for record in active)
              for class_name in sorted(HIERARCHICAL_CLASSES | {"MjActuator"})}
    root_bodies = [body for body in bodies if find_parent_body(body, bodies) is None]
    free_joint_count = counts["MjFreeJoint"]
    hinge_count = counts["MjHingeJoint"]
    max_shear = max(transform.shear for transform in world.values())
    report = {
        "schema": "rek.mujoco_plant_generation.v1",
        "build_fingerprint": probe["build_fingerprint"],
        "root_name": root_name,
        "container": container,
        "probe_path": str(probe_path),
        "probe_sha256": sha256_file(probe_path),
        "static_survey_path": str(static_survey_path),
        "static_survey_sha256": sha256_file(static_survey_path),
        "counts": counts,
        "root_body_count": len(root_bodies),
        "skipped_disabled_count": len(skipped),
        "unresolved_actuator_count": 0,
        "predicted_model_dimensions": {
            "njnt": free_joint_count + hinge_count,
            "nu": counts["MjActuator"],
            "nq": 7 * free_joint_count + hinge_count,
            "nv": 6 * free_joint_count + hinge_count,
        },
        "max_transform_shear": max_shear,
        "global_configuration": global_report,
        "unity_mapping_source": {
            "repository": "https://github.com/google-deepmind/mujoco",
            "tag": "3.7.0",
            "commit": "72cb2b210da666617924de709406d6aadbe60c71",
            "path": "unity/Runtime/Tools/MjEngineTool.cs",
            "source_blob_sha1": "bf27a1d950099b11a38fcb83d3e03763a5592166",
            "unity_vector_to_mujoco": ["x", "z", "y"],
            "unity_quaternion_to_mujoco_wxyz": ["-w", "x", "z", "y"],
        },
        "control_equivalent": False,
        "limits": [
            "plant values are recovered from serialized client assets, not server runtime state",
            "REKApp.Robot runtime PD gains, force limits, and passive damping are not applied",
            "prefab-root transform is serialized asset state, not a measured arena spawn pose",
            "game controller, observations, rewards, contacts, damage, and opponent logic are absent",
            "Unity-to-MJCF mapping matches official MuJoCo tag 3.7.0; exact shipped managed-assembly equivalence is not established",
        ],
    }
    return ET.ElementTree(document_root), report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--static-survey", type=Path, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--container")
    parser.add_argument("--settings-container", default="level1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    tree, report = generate(
        args.probe.resolve(), args.static_survey.resolve(), args.root,
        args.container, args.settings_container)
    ET.indent(tree, space="  ")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.out, encoding="utf-8", xml_declaration=True)
    report["mjcf_path"] = str(args.out.resolve())
    report["mjcf_sha256"] = sha256_file(args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "schema", "build_fingerprint", "root_name", "container", "counts",
        "predicted_model_dimensions", "skipped_disabled_count", "max_transform_shear",
        "control_equivalent", "mjcf_sha256",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
