"""Create an explicitly approximate export using Puffysics-supported capsules."""
import argparse
import hashlib
import json
from pathlib import Path


def convert(source, output):
    raw = source.read_bytes()
    export = json.loads(raw)
    changed = []
    for shape in export["shapes"]:
        if shape["kind"] == "cylinder":
            changed.append(shape["source_geom_id"])
            shape["kind"] = "capsule"
    if not changed:
        raise ValueError("source export contains no cylinders")
    export["diagnostic_geometry_substitution"] = {
        "name": "cylinders_to_capsules_same_radius_and_centerline_half_length",
        "source_export_sha256": hashlib.sha256(raw).hexdigest(),
        "changed_source_geom_ids": changed,
        "parity_claim": False,
        "geometry_effect": "round ends extend the axial bound by radius at each end",
        "masses_and_inertias": "unchanged from source, not recomputed for capsules",
        "purpose": "exclude the locally added cylinder GJK/EPA path from a performance diagnostic",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(export, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(export["diagnostic_geometry_substitution"], sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.out)
