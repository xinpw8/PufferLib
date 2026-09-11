"""Compare complete CPU fixture outputs without changing acceptance bounds."""
import array
import hashlib
import json
import math
from pathlib import Path

base=Path(__file__).resolve().parent / "art-cache-cpu-results"
def read(name):
    values=array.array("f")
    values.frombytes((base / (name+".bin")).read_bytes())
    return values
original=read("original")
results={}
for name in ("uncached","cached"):
    actual=read(name)
    assert len(actual)==len(original)
    errors=[abs(a-b) for a,b in zip(actual,original)]
    results[name]=dict(floats=len(actual),different_numeric_values=sum(x!=0 for x in errors),
        max_absolute_error=max(errors),rms_error=math.sqrt(sum(x*x for x in errors)/len(errors)),
        bitwise_equal=(base/(name+".bin")).read_bytes()==(base/"original.bin").read_bytes())
report=dict(scope="CPU only; 12 randomized forests, 4-row signed RHS, velocity updates, two contact phases, position integration and full ABA",
            original_source_sha256="299dfcfcbe3de97252520b454c4db0eddc30ecb2c370163e91e5877fb13c8969",
            acceptance="Require bitwise equality to original for this bounded fixture. No GPU parity claim.",results=results)
report["hashes"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.glob("*.bin")}
(base/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
raise SystemExit(not all(v["bitwise_equal"] for v in results.values()))
