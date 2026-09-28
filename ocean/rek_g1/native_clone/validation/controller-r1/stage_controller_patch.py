"""Apply only bounded controller changes inside this task's private snapshot."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
G1=ROOT/'source-r1/ocean/rek_g1'
RESET=Path(r'C:\rekagent\work\rek-residual-parity-20260927-r1\motion-reset\source-r1')

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    expected={'sonic_motion_composer_native.c':'52fc10a2541e6bfa59c2a9571963fe1e78021580f760046204fbd768a8d0d0d2','sonic_motion_composer_native.h':'b05c38f2b3add435eb2cf9b1ed0a604b64dfc76b1ef828addd7efbd266fd0f22'}
    original={'sonic_motion_composer_native.c':'49ddd897fab689c4af4c0b63f66cbac1d96835294c6486d8304f39db0623e1a1','sonic_motion_composer_native.h':'2ccac0b9609f544fbd59d287abea111907149a7a7ed17f6ce11216557ec71154'}
    for name,h in expected.items():
        assert digest(RESET/name)==h
        assert digest(G1/name)==original[name],'current working source differs from validated reset base'
        (G1/name).write_bytes((RESET/name).read_bytes())
    print(json.dumps({'validated_Reset_extension_imported':expected,'working_repository_modified':False}))

if __name__=='__main__':main()
