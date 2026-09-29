"""Fixed fresh-viewer deployment contract. Importing this file has no effects."""
from pathlib import Path
import copy,hashlib,json

HERE=Path(__file__).resolve().parent
BASE=Path('/home/spark-advantage/rek-training/rek-native-clone-20260927-r1')
RUN=BASE/'run-r8'
APP=Path('/home/spark-advantage/rek-training/rek-performance-audit-20260929-r1/app-stage-r9/app')
APP_SHA='9138485e02c0c6ec69a8f84aad87f3469b046170e9db720433b2508a674481a2'
BINARY=Path('/home/spark-advantage/rek-training/rek-playback-native-20260928-r1/build-r1/rek-native-clone')
BINARY_SHA='ef519db4c8b3b3a6696ebc8dfd7b686bffc789a7b91f1ef8d60dfab3494e0975'
LAUNCHER_SHA='0d411a7ae5e7a66d103d85a568b3d3b5ff4b356c62b60409afc49819bcbae570'
GUARDS_SHA='91364fb89fe4a1a1f6803371b55318cc324cff6130a94a4c6d713c91817fb09f'
PASSIVE_SHA='c21796cf490a4f0c172a7454d65d6974e3ebbb95eb18bc8c4b6948f80cfec52a'
RESOURCE_SHA='5f52525e62e4a955c7e41198cc2b0fd2e36045769233fdf652c9100987edb4ce'
BASE_PINS={'worker.json':'5c9b7f8c482d46c64601384c7e95aa14a190b3bcf5abb0664fd7e7b47d955f83',
           'server.json':'1001434e139ba581dc58a7fd86930348365de56b9857a277a79c9058b9a04e79',
           'identity.json':'18680b32fa142e22dadd506b2d290d9ecdb55dd53cd11b2aad410851e54db4f1'}
PORT=18774
CPUS='5,6,7,8,9,15,16,17,18,19'

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def guards():
    assert sha(HERE/'human-viewers.json')==GUARDS_SHA
    return json.loads((HERE/'human-viewers.json').read_text())

def worker_for(mode,baseline):
    if mode not in ('on','off'):raise ValueError('Explicit graph decision must be on or off')
    assert baseline['arenas']==1 and baseline['cuda_graph_step'] is False
    worker=copy.deepcopy(baseline);worker['cuda_graph_step']=mode=='on'
    return worker

def validate_worker(worker,mode,baseline):
    expected=worker_for(mode,baseline)
    expected['render_model_path']=str(APP/'presentation/assets/presentation.playable.xml')
    assert worker==expected,'Unexpected worker change beyond graph mode and r9 presentation path'

def verify_package():
    assert sha(HERE/'launch_viewer.py')==LAUNCHER_SHA
    assert sha(HERE/'passive.py')==PASSIVE_SHA
    assert sha(HERE/'resource_watch.py')==RESOURCE_SHA
    guards()
    for name,digest in BASE_PINS.items():assert sha(HERE/'baseline'/name)==digest

def validate_prepared(receipt,run=RUN):
    assert receipt['schema']=='rek.performance_viewer_prepared.v1'
    assert receipt['run']==str(RUN) and receipt['app']==str(APP) and receipt['port']==PORT
    assert receipt['app_manifest_sha256']==APP_SHA and receipt['binary_sha256']==BINARY_SHA
    assert receipt['graph_mode'] in ('on','off')
    assert sha(APP/'SOURCE-MANIFEST.json')==APP_SHA
    assert sha(BINARY)==BINARY_SHA
    for name,digest in receipt['prepared_files'].items():
        assert name in ('worker.json','server.json','identity.json','league.json')
        assert sha(run/name)==digest
    assert set(receipt['prepared_files'])=={'worker.json','server.json','identity.json','league.json'}
    baseline=json.loads((HERE/'baseline/worker.json').read_text())
    validate_worker(json.loads((run/'worker.json').read_text()),receipt['graph_mode'],baseline)

