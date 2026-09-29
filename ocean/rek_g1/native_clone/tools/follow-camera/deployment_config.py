"""Fixed fresh-viewer deployment contract for the r10 follow camera. Importing this file has no effects."""
from pathlib import Path
import copy,hashlib,json

HERE=Path(__file__).resolve().parent
BASE=Path('/home/spark-advantage/rek-training/rek-native-clone-20260927-r1')
BASELINE_RUN=BASE/'run-r8'
RUN=BASE/'run-r9'
TASK=Path('/home/spark-advantage/rek-training/rek-follow-camera-20260929-r1')
APP=TASK/'app-stage-r10/app'
APP_SHA='e8e52f788603b95deb0935fe9796a05ec8b13b5d780ebee2462ee35d37cc7ad7'
BINARY=TASK/'build-r1/rek-native-clone'
BINARY_SHA='dba48c455123a4fa7d31c34034102b3756dfd469c26963823eb8711c92b5661f'
LAUNCHER_SHA='0d411a7ae5e7a66d103d85a568b3d3b5ff4b356c62b60409afc49819bcbae570'
GUARDS_SHA='bad77fad2f9f2e8a99b6973078e96716e87aa35081a5fe30ec1635dd72a06300'
PASSIVE_SHA='c21796cf490a4f0c172a7454d65d6974e3ebbb95eb18bc8c4b6948f80cfec52a'
RESOURCE_SHA='5f52525e62e4a955c7e41198cc2b0fd2e36045769233fdf652c9100987edb4ce'
# run-r8 (18774, r9 app, graph on) as prepared; r10 changes only the presentation path.
BASE_PINS={'worker.json':'60ce3d4284a9a5f82baaeab6f5b63e3d4014e00effac3b7edc14d2f2df56448b',
           'server.json':'534b77f5b24648915df6e0dca030e86cbff7894eaee554485125d7913a3c23e6',
           'identity.json':'468f75c3321fc921007996abba1d7839d6fee63a41567ac1c2c08e3e7dcb24dd'}
PORT=18775
CPUS='5,6,7,8,9,15,16,17,18,19'
# Frames are recorded without an app-side cap (~0.55 MB each, ~20/s while playing).
# disk_cap.py stops this viewer's own session before run-r9 exceeds this size.
RUN_CAP_BYTES=60*1024**3
DISK_CAP_FRACTION=.78

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def guards():
    assert sha(HERE/'human-viewers.json')==GUARDS_SHA
    return json.loads((HERE/'human-viewers.json').read_text())

def worker_template(baseline):
    assert baseline['arenas']==1 and baseline['cuda_graph_step'] is True
    return copy.deepcopy(baseline)

def validate_worker(worker,baseline):
    expected=worker_template(baseline)
    expected['render_model_path']=str(APP/'presentation/assets/presentation.playable.xml')
    assert worker==expected,'Unexpected worker change beyond r10 presentation path'

def verify_package():
    assert sha(HERE/'launch_viewer.py')==LAUNCHER_SHA
    assert sha(HERE/'passive.py')==PASSIVE_SHA
    assert sha(HERE/'resource_watch.py')==RESOURCE_SHA
    guards()
    for name,digest in BASE_PINS.items():assert sha(HERE/'baseline'/name)==digest

def validate_prepared(receipt,run=RUN):
    assert receipt['schema']=='rek.follow_camera_viewer_prepared.v1'
    assert receipt['run']==str(RUN) and receipt['app']==str(APP) and receipt['port']==PORT
    assert receipt['app_manifest_sha256']==APP_SHA and receipt['binary_sha256']==BINARY_SHA
    assert sha(APP/'SOURCE-MANIFEST.json')==APP_SHA
    assert sha(BINARY)==BINARY_SHA
    for name,digest in receipt['prepared_files'].items():
        assert name in ('worker.json','server.json','identity.json','league.json')
        assert sha(run/name)==digest
    assert set(receipt['prepared_files'])=={'worker.json','server.json','identity.json','league.json'}
    baseline=json.loads((HERE/'baseline/worker.json').read_text())
    validate_worker(json.loads((run/'worker.json').read_text()),baseline)
