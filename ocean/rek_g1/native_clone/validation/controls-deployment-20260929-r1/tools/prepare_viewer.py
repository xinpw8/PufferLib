"""Prepare a fresh corrected viewer without changing existing human sessions."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('remote_task',r'C:\rekagent\work\rek-native-clone-20260927-r1\remote_task.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
BASE='/home/spark-advantage/rek-training/rek-native-clone-20260927-r1'
BINARY='/home/spark-advantage/rek-training/rek-playback-native-20260928-r1/build-r1/rek-native-clone'
BIN_SHA='ef519db4c8b3b3a6696ebc8dfd7b686bffc789a7b91f1ef8d60dfab3494e0975'

def main():
    p=argparse.ArgumentParser();p.add_argument('--app',required=True);p.add_argument('--manifest-sha256',required=True);a=p.parse_args()
    assert hashlib.sha256((HERE/'deployment/launch_viewer.py').read_bytes()).hexdigest()=='0d411a7ae5e7a66d103d85a568b3d3b5ff4b356c62b60409afc49819bcbae570'
    with remote.connect() as client:
        target=BASE+'/controls-deploy-20260929-r1'
        remote.upload_tree(client,HERE/'deployment',target)
        script='''from pathlib import Path
import hashlib,json,subprocess
base=Path(%r);target=Path(%r);app=Path(%r)
assert hashlib.sha256((app/'SOURCE-MANIFEST.json').read_bytes()).hexdigest()==%r
assert hashlib.sha256(Path(%r).read_bytes()).hexdigest()==%r
worker=json.loads((base/'run-r6/worker.json').read_text())
env=json.loads((base/'run-r6/server.json').read_text())['backends'][0]['env']
assert worker['arenas']==1 and worker['cuda_graph_step'] is False
for name,obj in [('worker-template.json',worker),('env.json',env)]:
    with (target/name).open('x') as f:f.write(json.dumps(obj,indent=2)+'\\n')
subprocess.run(['node',str(app/'prepare.cjs'),str(target/'worker-template.json'),%r,str(base/'run-r7'),str(target/'env.json'),'18773'],check=True)
'''%(BASE,target,a.app,a.manifest_sha256,BINARY,BIN_SHA,BINARY)
        assert remote.run(client,['python3','-c',script])==0
    receipt={'app':a.app,'app_manifest_sha256':a.manifest_sha256,'binary':BINARY,'binary_sha256':BIN_SHA,
             'run':BASE+'/run-r7','port':18773,'launcher':target+'/launch_viewer.py','guard':target+'/guarded_start.py'}
    with (HERE/'PREPARED.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
