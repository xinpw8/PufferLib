"""Prepare a pinned single-arena run; starts no process and sends no UI input."""
from pathlib import Path
import argparse, hashlib, importlib.util, json

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('remote_task',r'C:\rekagent\work\rek-native-clone-20260927-r1\remote_task.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
BASE='/home/spark-advantage/rek-training/rek-native-clone-20260927-r1'
BINARY='/home/spark-advantage/rek-training/rek-playback-native-20260928-r1/build-r1/rek-native-clone'
BIN_SHA='ef519db4c8b3b3a6696ebc8dfd7b686bffc789a7b91f1ef8d60dfab3494e0975'
ENC='/home/spark-advantage/codexrook-runtime/generated/gear-sonic-batch2-20260908T2314Z/model_encoder.batch2.onnx'
DEC='/home/spark-advantage/codexrook-runtime/generated/gear-sonic-batch2-20260908T2314Z/model_decoder.batch2.onnx'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--app',required=True);parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args()
    launcher=Path(r'C:\rekagent\work\rek-playback-speed-20260928-r1\launch_viewer.py').read_bytes()
    assert hashlib.sha256(launcher).hexdigest()=='e8ea7a2ba243d8a548ee99f7b35229679b17803936a1812d31f50efb48b146e0'
    closer=(HERE/'close_unused_viewer.py').read_bytes()
    assert hashlib.sha256(closer).hexdigest()=='be8a00cc5c82447727d3b6503468d8b70d2744b58e8fe5938cbe5d5c7d0523e6'
    destination=BASE+'/realtime-deploy-20260928-r1'
    with remote.connect() as client:
        with client.open_sftp() as sftp:
            sftp.mkdir(destination,0o700)
            for name,data in [('launch_viewer.py',launcher),('close_unused_viewer.py',closer)]:
                with sftp.open(destination+'/'+name,'wx') as f:f.write(data)
                with sftp.open(destination+'/'+name,'rb') as f:assert f.read()==data
        script='''from pathlib import Path
import hashlib,json,subprocess
base=Path(%r);app=Path(%r);target=Path(%r)
assert hashlib.sha256((app/'SOURCE-MANIFEST.json').read_bytes()).hexdigest()==%r
assert hashlib.sha256(Path(%r).read_bytes()).hexdigest()==%r
worker=json.loads((base/'run-r5/worker.json').read_text())
worker['arenas']=1;worker['controller_encoder_path']=%r;worker['controller_decoder_path']=%r
env=json.loads((base/'run-r5/server.json').read_text())['backends'][0]['env']
for name,obj in [('worker-template.json',worker),('env.json',env)]:
    with (target/name).open('x') as f:f.write(json.dumps(obj,indent=2)+'\\n')
subprocess.run(['node',str(app/'prepare.cjs'),str(target/'worker-template.json'),%r,str(base/'run-r6'),str(target/'env.json'),'18772'],check=True)
'''%(BASE,args.app,destination,args.manifest_sha256,BINARY,BIN_SHA,ENC,DEC,BINARY)
        assert remote.run(client,['python3','-c',script])==0
    receipt={'app':args.app,'app_manifest_sha256':args.manifest_sha256,'binary':BINARY,'binary_sha256':BIN_SHA,
             'run':BASE+'/run-r6','port':18772,'launcher':destination+'/launch_viewer.py','closer':destination+'/close_unused_viewer.py'}
    with (HERE/'PREPARED.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
