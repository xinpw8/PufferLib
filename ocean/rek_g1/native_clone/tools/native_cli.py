"""Run the pinned native worker directly, forwarding JSON stdin/stdout."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_directory',type=Path)
    args=parser.parse_args()
    identity=json.loads((args.run_directory/'identity.json').read_text())
    for name,digest in identity['filePins'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Pinned dependency changed: '+name)
    backend=json.loads((args.run_directory/'server.json').read_text())['backends'][0]
    if backend['env']!=identity['env'] or json.loads(Path(backend['workerConfig']).read_text())!=identity['worker']:
        raise RuntimeError('Prepared environment or worker configuration changed')
    if backend['executable'] not in identity['filePins']:
        raise RuntimeError('Executable is not pinned by run identity')
    env={k:v for k,v in os.environ.items() if not k.startswith('REK_')}
    env.update(backend['env'])
    raise SystemExit(subprocess.call([backend['executable'],'--config',backend['workerConfig']],env=env))

if __name__=='__main__':main()
