"""Verify and optionally relink a fixed pair. Never invokes either executable."""
from pathlib import Path
import argparse,hashlib,json,subprocess
PLAN_SHA='e98d56d0b3439c5b4acc7d1ed64853a3a574a9b90bda03c6379fa4ce6c28963c'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--link',action='store_true');args=parser.parse_args()
    path=Path(__file__).resolve().parent/'PLAN.json';assert sha(path)==PLAN_SHA,'Plan changed'
    p=json.loads(path.read_text());dst=Path(p['output_root']);assert not dst.exists(),'Fresh output required'
    pins=p['common_objects']+list(p['workers'].values())+[p['catalog']]+p['runtime_catalog_dependencies']
    for item in pins:assert sha(item['path'])==item['sha256'],'Input changed: '+item['path']
    commands={name:[p['linker'],*p['flags'],*[item['path'] for item in p['common_objects']],worker['path'],*p['link_tail'],'-o',str(dst/(name+'-rek-native-clone'))] for name,worker in p['workers'].items()}
    if not args.link:print(json.dumps({'verified_plan_sha256':PLAN_SHA,'commands':commands,'environment':p['environment'],'executes_gpu':False},indent=2));return
    dst.mkdir()
    (dst/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    results={}
    for name,command in commands.items():
        with (dst/(name+'.stdout.txt')).open('wb') as out,(dst/(name+'.stderr.txt')).open('wb') as err:
            process=subprocess.run(command,stdout=out,stderr=err)
        (dst/(name+'.exit-code.txt')).write_text(str(process.returncode)+'\n')
        assert process.returncode==0,'Link failed: '+name
        results[name]={'path':command[-1],'sha256':sha(command[-1])}
    for item in pins:assert sha(item['path'])==item['sha256'],'Input changed during link: '+item['path']
    receipt={'plan_sha256':PLAN_SHA,'outputs':results,'environment':p['environment'],'executes_gpu':False}
    (dst/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
