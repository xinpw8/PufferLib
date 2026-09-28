"""Root-authorized GPU entry point: six fresh processes, one control tick each."""
import argparse,datetime,hashlib,json,os,subprocess
from pathlib import Path
SCENARIOS=['ignore_side0_nan','ignore_side1_nan','ignore_side0_oob','ignore_side1_oob','reject_side0_nan_with_side1_direct','reject_side1_oob_with_side0_direct']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--environment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;binary=root/'build-r1/direct-api-test'
    env={k:v for k,v in os.environ.items() if not k.startswith('REK_')};configured=json.loads(a.environment.read_text());env.update(configured)
    assert configured['REK_PHYSICS_BACKEND']=='mujoco_cuda' and configured['REK_ALLOW_CPU_EVALUATION']=='0' and configured['REK_PHYSICAL_OPPONENT']=='recovered_bot1_g1_v1'
    config=json.loads(a.config.read_text());assert config['backend']=='mujoco' and config['arenas']==4
    a.output.mkdir();records=[]
    for scenario in SCENARIOS:
        cmd=[str(binary),'--config',str(a.config.resolve()),'--scenario',scenario]
        start=datetime.datetime.now(datetime.timezone.utc).isoformat()
        r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120)
        (a.output/(scenario+'.stdout.txt')).write_text(r.stdout);(a.output/(scenario+'.stderr.txt')).write_text(r.stderr)
        receipt={'scenario':scenario,'argv':cmd,'started_utc':start,'exit_code':r.returncode}
        (a.output/(scenario+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        assert r.returncode==0,(scenario,r.stdout,r.stderr[-2000:])
        result=json.loads(r.stdout);assert result['pass'] and result['scenario']==scenario
        records.append(result)
    receipt={'schema':'rek.native_clone.direct_api_regression.v1','pass':True,'binary_sha256':sha(binary),'config_sha256':sha(a.config),'environment_sha256':sha(a.environment),'scenarios':records,'processes':6,'control_ticks_per_process':1}
    (a.output/'RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
