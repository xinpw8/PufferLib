"""Compare measured-only callback replay with the preserved original composer trace."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'comparison'))
from compare_traces import load_canonical,compare_rows,repeatability
from original_trace import load_original,normalized_rows

def main():
    p=argparse.ArgumentParser();p.add_argument('native',type=Path);p.add_argument('calls',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    calls=[json.loads(x) for x in a.calls.read_text().splitlines()]
    assert calls[0]['mode']=='oracle_substitution'
    assert calls[-1]=={'event':'end','complete':True,'calls':len(calls)-2}
    assert all(r['event']=='call' and r['call']==i for i,r in enumerate(calls[1:-1]))
    original_path=ROOT/'comparison/original-composer.jsonl'
    assert hashlib.sha256(original_path.read_bytes()).hexdigest()=='e796304f4606002d3edaf755277831b4e84265a48c97a89639326d718fdb18df'
    native=load_canonical(a.native,'native_cpu_port')
    original=load_original(original_path,ROOT/'comparison/oracle-fixture.json',ROOT/'native')
    assert original['all_loaded_dof_arrays_match_declared_map']
    comparison=compare_rows(native['rows'],normalized_rows(original['rows'],original['inverse_map']),('has_clip',))
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result={'schema':'rek.slerp_boundary.composer_replay_comparison.v1','native_trace_sha256':sha(a.native),'calls_sha256':sha(a.calls),
      'original_trace_sha256':sha(original_path),'actual_callback_calls':len(calls)-2,'complete_exact_key_coverage':True,
      'source_justified_joint_order_comparison':comparison,'native_repeatability':repeatability(native['rows']),
      'general_slerp_implementation':False,'production_code_changed':False,'physics_or_server_parity_claim':False,
      'scope':'Exact measured lookup outputs for this explicit fixture only. Native reset is the staged original-semantics extension. Original reference velocity and root-position APIs remain outside native implementation coverage.'}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'report':str(a.output),'sha256':sha(a.output),'bit_exact':comparison['all_supported_fields_bit_exact'],'calls':len(calls)-2,'rows':comparison['rows_compared']}))
if __name__=='__main__':main()
