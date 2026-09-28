"""Read-only check of original trace attestation, fixtures and decoder packing.
This CPU contract check is separate from the actual native/CUDA implementation.
"""
import argparse, hashlib, json, math, struct
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def bits(x): return '0x'+struct.pack('>f',x).hex()
def checkpack(p,expected):
    assert p['bits']==[bits(v) for v in expected]
    assert len(p['values'])==len(expected)
    assert all(math.isfinite(v) and bits(v)==b for v,b in zip(p['values'],p['bits']))

def verify(trace, fixture):
    raw=trace.read_bytes();lines=[json.loads(x) for x in raw.splitlines()]
    f=json.loads(fixture.read_bytes());header=lines[0];footer=lines[-1]
    assert header['schema']=='rek.original_history.trace.v1' and header['fixture_sha256']==sha(fixture)
    assert header['game_sha256']=='6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412'
    assert header['metadata_sha256']=='e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd'
    assert header['interop_sha256']=='faa94fb58e24fda95e2c06810e28b9eb2d6d9f9f8327541976a0dc1011f646d2'
    assert footer['event']=='oracle_end' and footer['success'] and footer['rows']==28
    att=next(x for x in lines if x['event']=='entrypoints')['methods']
    assert len(att)==9 and all(x['module'].replace('\\','/').endswith('/GameAssembly.dll') for x in att)
    assert {x['method'] for x in att}=={'Push','GetLatest','Clear','BuildObsPlans','FillHistoryChannel','BuildDecoderObs','CalcHeadingMj','YawQuatMj','QuatMulMj'}
    for x in att:assert int(x['entrypoint'],16)>0 and int(x['rva'],16)>0 and len(x['first32_sha256'])==64
    lifecycles=[x for x in lines if x['event']=='lifecycle']
    assert len(lifecycles)==2
    for x in lifecycles:
        assert not x['active'] and not x['enabled'] and not x['is_ready']
        assert x['capacity']==21 and x['decoder_size']==994 and x['encoder_size']==1762
        assert [a['offset'] for a in x['decoder_plan']]==[0,64,94,384,674,964]
        assert [a['dim'] for a in x['decoder_plan']]==[64,30,290,290,290,30]
        checkpack(x['tokens'],f['tokens'])
    queries=[x for x in lines if x['event']=='history_query']
    totals={'decoder_float_bits':0,'snapshot_channel_float_bits':0,'direct_fill_float_bits':0}
    for repeat in range(2):
        history=[];write=0
        actual={x['operation']:x for x in queries if x['repeat']==repeat}
        for operation,op in enumerate(f['operations']):
            if op['kind']=='push':
                history.append(f['snapshots'][op['snapshot_id']]);history=history[-21:];write=(write+1)%21
            elif op['kind']=='clear':history=[];write=0
            else:
                a=actual[operation];assert a['count']==len(history) and a['write_index']==write and a['label']==op['label']
                sampled={}
                for n,step,key in [(10,1,'latest_step1'),(5,2,'latest_step2')]:
                    take=min(n,len(history)//step)
                    expected=[None]*(n-take)+[history[-1-(take-i-1)*step] for i in range(take)]
                    sampled[step]=expected
                    assert len(a[key])==n
                    for got,snapshot in zip(a[key],expected):
                        for channel,width in f['history']['channels'].items():
                            if snapshot is None:assert got[channel] is None
                            else:checkpack(got[channel],snapshot[channel]);totals['snapshot_channel_float_bits']+=width
                decoder=list(f['tokens'])
                for index,(channel,width) in enumerate(f['history']['channels'].items()):
                    values=[];direct=[-333.5,-333.5]
                    for snap in sampled[1]:
                        values+=([0]*width if snap is None else snap[channel])
                        direct+=([-333.5]*width if snap is None else snap[channel])
                    decoder+=values;direct += [-333.5,-333.5]
                    checkpack(a['direct_channel_fills'][index]['values'],direct)
                    totals['direct_fill_float_bits']+=len(direct)
                checkpack(a['decoder'],decoder);totals['decoder_float_bits']+=len(decoder)
    one=[{k:v for k,v in x.items() if k!='repeat'} for x in queries if x['repeat']==0]
    two=[{k:v for k,v in x.items() if k!='repeat'} for x in queries if x['repeat']==1]
    assert one==two,'fresh-repeat mismatch'
    functions=[x for x in lines if x['event']=='quaternion_function'];assert len(functions)==4
    calls=footer['completed_wrapper_calls']
    for name,n in {'BuildObsPlans':2,'Push':62,'Clear':2,'GetLatest':56,'BuildDecoderObs':28,'FillHistoryChannel':140,'CalcHeadingMj':4,'YawQuatMj':4,'QuatMulMj':4}.items():assert calls[name]==n
    return {'schema':'rek.original_history.verified.v1','trace':str(trace),'trace_sha256':sha(trace),'trace_bytes':len(raw),'fixture_sha256':sha(fixture),'query_rows':len(queries),'fresh_repeats_bit_identical':True,'attested_original_methods':att,'exact_contract_checks':totals,'completed_wrapper_calls':calls,'scope':'Original supplied-state history/decoder packing only. Independent native implementation comparison is a separate receipt. No full runner lifecycle, motor inference, physics or server authority claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=verify(a.trace,a.fixture)
    with a.output.open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result,indent=2))
