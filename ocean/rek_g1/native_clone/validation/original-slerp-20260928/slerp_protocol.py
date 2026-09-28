"""Exact-bit request reduction and closed-oracle table conversion, no Unity execution."""
from pathlib import Path
import argparse, hashlib, json, math, re, struct

SCHEMA='rek.original_slerp.fixture.v1'
GAME='6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412'
META='e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd'

def words(items,n):
    if len(items)!=n:raise ValueError('wrong word count')
    result=[]
    for item in items:
        if not isinstance(item,str) or not re.fullmatch(r'0x[0-9a-f]{8}',item):raise ValueError('noncanonical FP32 word')
        value=int(item,16)
        if not math.isfinite(struct.unpack('<f',struct.pack('<I',value))[0]):raise ValueError('nonfinite FP32 word')
        result.append(value)
    return tuple(result)

def bits(values):return [f'0x{struct.unpack("<I",struct.pack("<f",v))[0]:08x}' for v in values]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def make_fixture(path):
    rows=read_rows(path)
    if not rows or rows[0].get('schema')!='rek.slerp_calls.v1' or rows[0].get('mode')!='libm_capture':raise ValueError('capture header')
    calls=rows[1:-1]
    if not calls or rows[-1]!={'event':'end','complete':True,'calls':len(calls)}:raise ValueError('missing or mismatched closed footer')
    cases={}
    for index,row in enumerate(calls):
        if row.get('event')!='call' or row.get('call')!=index:raise ValueError('call sequence')
        key=words(row['input_bits'],9);output=words(row['output_bits'],4)
        if key in cases:
            if cases[key]['native_bits']!=row['output_bits']:raise ValueError('nondeterministic native callback')
            cases[key]['occurrences']+=1
        else:cases[key]={'id':len(cases),'kind':'recorded_native_callback','input_bits':row['input_bits'],'native_bits':row['output_bits'],
            'first_call':index,'first_context':{k:row[k] for k in ['trial','tick','phase']},'occurrences':1}
    # Explicit controls are additional synthetic inputs, never recorded evidence.
    for label,a,b,t in [
        ('identical',[1,0,0,0],[1,0,0,0],.5),('antipodal',[1,0,0,0],[-1,0,0,0],.5),
        ('quarter_turn',[1,0,0,0],[0,0,0,1],.25),('nonunit',[2,0,0,0],[0,0,0,3],.25),
        ('clamp_low',[1,0,0,0],[0,0,0,1],-.25),('clamp_high',[1,0,0,0],[0,0,0,1],1.25)]:
        values=bits(a+b+[t]);key=words(values,9)
        if key not in cases:cases[key]={'id':len(cases),'kind':'synthetic_control','label':label,'input_bits':values}
    return {'schema':SCHEMA,'source_capture':{'path':str(Path(path).resolve()),'sha256':digest(path),'calls':len(calls)},
            'composer_fixture_sha256':rows[0]['fixture_sha256'],'repeat_count':2,'input_order':'a_wxyz,b_wxyz,t',
            'normalization_by_harness':False,'cases':list(cases.values())}

def oracle_table(fixture_path,trace_path,plugin_sha256):
    fixture=json.loads(Path(fixture_path).read_text());rows=read_rows(trace_path)
    if fixture['schema']!=SCHEMA or fixture['repeat_count']!=2:raise ValueError('fixture contract')
    header=rows[0]
    if header.get('schema')!='rek.original_slerp.trace.v1' or header.get('fixture_sha256')!=digest(fixture_path):raise ValueError('trace fixture identity')
    if header.get('game_sha256')!=GAME or header.get('metadata_sha256')!=META:raise ValueError('original binary identity')
    if not re.fullmatch('[0-9a-f]{64}',plugin_sha256) or header.get('plugin_sha256')!=plugin_sha256:raise ValueError('plugin identity')
    if header.get('unity_player_sha256')!='277953a7035b1633c239904853bfbea7b2948937ef5567e70c1911c260dd1414':raise ValueError('Unity player identity')
    if header.get('rek_interop_sha256')!='faa94fb58e24fda95e2c06810e28b9eb2d6d9f9f8327541976a0dc1011f646d2' or header.get('unity_interop_sha256')!='3ac45305a21f5107c0a9e813c48503cf27944c15ab9f6495396f20f61840104a':raise ValueError('interop identity')
    methods=[r for r in rows if r.get('event')=='entrypoints']
    if len(methods)!=1 or {m['method'] for m in methods[0]['methods']}!={'SlerpWxyz','Slerp'}:raise ValueError('method attestation')
    root=header['game_root'].replace('\\','/').rstrip('/').lower()
    for m in methods[0]['methods']:
        if m['module'].replace('\\','/').lower()!=root+'/gameassembly.dll' or not re.fullmatch('[0-9a-f]{64}',m['first32_sha256']):raise ValueError('method module binding')
    cases=fixture['cases'];measured=[r for r in rows if r.get('event')=='slerp']
    footer=rows[-1]
    if footer.get('event')!='oracle_end' or footer.get('success') is not True or footer.get('rows')!=len(cases)*2 or len(measured)!=len(cases)*2:raise ValueError('incomplete original trace')
    seen={};table={};different=0;maxabs=0.0
    for row in measured:
        cid=row['case_id'];rep=row['repeat'];key=(rep,cid)
        if rep not in [0,1] or cid<0 or cid>=len(cases) or key in seen:raise ValueError('duplicate/missing query')
        c=cases[cid];inputs=words(row['input_bits'],9)
        if inputs!=words(c['input_bits'],9):raise ValueError('query input changed')
        a=words(row['rek_wxyz_bits'],4);b=words(row['unity_wxyz_bits'],4)
        if a!=b:raise ValueError('REK wrapper differs from Unity public Slerp; no substitution allowed')
        if inputs in table and table[inputs]!=a:raise ValueError('original repeat nondeterminism')
        seen[key]=True;table[inputs]=a
        if rep==0 and 'native_bits' in c:
            native=words(c['native_bits'],4)
            different+=sum(x!=y for x,y in zip(a,native))
            for x,y in zip(a,native):maxabs=max(maxabs,abs(struct.unpack('<f',struct.pack('<I',x))[0]-struct.unpack('<f',struct.pack('<I',y))[0]))
    if len(seen)!=len(cases)*2:raise ValueError('coverage')
    blob=b'RSLPTB1\0'+struct.pack('<I',len(table))+b''.join(struct.pack('<13I',*key,*table[key]) for key in sorted(table))
    return blob,{'schema':'rek.slerp_boundary.direct_comparison.v1','fixture_sha256':digest(fixture_path),'oracle_sha256':digest(trace_path),
                 'unique_tuples':len(table),'repeat_bit_equality':True,'rek_unity_wrapper_bit_equality':True,
                 'differing_recorded_components':different,'max_abs_recorded_component_error':maxabs,
                 'component_replay_performed':False,'physics_or_transfer_claim':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('requests');p.add_argument('calls');p.add_argument('output',type=Path)
    p=sub.add_parser('table');p.add_argument('fixture');p.add_argument('oracle');p.add_argument('output',type=Path);p.add_argument('--plugin-sha256',required=True)
    args=parser.parse_args()
    if args.op=='requests':
        with args.output.open('x') as f:json.dump(make_fixture(args.calls),f,indent=2);f.write('\n')
    else:
        blob,report=oracle_table(args.fixture,args.oracle,args.plugin_sha256)
        with args.output.open('xb') as f:f.write(blob)
        with args.output.with_suffix('.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
