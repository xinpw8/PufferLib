"""Compare independent original-binary and native-port traces without fitting.

Raw float32 bits are authoritative. Numerical error and quaternion orientation
error are reported separately; no tolerance silently turns a mismatch into parity.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import struct

ORDERS=('reset_then_idle','idle_then_reset')
OFFSETS=(0,1,5,10,15)
LAYER_FLOATS=('cursor','speed','per_tick','prev_heading','last_heading_delta')
LAYER_INTS=('clip_id','has_clip','has_config','active','heading_valid','heading_resync','start_frame','end_frame')
STATE_INTS=('xt','w_in','w_out','w_total','action_playing','action_move_id')
STATE_FLOATS=('pending_heading_delta','heading_ownership')

def require(test,message):
    if not test: raise ValueError(message)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def parse_integer(text):
    # JSON allows -0; Python's ordinary int parser erases that floating sign.
    # Keep all nonzero counters as integers and preserve only this literal.
    return -0.0 if text=='-0' else int(text)

def fp(value):
    require(isinstance(value,dict) and set(value)=={'value','bits'},'invalid scalar representation')
    bits=int(value['bits'],16)
    require(0<=bits<=0xffffffff,'float32 bits out of range')
    raw=struct.unpack('<f',struct.pack('<I',bits))[0]
    require(math.isfinite(raw) and math.isfinite(value['value']),'nonfinite scalar')
    require(struct.pack('<f',value['value'])==struct.pack('<I',bits),'decimal/bit disagreement')
    return raw,bits

def vector(value,width):
    require(isinstance(value,dict) and set(value)=={'values','bits'},'invalid vector representation')
    require(len(value['values'])==len(value['bits'])==width,'vector width mismatch')
    return [fp({'value':v,'bits':b}) for v,b in zip(value['values'],value['bits'])]

def fields(row,omit_layer_fields=()):
    result={}
    for name in ('pre','post_advance','post_consume'):
        state=row[name]
        for side in ('current','from'):
            layer=state[side]
            for key in LAYER_INTS:
                if key in omit_layer_fields: continue
                require(type(layer[key]) is int,'noninteger layer field')
                result[f'{name}.{side}.{key}']=('integer',layer[key])
            for key in LAYER_FLOATS:
                result[f'{name}.{side}.{key}']=('float',fp(layer[key]))
        for key in STATE_INTS:
            require(type(state[key]) is int,'noninteger state field')
            result[f'{name}.{key}']=('integer',state[key])
        for key in STATE_FLOATS:
            result[f'{name}.{key}']=('float',fp(state[key]))
    require([r['frames_ahead'] for r in row['references']]==list(OFFSETS),'reference offsets differ')
    for ref in row['references']:
        offset=ref['frames_ahead']
        for name,width in (('dof_position_mujoco',29),('root_quaternion_wxyz',4)):
            for i,value in enumerate(vector(ref[name],width)):
                result[f'references.{offset}.{name}.{i}']=('float',value)
    result['consumed_heading_delta']=('float',fp(row['consumed_heading_delta']))
    return result

def load_canonical(path,producer):
    path=Path(path);before=path.stat();raw=path.read_bytes();after=path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'trace changed during read')
    require(raw.endswith(b'\n'),'unterminated trace')
    records=[json.loads(x,parse_int=parse_integer) for x in raw.splitlines()]
    header=records[0]
    require(header['event']=='header' and header['producer']==producer,'wrong trace producer/header')
    require(header.get('root_quaternion_order')=='wxyz' and header.get('fp32_bits_authoritative') is True,'ambiguous float/quaternion convention')
    require(header.get('gpu') is False and header.get('physics') is False,'wrong trace scope')
    footer=records[-1]
    require(footer['event'] in ('native_end','oracle_end') and footer.get('complete') is True,'trace incomplete')
    rows=[r for r in records if r['event']=='row']
    require(len(rows)==640,'expected four160-row trials')
    for index,row in enumerate(rows):
        order,repeat,tick=ORDERS[index//320],(index//160)%2,index%160
        require((row['order'],row['repeat'],row['tick'],row['trial'])==(order,repeat,tick,index//160),'trace row order/identity mismatch')
        fields(row)
    require(footer.get('rows')==640 and footer.get('trials')==4,'footer count mismatch')
    require([r['trial'] for r in records if r['event']=='trial_end']==[0,1,2,3],'missing trial terminal')
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'header':header,'rows':rows}

def ordered_bits(bits):
    return 0x80000000-(bits&0x7fffffff) if bits&0x80000000 else 0x80000000+bits

def category(path):
    if path.startswith('references.'):
        return path.split('.')[2]
    return path

def compare_rows(native,oracle,omit_layer_fields=()):
    require(len(native)==len(oracle),'different row coverage')
    stats={};integer={};first=None;angular=[]
    for n,o in zip(native,oracle):
        identity={k:n[k] for k in ('order','repeat','tick','trial')}
        require(all(n[k]==o[k] for k in identity),'row alignment differs')
        nf,of=fields(n,omit_layer_fields),fields(o,omit_layer_fields);require(nf.keys()==of.keys(),'field coverage differs')
        for path,(kind,a) in nf.items():
            b=of[path][1];key=category(path)
            if kind=='integer':
                s=integer.setdefault(key,{'values':0,'different':0,'first_difference':None});s['values']+=1
                if a!=b:
                    s['different']+=1
                    if s['first_difference'] is None:s['first_difference']=identity|{'path':path,'native':a,'original':b}
                    if first is None:first=identity|{'path':path,'kind':'integer'}
                continue
            s=stats.setdefault(key,{'values':0,'different_bits':0,'maximum_absolute_error':0.,'sum_squared_error':0.,'maximum_ulp_difference':0,'first_difference':None})
            av,ab=a;bv,bb=b;error=abs(av-bv);s['values']+=1;s['sum_squared_error']+=error*error
            s['maximum_absolute_error']=max(s['maximum_absolute_error'],error)
            s['maximum_ulp_difference']=max(s['maximum_ulp_difference'],abs(ordered_bits(ab)-ordered_bits(bb)))
            if ab!=bb:
                s['different_bits']+=1
                if s['first_difference'] is None:s['first_difference']=identity|{'path':path,'native_bits':f'0x{ab:08x}','original_bits':f'0x{bb:08x}','native':av,'original':bv}
                if first is None:first=identity|{'path':path,'kind':'float32_bits'}
        for nr,orr in zip(n['references'],o['references']):
            a=[x[0] for x in vector(nr['root_quaternion_wxyz'],4)];b=[x[0] for x in vector(orr['root_quaternion_wxyz'],4)]
            an=math.sqrt(sum(x*x for x in a));bn=math.sqrt(sum(x*x for x in b));require(an>0 and bn>0,'zero quaternion')
            dot=abs(sum(x*y for x,y in zip(a,b))/(an*bn));dot=min(1.,dot)
            angular.append(2*math.acos(dot))
    for s in stats.values():s['rms_error']=math.sqrt(s.pop('sum_squared_error')/s['values'])
    return {'rows_compared':len(native),'float_fields':stats,'integer_fields':integer,'first_difference':first,
            'all_supported_fields_bit_exact':first is None,
            'quaternion_orientation_maximum_error_radians':max(angular,default=None),
            'quaternion_orientation_note':'sign-equivalent rotations may have different components; raw bit results remain unchanged'}

def repeatability(rows,omit_layer_fields=()):
    out=[]
    for i,order in enumerate(ORDERS):
        a=rows[i*320:i*320+160];b=rows[i*320+160:(i+1)*320]
        # Only comparison metadata is normalized. State/reference values are untouched.
        b=[r|{k:a[j][k] for k in ('repeat','trial')} for j,r in enumerate(b)]
        out.append({'order':order,'comparison':compare_rows(a,b,omit_layer_fields)})
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--oracle',type=Path);p.add_argument('--oracle-fixture',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'output exists')
    n=load_canonical(a.native,'native_cpu_port')
    result={'schema':'rek.composer_trace_comparison.v1','comparator_sha256':sha(__file__),
            'native':{k:v for k,v in n.items() if k!='rows'},'native_repeatability':repeatability(n['rows']),
            'original_binary_compared':False,'gpu':False,'physics':False,'server_parity_claim':False,
            'whole_composer_parity_claim':False,'overall_parity_verdict':'original_sequence_pending',
            'scope_limits':['Native Reset is the staged original-semantics extension52fc10a2, not the current runtime reconstruct-composer reset.',
              'Native math uses the existing explicitly named libm candidate backend; identical original engine math is unproven.',
              'Native reference velocities are separately labeled finite differences; original TryGetReferenceVelocity implementation is not present in the native API.',
              'Native reference root positions are unsupported; a successful supported-field comparison cannot establish whole-composer or current-runtime parity.']}
    if a.oracle:
        from original_trace import load_original, normalized_rows, auxiliary_comparison
        require(a.oracle_fixture is not None,'--oracle-fixture required with original trace')
        require(n['header']['fixture_sha256']==sha(Path(__file__).resolve().parent/'fixture.json'),'native trace fixture pin differs')
        o=load_original(a.oracle,a.oracle_fixture,Path(__file__).resolve().parent)
        result['oracle']={k:v for k,v in o.items() if k not in ('rows','records')}
        result['raw_index_comparison']=compare_rows(n['rows'],o['rows'],('has_clip',))
        result['raw_index_comparison']['joint_order_note']='Raw index against raw index; original BuildClip applies its supplied map, native loader leaves NPZ columns unchanged.'
        result['oracle_repeatability']=repeatability(o['rows'],('has_clip',))
        if o['all_loaded_dof_arrays_match_declared_map']:
            mapped=normalized_rows(o['rows'],o['inverse_map'])
            result['source_justified_joint_order_comparison']=compare_rows(n['rows'],mapped,('has_clip',))
            result['source_justified_joint_order_comparison']['joint_order_note']='Original output is explicitly inverted through its logged, config-pinned BuildClip map. Raw comparison remains preserved; no output-driven fitting.'
        else:
            result['source_justified_joint_order_comparison']=None
        result['auxiliary_outputs']=auxiliary_comparison(n['rows'],o['rows'],o['inverse_map'],o['all_loaded_dof_arrays_match_declared_map'])
        result['unsupported_state_fields']=['original layer has_clip is not emitted by this frozen oracle; omitted from cross-implementation comparison']
        result['original_binary_compared']=True
        measured=result['source_justified_joint_order_comparison'] or result['raw_index_comparison']
        result['overall_parity_verdict']='measured_fields_differ' if not measured['all_supported_fields_bit_exact'] else 'incomplete_coverage_no_whole_composer_parity_claim'
    with a.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'output':str(a.output.resolve()),'sha256':sha(a.output),'original_binary_compared':result['original_binary_compared']}))

if __name__=='__main__':main()
