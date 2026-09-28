import argparse,hashlib,json,struct
from pathlib import Path
root=Path(__file__).parent
oracle=root.parent/'original-history/runtime-r1/oracle.jsonl'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[json.loads(line) for line in oracle.read_text().splitlines()]
assert sha(oracle)=='8495728eb588d9ff51da93d1f6783fbb3c8757cc80f56a16f6892a85bd9f308c'
assert records[-1]['event']=='oracle_end' and records[-1]['success'] and records[-1]['rows']==28
fixture=json.loads((root/'tests/fixtures.json').read_text())
assert records[0]['fixture_sha256']==sha(root/'tests/fixtures.json')
rows=[r for r in records if r['event']=='history_query']
binary=(root/'build-cpu-r1/history-native.f32').read_bytes()
labels=[o['label'] for o in fixture['operations'] if o['kind']=='query']
assert len(binary)==len(labels)*994*4 and len(rows)==len(labels)*2
checks=[]
for row in rows:
    index=labels.index(row['label']);candidate=binary[index*3976:(index+1)*3976]
    expected=b''.join(struct.pack('<I',int(b,16)) for b in row['decoder']['bits'])
    assert len(expected)==3976
    errors=[i for i in range(994) if expected[i*4:(i+1)*4]!=candidate[i*4:(i+1)*4]]
    checks.append({'repeat':row['repeat'],'operation':row['operation'],'label':row['label'],'compared':994,'bit_mismatches':errors})
heading=(root/'build-cpu-r1/heading-native.f32').read_bytes()
quats=[r for r in records if r['event']=='quaternion_function']
assert len(heading)==len(quats)*20
headings=[]
for i,row in enumerate(quats):
    expected=b''.join(struct.pack('<I',int(b,16)) for b in [row['heading']['bits']]+row['yaw_wxyz']['bits'])
    actual=heading[i*20:(i+1)*20]
    headings.append({'label':row['label'],'compared':5,'bit_mismatches':[j for j in range(5) if expected[j*4:(j+1)*4]!=actual[j*4:(j+1)*4]],'native_values':list(struct.unpack('<5f',actual))})
report={'schema':'rek.native_clone.original_history_comparison.v1','oracle':{'path':str(oracle),'sha256':sha(oracle)},'fixture_sha256':sha(root/'tests/fixtures.json'),
    'native_history_sha256':sha(root/'build-cpu-r1/history-native.f32'),'native_heading_sha256':sha(root/'build-cpu-r1/heading-native.f32'),
    'history_values_compared':27832,'history_bit_mismatch_count':sum(len(x['bit_mismatches']) for x in checks),'queries':checks,
    'heading_values_compared':20,'heading_bit_mismatch_count':sum(len(x['bit_mismatches']) for x in headings),'heading_cases':headings,
    'scope':'Shared production step-1 history append and decoder packing with explicit pretransformed snapshots; original ring step2 separately verified by oracle, not implemented in this fixed step1 native path. Heading compares original CalcHeadingMj and YawQuatMj on four explicit inputs using CPU libm callbacks. No physics, source snapshot transforms, model inference, GPU, server, or all-input transcendental parity claim.'}
(root/'HISTORY-COMPARISON.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['history_values_compared','history_bit_mismatch_count','heading_values_compared','heading_bit_mismatch_count']}))
assert not report['history_bit_mismatch_count'] and not report['heading_bit_mismatch_count']
