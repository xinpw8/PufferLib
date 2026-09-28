"""One bounded exact-input query expansion; never estimates an oracle output."""
import argparse,hashlib,itertools,json
from pathlib import Path
from slerp_protocol import words,oracle_table,read_rows

def expand(fixture_path,oracle_path,replay_path,plugin_sha):
    # Revalidate original binary identity, complete repeats and wrapper equality.
    oracle_table(fixture_path,oracle_path,plugin_sha)
    fixture=json.loads(Path(fixture_path).read_text());original=read_rows(oracle_path)
    outputs={r['case_id']:r['rek_wxyz_bits'] for r in original if r.get('event')=='slerp' and r['repeat']==0}
    replacement={};provenance={}
    for case in fixture['cases']:
        if case['kind']!='recorded_native_callback':continue
        replacement.setdefault(tuple(case['native_bits']),set()).add(tuple(outputs[case['id']]))
        provenance.setdefault(tuple(case['native_bits']),[]).append(case['id'])
    ambiguous=sum(len(v)>1 for v in replacement.values())
    cases={tuple(c['input_bits']):dict(c) for c in fixture['cases']}
    for c in fixture['cases']:
        if c['kind']!='recorded_native_callback':continue
        q=c['input_bits'];a=tuple(q[:4]);b=tuple(q[4:8])
        for aa,bb in itertools.product(sorted({a}|replacement.get(a,set())),sorted({b}|replacement.get(b,set()))):
            key=aa+bb+(q[8],);words(key,9)
            cases.setdefault(key,{'kind':'derived_exact_intermediate_substitution','input_bits':list(key),
                                  'source_recorded_case_id':c['id'],'outputs_estimated':False,
                                  'a_origin':{'mode':'unchanged_literal' if aa==a else 'exploratory_measured_alternative','source_case_ids':[] if aa==a else [i for i in provenance[a] if tuple(outputs[i])==aa]},
                                  'b_origin':{'mode':'unchanged_literal' if bb==b else 'exploratory_measured_alternative','source_case_ids':[] if bb==b else [i for i in provenance[b] if tuple(outputs[i])==bb]}})
    replay=read_rows(replay_path)
    if replay[0].get('mode')!='oracle_substitution' or replay[-1].get('event')!='missing_tuple':raise ValueError('expected closed failed lookup trace')
    missing=tuple(replay[-1]['input_bits'])
    if missing not in cases:raise ValueError('observed missing tuple is not explained by exact measured substitutions')
    result=dict(fixture)
    result['cases']=[dict(c,id=i) for i,c in enumerate(cases.values())]
    if not 0<len(result['cases'])<=100000:raise ValueError('query bound')
    result['followup']={'schema':'rek.slerp_boundary.single_followup.v1','previous_fixture_sha256':hashlib.sha256(Path(fixture_path).read_bytes()).hexdigest(),
      'previous_oracle_sha256':hashlib.sha256(Path(oracle_path).read_bytes()).hexdigest(),'failed_replay_sha256':hashlib.sha256(Path(replay_path).read_bytes()).hexdigest(),
      'observed_first_missing_tuple':list(missing),'previous_cases':len(fixture['cases']),'additional_cases':len(cases)-len(fixture['cases']),
      'ambiguous_output_keys':ambiguous,'unique_mapping_claimed':False,
      'derivation':'Unchanged captured inputs plus every measured alternative for matching recorded quaternion outputs, including cartesian combinations; t unchanged. No alternative is selected as causally correct.',
      'source_scope':'sample_root_wxyz callback result can feed final blend; fixture mirror=false. Literal transforms are untouched. Exploratory tuples are not assumed to occur until actual replay.',
      'estimated_outputs':False,'fallback_permitted':False}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fixture');p.add_argument('oracle');p.add_argument('failed_replay');p.add_argument('output',type=Path);p.add_argument('--plugin-sha256',required=True);a=p.parse_args()
    result=expand(a.fixture,a.oracle,a.failed_replay,a.plugin_sha256)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['followup']))
