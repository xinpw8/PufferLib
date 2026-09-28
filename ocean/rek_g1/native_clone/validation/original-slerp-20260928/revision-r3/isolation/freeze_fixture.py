"""Freeze one already captured request fixture into the new private session."""
from pathlib import Path
import argparse,hashlib,json,shutil
from prepare_runtime import SESSION

def freeze(path):
    if (SESSION/'run.started').exists() or (SESSION/'INPUT.json').exists():raise ValueError('session already frozen or started')
    if not (SESSION/'STAGED.json').is_file() or not (SESSION/'PLUGIN.json').is_file():raise ValueError('private game/plugin not staged')
    data=Path(path).read_bytes();fixture=json.loads(data)
    if fixture['schema']!='rek.original_slerp.fixture.v1' or fixture['repeat_count']!=2 or not 0<len(fixture['cases'])<=100000:raise ValueError('fixture contract')
    target=SESSION/'input/fixtures.json'
    with target.open('xb') as f:f.write(data)
    fixture_sha=hashlib.sha256(data).hexdigest()
    config={'schema':'rek.original_slerp.run.v1','run_id':'slerp-boundary-20260928-r3','expected_game_root':'Z:/session/game',
            'output_directory':'Z:/session/out/oracle','fixture_manifest':'Z:/session/input/fixtures.json','fixture_sha256':fixture_sha}
    raw=(json.dumps(config,indent=2)+'\n').encode()
    with (SESSION/'input/run-config.json').open('xb') as f:f.write(raw)
    marker={'run_id':config['run_id'],'config_sha256':hashlib.sha256(raw).hexdigest()}
    with (SESSION/'game/REK_SLERP_ORACLE_ISOLATED.json').open('x') as f:json.dump(marker,f,indent=2);f.write('\n')
    return {'fixture_sha256':fixture_sha,'config_sha256':marker['config_sha256'],'started':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);a=p.parse_args();print(json.dumps(freeze(a.fixture)))
