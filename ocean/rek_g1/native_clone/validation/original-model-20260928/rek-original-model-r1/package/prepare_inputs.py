"""Stage verified inventory inputs in a separately prepared private session. Never launches."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,re
def digest(b):return hashlib.sha256(b).hexdigest()
def encoded(v):return (json.dumps(v,indent=2)+'\n').encode()
def exclusive(p,b):
    with p.open('xb') as f:f.write(b)
    assert p.read_bytes()==b
def prepare(fixture,target,game,run_id):
    fixture,target,game=map(Path,(fixture,target,game));game=game.resolve();target=target.resolve()
    assert re.fullmatch(r'[A-Za-z0-9_-]{8,100}',run_id),'run id'
    assert game.is_dir() and (game/'REK.exe').is_file() and (game.parent/'STAGED.json').is_file(),'prepared private session required'
    assert target==game.parent/'input','input must be private session input'
    assert not target.exists() or (target.is_dir() and not any(target.iterdir())),'input must be absent or empty'
    assert not (game/'REK_COMPOSER_ORACLE_ISOLATED.json').exists(),'marker exists'
    assert not (game.parent/'out/oracle').exists(),'output already exists'
    raw=fixture.read_bytes();d=json.loads(raw)
    assert d['schema']=='rek.original_model_inventory.fixture.v1' and d['robot_id']=='g1' and d['catalog_resource']=='Workshop/RobotCatalog'
    seen=set()
    for f in d['game_files']:
        rel=PurePosixPath(f['file']);assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in f['file'] and ':' not in f['file'],'path escape'
        assert f['file'] not in seen,'duplicate asset';seen.add(f['file'])
        p=(game/Path(*rel.parts)).resolve();assert p.is_relative_to(game),'path escape'
        b=p.read_bytes();assert len(b)==f['bytes'] and digest(b)==f['sha256'],'private asset pin mismatch'
    for name,key in [('Mujoco.Runtime.dll','mujoco_interop_sha256'),('UnityEngine.CoreModule.dll','unity_core_interop_sha256'),('UnityEngine.JSONSerializeModule.dll','unity_json_interop_sha256')]:
        assert digest((game/'BepInEx/interop'/name).read_bytes())==d[key],'private interop pin mismatch'
    target.mkdir(exist_ok=True)
    config={'schema':'rek.original_model_inventory.run.v1','run_id':run_id,'expected_game_root':'Z:/session/game','output_directory':'Z:/session/out/oracle','fixture_manifest':'Z:/session/input/fixtures.json','fixture_sha256':digest(raw)}
    cfg=encoded(config);marker={'run_id':run_id,'config_sha256':digest(cfg)}
    exclusive(target/'fixtures.json',raw);exclusive(target/'run-config.json',cfg);exclusive(game/'REK_COMPOSER_ORACLE_ISOLATED.json',encoded(marker))
    receipt={'schema':'rek.original_model_inventory.input_staging.v1','run_id':run_id,'fixture_sha256':digest(raw),'config_sha256':digest(cfg),'marker':marker,'required_environment':{'REK_COMPOSER_ORACLE_ENABLE':'detached-v1','REK_COMPOSER_ORACLE_CONFIG':'Z:/session/input/run-config.json'},'required_cli':['--rek-composer-oracle'],'output_must_be_absent':'Z:/session/out/oracle','asset_pins_checked':len(seen),'launch_performed':False}
    exclusive(target/'INPUT-STAGING.json',encoded(receipt));return receipt
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--fixture',required=True);a.add_argument('--input',required=True);a.add_argument('--private-game',required=True);a.add_argument('--run-id',required=True);v=a.parse_args();print(json.dumps(prepare(v.fixture,v.input,v.private_game,v.run_id),indent=2))
