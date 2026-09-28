"""Stage immutable explicit inputs for a separately owned isolated Unity process. No launch."""
import argparse, hashlib, json, pathlib

def sha(b): return hashlib.sha256(b).hexdigest()
def exclusive(path, data):
    with path.open('xb') as f: f.write(data)
    assert path.read_bytes() == data
def encoded(value): return (json.dumps(value, indent=2) + '\n').encode()

def prepare(fixture, target, game, mode, run_id, windows_game_root):
    fixture, target, game = map(pathlib.Path, (fixture, target, game))
    assert mode in ('smoke', 'sequence')
    assert 8 <= len(run_id) <= 100
    assert game.is_dir() and (game / 'REK.exe').is_file(), 'expected pre-created PRIVATE game copy'
    assert not target.exists() or (target.is_dir() and not any(target.iterdir())), 'input directory must be absent or verified empty'
    assert not (game / 'REK_COMPOSER_ORACLE_ISOLATED.json').exists(), 'marker already exists'
    raw = fixture.read_bytes(); data = json.loads(raw)
    assert data['schema'] == 'rek.original_history.fixture.v1'
    bindings = [data['joint_config']]
    verified = {}
    for bind in bindings:
        source = (fixture.parent / bind['file']).resolve()
        assert source.parent == fixture.parent.resolve(), 'fixture path escape'
        blob = source.read_bytes()
        assert len(blob) == bind['bytes'] and sha(blob) == bind['sha256'], 'bound fixture mismatch'
        assert bind['file'] not in verified
        verified[bind['file']] = blob
    target.mkdir(exist_ok=True)
    for name, blob in verified.items(): exclusive(target / name, blob)
    exclusive(target / 'fixtures.json', raw)
    config = {'schema':'rek.original_composer.run.v1', 'run_id':run_id,
        'expected_game_root':windows_game_root, 'output_directory':'Z:/session/out/oracle',
        'fixture_manifest':'Z:/session/input/fixtures.json','fixture_sha256':sha(raw),'mode':mode}
    config_bytes = encoded(config)
    exclusive(target / 'run-config.json', config_bytes)
    marker = {'run_id':run_id, 'config_sha256':sha(config_bytes)}
    exclusive(game / 'REK_COMPOSER_ORACLE_ISOLATED.json', encoded(marker))
    receipt = {'schema':'rek.original_composer.input_staging.v1','fixture_sha256':sha(raw),
        'config_sha256':sha(config_bytes),'marker':marker,'run_id':run_id,'mode':mode,
        'required_environment':{'REK_COMPOSER_ORACLE_ENABLE':'detached-v1',
        'REK_COMPOSER_ORACLE_CONFIG':'Z:/session/input/run-config.json'},
        'required_cli':['--rek-composer-oracle'], 'output_must_be_absent':'Z:/session/out/oracle',
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in target.iterdir()}}
    exclusive(target / 'INPUT-STAGING.json', encoded(receipt))
    return receipt

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--input',required=True)
    p.add_argument('--private-game',required=True);p.add_argument('--mode',choices=['smoke','sequence'],required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--windows-game-root',required=True)
    a=p.parse_args(); print(json.dumps(prepare(a.fixture,a.input,a.private_game,a.mode,a.run_id,a.windows_game_root),indent=2))
