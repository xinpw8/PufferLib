"""Read-only derivation from saved preferences, original key getters and native assets."""
from pathlib import Path
import base64, datetime, hashlib, json, re, winreg

OUT = Path(__file__).parent
CLONE = Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile\ocean\rek_g1\native_clone')
ISIL = Path(r'C:\rekagent\work\controller-audit-isil\IsilDump')
PATHS = {
    'windows_export': Path(r'\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-25\human-imitation-session-r1\spark-controls\windows-controls.json'),
    'saved_browser_bindings': Path(r'C:\rekagent\work\rek-playback-app-20260928-r7\app\saved-g1-bindings.json'),
    'original_key_getters': ISIL/'Unity.InputSystem/UnityEngine/InputSystem/Keyboard.txt',
    'original_keyboard_scheme': ISIL/'REKApp/REKApp/KeyboardControlScheme.txt',
    'original_chord_resolver': ISIL/'REKApp/REKApp/ControlScheme.txt',
    'original_move_dispatch': ISIL/'REKApp/REKApp/RobotInputController.txt',
    'native_move_assets': CLONE/'validation/original-slerp-20260928/native/assets/semantic_duel_assets_manifest.json',
    'policy_translation': CLONE/'dependency_source/rek-training/native-mujoco-gpu-20260914/native-source/ocean/rek_g1/g1_semantic_action_table.c',
    'native_direct_dispatch': CLONE/'source/ocean/rek_g1/g1_semantic_scheduler_cuda.cu',
    'native_asset_parser': CLONE/'source/ocean/rek_g1/native5/motion_assets.cu',
    'buggy_browser_to_native': Path(r'C:\rekagent\work\rek-playback-app-20260928-r7\app\league\input.cjs'),
    'old_keyboard_gestures': Path(r'C:\rekagent\work\rek-playback-app-20260928-r7\app\league\public\controls.js'),
}
def sha(data): return hashlib.sha256(data).hexdigest()
def read_json(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def write(name, value):
    path=OUT/name
    with path.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())}

pins={key:{'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())} for key,path in PATHS.items()}
assert pins['windows_export']['sha256']=='118ab0ca87c5a2a37f985aaa03b8892198e92a22051b8e5bc38badce966f7d89'
source=read_json(PATHS['windows_export'])
g1=next(v for v in source['values'] if v['name']=='rek.controls.custom.g1_h4186581798')
decoded=base64.b64decode(g1['encoded']).rstrip(b'\x00').decode('utf-8')
assert decoded==g1['decoded']
keyboard=json.loads(decoded)['keyboard']
saved=read_json(PATHS['saved_browser_bindings'])['bindings']
assert len(keyboard)==len(saved)==17

# Derive browser physical codes from the original native key getter methods.
key_text=PATHS['original_key_getters'].read_text()
numeric_to_code={}
for key in ['space','quote','semicolon','h','i','j','k','l','o','u','y']:
    method=re.search(r'Method: [^\n]* get_'+key+r'Key\(\)(.*?)(?=\nMethod:|\Z)',key_text,re.S).group(1)
    numeric=int(re.search(r'Call Keyboard.get_Item[^\n]*, (\d+), 0',method).group(1))
    numeric_to_code[numeric]=key.capitalize() if len(key)>1 else 'Key'+key.upper()

assets=read_json(PATHS['native_move_assets'])
clips={clip['npz_path_id']:clip for clip in assets['clips']}
routes={}
for route in assets['routes']:
    if route['runtime_move_index'] is None: continue
    clip=clips[route['npz_path_id']]
    command_id='move:'+Path(clip['source_file']).stem
    assert command_id not in routes
    routes[command_id]=(route,clip)
assert len(routes)==17
table_text=PATHS['policy_translation'].read_text()
table=list(map(int,re.findall(r'\d+',re.search(r'MOVE_REGISTRY_ORDER\[.*?\]\s*=\s*\{(.*?)\}',table_text,re.S).group(1))))
assert sorted(table)==list(range(17))
by_index={r['runtime_move_index']:name for name,(r,c) in routes.items()}
rows=[]
for actual,browser in zip(keyboard,saved):
    assert actual['commandId']==browser['commandId']
    numeric=[actual[f'slot{i}'] for i in [1,2,3] if actual[f'slot{i}']!=0]
    codes=[numeric_to_code[n] for n in numeric]
    assert codes==browser['codes'] and actual['doubleTap']==browser['doubleTap']
    route,clip=routes[actual['commandId']]
    index=route['runtime_move_index']
    category=browser['category']
    assert table[category-16]==index
    rows.append({'commandId':actual['commandId'],'codes':codes,'numericKeys':numeric,
        'doubleTap':actual['doubleTap'],'category':category,'nativeMoveIndex':index,
        'routeId':route['route_id'],'npzPathId':route['npz_path_id'],
        'npzSourceSha256':clip['source_sha256'],
        'oldIncorrectIndex':category-16,'oldIncorrectCommandId':by_index[category-16]})
fixture={'schema':'rek.saved_controls.native_move_contract.v1','sources':pins,
    'derivation':'Original saved commandId and numeric keys joined to original Keyboard getters and native routes/clip source names; independently cross-checked against policy category table. No corrected app source used.',
    'category16Through32NativeMoveIndices':table,'bindings':rows,
    'doubleTapWindowMs':300,'windowProvenance':'Original KeyboardControlScheme constructor default only; serialized seed tuning is not established by saved profile.'}
receipt=write('EXPECTED-MOVES.json',fixture)

# Query only the six exact value names in the known controls-only export.
checks=[]
with winreg.OpenKey(winreg.HKEY_CURRENT_USER,r'Software\REK\REK',0,winreg.KEY_READ) as key:
    for original in source['values']:
        assert original['name'].startswith('rek.controls.')
        value,kind=winreg.QueryValueEx(key,original['name'])
        if original['kind']=='Binary':
            expected=base64.b64decode(original['encoded'])
            assert kind==winreg.REG_BINARY
            observed_bytes=value
        else:
            assert original['kind']=='DWord' and kind==winreg.REG_DWORD
            expected=(int(original['encoded']) & 0xffffffff).to_bytes(4,'little')
            observed_bytes=int(value).to_bytes(4,'little')
        checks.append({'name':original['name'],'kind':original['kind'],
            'bytes':len(observed_bytes),'expectedSha256':sha(expected),
            'currentSha256':sha(observed_bytes),'exactMatch':observed_bytes==expected})
registry_receipt=write('CURRENT-WINDOWS-CONTROLS.json',{
    'schema':'rek.controls.selected_registry_read.v1',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'registryKey':r'HKEY_CURRENT_USER\Software\REK\REK',
    'scope':'Only six names from original controls export. No registry enumeration, writes, auth or unrelated value reads.',
    'sourceSha256':pins['windows_export']['sha256'],'allSixExact':all(c['exactMatch'] for c in checks),'checks':checks})
print(json.dumps({'fixture':receipt,'registry':registry_receipt,'allSixExact':all(c['exactMatch'] for c in checks)},indent=2))
