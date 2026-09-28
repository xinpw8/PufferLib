"""Read-only plan by default. --prepare creates one fresh task-owned session only."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

SOURCE = Path('/home/spark-advantage/codexrook-runtime/rek-core-referee-20260924-r1')
RUNTIME = SOURCE.parent
SESSION = Path('/home/spark-advantage/rek-training/rek-parity-continuation-20260928-r1/unity-slerp-r2')
BASE_IMAGE_ID = 'sha256:4166bcf2f07dfc7c392ca8b2f8e8a69351700bf37f734b605f441a6d122b9f92'
FILES = ['REK.exe','GameAssembly.dll','UnityPlayer.dll','UnityCrashHandler64.exe','baselib.dll',
         'winhttp.dll','doorstop_config.ini','.doorstop_version','DirectML.dll','dstorage.dll','dstoragecore.dll']
DIRECTORIES = ['REK_Data','dotnet','D3D12','BepInEx/core','BepInEx/interop','BepInEx/unity-libs']
PINS = {'UnityPlayer.dll':'277953a7035b1633c239904853bfbea7b2948937ef5567e70c1911c260dd1414',
        'BepInEx/interop/UnityEngine.CoreModule.dll':'3ac45305a21f5107c0a9e813c48503cf27944c15ab9f6495396f20f61840104a',
        'REK.exe':'5fe6a5c3da371cb7b75a1795eb073b2e69ec718197cf68826bdcd461dcd986c1',
        'GameAssembly.dll':'6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412',
        'REK_Data/il2cpp_data/Metadata/global-metadata.dat':'e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd',
        'BepInEx/interop/REKApp.dll':'faa94fb58e24fda95e2c06810e28b9eb2d6d9f9f8327541976a0dc1011f646d2',
        'BepInEx/config/BepInEx.cfg':'eb9c78ef7da7af7f0228aa74f23a4c1b5197850804aec66873bfef9c7310b675'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def create_command():
    session=SESSION.as_posix()
    assert sha(RUNTIME/'box64-upstream-20260924-r1/bin/box64')=='12a50a0f629f1ddeb08524c8f7399829e0b7101f79f4e20c094376c73a1af7ae','Translator pin changed'
    args=['docker','create','--name','rek-slerp-oracle-20260928-r2','--runtime','runc',
          '--network','none','--ipc','private','--cap-drop','ALL','--security-opt','no-new-privileges',
          '--read-only','--user','1000:1000','--pids-limit','256','--cpus','4','--memory','8g','--memory-swap','8g',
          '--tmpfs','/tmp:rw,nosuid,size=512m,mode=1777','--tmpfs','/run:rw,nosuid,size=16m,mode=755',
          '--env','DISPLAY=:99','--env','HOME=/session/home','--env','WINEPREFIX=/session/wineprefix',
          '--env','NVIDIA_VISIBLE_DEVICES=void','--mount',f'type=bind,src={session},dst=/session',
          '--mount',f'type=bind,src={session}/input,dst=/session/input,readonly',
          '--mount',f'type=bind,src={session}/game,dst=/session/game,readonly',
          '--mount',f'type=bind,src={session}/game/BepInEx,dst=/session/game/BepInEx']
    # Runtime inputs are private byte-verified copies with read-only mounts.
    # Only private BepInEx config/plugins/cache/logs remain writable.
    for name in ['core','interop','unity-libs']:
        args+=['--mount',f'type=bind,src={session}/game/BepInEx/{name},dst=/session/game/BepInEx/{name},readonly']
    for name in ['box64','wine-11.13','x64root-ubuntu-24.04']:
        args+=['--mount',f'type=bind,src={(RUNTIME/('box64-upstream-20260924-r1' if name=='box64' else name)).as_posix()},dst=/opt/codexrook/{name},readonly']
    args+=['--mount','type=bind,src=/home/spark-advantage/rek-training/rek-parity-continuation-20260928-r1/revision-r2/isolation/entrypoint.sh,dst=/usr/local/bin/rek-original-functions-entrypoint,readonly']
    args+=['sha256:5fb2c698c065d4502b3c09f3b847efac6aa5428541c4b4f26110e3a530cae879']
    return args
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--install-plugin',type=Path)
    p.add_argument('--plugin-sha256')
    args=p.parse_args()
    if args.prepare and args.install_plugin:p.error('Prepare and plugin installation are separate stages')
    if not args.prepare and not args.install_plugin:
        print(json.dumps({'session':SESSION.as_posix(),'base_image_id':BASE_IMAGE_ID,'copy_files':FILES,
          'copy_directories':DIRECTORIES,'copy_config':'BepInEx/config/BepInEx.cfg',
          'excluded':['original Wine prefix/home/auth/control preferences','all original BepInEx plugins','evidence/logs/cache','network access','host X sockets','GPU devices'],
          'docker_create_argv':create_command(),'launch_performed':False},indent=2));return
    if args.install_plugin:
        if os.getuid()!=1000:raise SystemExit('Stage as the existing UID1000 user; container runs UID1000')
        if not args.plugin_sha256 or len(args.plugin_sha256)!=64:p.error('Exact plugin SHA256 is required')
        if args.install_plugin.name in ['RekUiBridgeAgent.dll','RekEvidenceRecorder.dll']:p.error('Original control/recorder plugins excluded')
        if not args.install_plugin.name.endswith('.dll'):p.error('Harness DLL required')
        if not (SESSION/'STAGED.json').is_file() or (SESSION/'PLUGIN.json').exists() or (SESSION/'run.started').exists():
            raise SystemExit('Session is not a fresh staged-only package')
        target=SESSION/'game/BepInEx/plugins'
        if any(target.iterdir()):raise SystemExit('Plugin directory must be empty')
        if sha(args.install_plugin)!=args.plugin_sha256:raise SystemExit('Harness plugin hash mismatch')
        dest=target/args.install_plugin.name
        with dest.open('xb') as out,args.install_plugin.open('rb') as src:shutil.copyfileobj(src,out)
        assert sha(dest)==args.plugin_sha256
        (SESSION/'PLUGIN.json').write_text(json.dumps({'path':str(dest.relative_to(SESSION)),'sha256':sha(dest),'bytes':dest.stat().st_size},indent=2)+'\n')
        print('Harness plugin staged. No container or game started.');return
    if SESSION.exists():raise SystemExit('Fresh destination required; existing data will not be overwritten')
    if os.getuid()!=1000:raise SystemExit('Stage as the existing UID1000 user; container runs UID1000')
    for rel,expected in PINS.items():
        if sha(SOURCE/rel)!=expected:raise SystemExit('Original input hash mismatch: '+rel)
    cfg=(SOURCE/'BepInEx/config/BepInEx.cfg').read_text()
    if 'UpdateInteropAssemblies = false' not in cfg or 'PreloadIL2CPPInteropAssemblies = false' not in cfg:
        raise SystemExit('Interop regeneration/preload guards absent')
    image=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','codexrook-rek-runtime:gb10'],text=True).strip()
    if image!=BASE_IMAGE_ID:raise SystemExit('Base image identity changed')
    selected=[SOURCE/x for x in FILES]+[SOURCE/'BepInEx/config/BepInEx.cfg']
    for rel in DIRECTORIES:
        for f in sorted((SOURCE/rel).rglob('*')):
            if f.is_symlink():raise SystemExit('Unexpected source symlink: '+str(f))
            if f.is_file():selected.append(f)
    if any(f.is_symlink() for f in selected):raise SystemExit('Source symlinks require separate review')
    if shutil.disk_usage(SESSION.parent if SESSION.parent.exists() else SESSION.parent.parent).free<4*1024**3:
        raise SystemExit('Less than4GiB free')
    SESSION.mkdir(parents=True,mode=0o700)
    for rel in ['game/BepInEx/plugins','game/BepInEx/patchers','home','wineprefix','out','input']:
        (SESSION/rel).mkdir(parents=True,mode=0o700,exist_ok=True)
    receipts=[]
    for f in selected:
        rel=f.relative_to(SOURCE);dest=SESSION/'game'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        before=sha(f)
        with f.open('rb') as src,dest.open('xb') as out:shutil.copyfileobj(src,out,1048576)
        if sha(dest)!=before or sha(f)!=before:raise SystemExit('Copy/source changed: '+str(rel))
        receipts.append({'path':str(dest.relative_to(SESSION)),'bytes':dest.stat().st_size,'sha256':before})
    (SESSION/'STAGED.json').write_text(json.dumps({'schema':'rek.original_functions.isolation.v1','base_image_id':BASE_IMAGE_ID,
        'original_root':str(SOURCE),'files':receipts,'original_plugins_copied':False,'original_profile_copied':False,
        'network_mode':'none','gpu_devices':False,'launch_performed':False},indent=2)+'\n')
    (SESSION/'docker-create-command.json').write_text(json.dumps(create_command(),indent=2)+'\n')
    print(json.dumps({'staged':str(SESSION),'files':len(receipts),'bytes':sum(r['bytes'] for r in receipts),'launch_performed':False}))
if __name__=='__main__':main()
