"""Explicit isolated stages; default prints the plan and performs no mutations."""
import argparse
import datetime
import json
import subprocess
from pathlib import Path
from prepare_runtime import BASE_IMAGE_ID, SESSION, create_command, sha, verify_translator

ROOT=Path(__file__).resolve().parent
IMAGE='rek-original-functions:20260927-r1'
NAME='rek-native-clone-history-20260927-r1'
def read_json_command(argv):return json.loads(subprocess.check_output(argv,text=True))
def write_new(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def verify_container(c):
    translator=verify_translator()
    h=c['HostConfig'];cfg=c['Config']
    assert c['Name']=='/'+NAME
    assert h['NetworkMode']=='none' and h['IpcMode']=='private' and not h['PidMode']
    assert not h['Privileged'] and h['Runtime']=='runc' and h['ReadonlyRootfs']
    assert not h.get('Devices') and not h.get('DeviceRequests')
    assert 'ALL' in h['CapDrop'] and any(s.startswith('no-new-privileges') for s in h['SecurityOpt'])
    assert cfg['User']=='1000:1000'
    image=read_json_command(['docker','image','inspect',c['Image']])[0]
    assert image['Config']['Labels']['rek.original-functions.base-image-id']==BASE_IMAGE_ID
    expected={}
    argv=create_command()
    for i,x in enumerate(argv):
        if x!='--mount':continue
        fields=argv[i+1].split(',');kv=dict(v.split('=',1) for v in fields if '=' in v)
        expected[kv['dst']]=(kv['src'],'readonly' not in fields)
    mounts={m['Destination']:(m['Source'],m['RW']) for m in c['Mounts'] if m['Type']=='bind'}
    assert mounts==expected, 'Unexpected bind mount'
    assert not h.get('PortBindings'), 'No published ports permitted'
    env=dict(x.split('=',1) for x in cfg['Env'] if '=' in x)
    assert env['DISPLAY']==':99' and env['WINEPREFIX']=='/session/wineprefix' and env['HOME']=='/session/home'
    assert env['NVIDIA_VISIBLE_DEVICES']=='void'
    assert not any(k in env for k in ['COMPlus_EnableAVX','DOTNET_EnableAVX','BOX64_AVX','BOX64_DYNAREC_NATIVEFLAGS']), 'Baseline diagnostic flags required'
    return {'id':c['Id'],'image':c['Image'],'network':'none','ipc':'private','pid_namespace':'private',
            'privileged':False,'gpu_devices':False,'host_X_mount':False,'original_profile_mount':False,
            'original_game_writable_mount':False,'bind_mounts':mounts,'state':c['State']['Status'],'translator':translator}
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--build-image',action='store_true')
    p.add_argument('--freeze-input',action='store_true')
    p.add_argument('--create',action='store_true')
    p.add_argument('--start',action='store_true')
    args=p.parse_args()
    if not any(vars(args).values()):
        print(json.dumps({'create_argv':create_command(),'launch_performed':False},indent=2));return
    if args.build_image:
        base=read_json_command(['docker','image','inspect','codexrook-rek-runtime:gb10'])[0]
        assert base['Id']==BASE_IMAGE_ID,'Base image changed'
        existing=subprocess.run(['docker','image','inspect',IMAGE],capture_output=True)
        assert existing.returncode!=0,'Derivative image already exists; preserve it and review before reuse'
        source_pins={name:sha(ROOT/name) for name in ['Dockerfile','entrypoint.sh']}
        subprocess.run(['docker','build','--pull=false','--tag',IMAGE,str(ROOT)],check=True)
        image=read_json_command(['docker','image','inspect',IMAGE])[0]
        write_new('image-build-receipt.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'base_image_id':BASE_IMAGE_ID,'image_id':image['Id'],'sources':source_pins,'started_game':False})
    if args.freeze_input:
        assert (SESSION/'STAGED.json').is_file() and not (SESSION/'run.started').exists()
        assert (SESSION/'input/fixtures.json').is_file(),'The plugin fixture file must be input/fixtures.json'
        entries=[]
        for f in sorted((SESSION/'input').rglob('*')):
            assert not f.is_symlink(),'Fixture symlinks are not accepted'
            if f.is_file():entries.append({'path':str(f.relative_to(SESSION)),'bytes':f.stat().st_size,'sha256':sha(f)})
        with (SESSION/'INPUT.json').open('x') as out:json.dump({'files':entries},out,indent=2);out.write('\n')
    if args.create:
        assert all((SESSION/x).is_file() for x in ['STAGED.json','PLUGIN.json','INPUT.json'])
        assert not (SESSION/'run.started').exists()
        assert subprocess.run(['docker','container','inspect',NAME],capture_output=True).returncode!=0,'Container already exists'
        subprocess.run(create_command(),check=True)
        c=read_json_command(['docker','container','inspect',NAME])[0]
        verified=verify_container(c)
        write_new('container-created-receipt.json',verified)
        print(json.dumps(verified))
    if args.start:
        c=read_json_command(['docker','container','inspect',NAME])[0]
        verified=verify_container(c)
        assert verified['state']=='created','Only a fresh never-started container may be started'
        assert not (SESSION/'run.started').exists()
        write_new('container-prestart-receipt.json',verified)
        # This is the sole operation that launches the isolated harness. The
        # entrypoint bounds its Wine/Unity command to180s plus10s termination.
        completed=subprocess.run(['docker','start','--attach',NAME])
        after=read_json_command(['docker','container','inspect',NAME])[0]
        write_new('container-finished-receipt.json',{'id':after['Id'],'state':after['State'],'docker_attach_exit':completed.returncode})
        raise SystemExit(completed.returncode)
if __name__=='__main__':main()
