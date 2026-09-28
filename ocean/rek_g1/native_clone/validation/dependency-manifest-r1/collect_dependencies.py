"""Read-only Spark inventory. Emits JSON; never loads a GPU or launches the worker."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, platform, re, shutil, subprocess

def collect(root, build_name, source_name, run_name, app_name, tools_name):
    root=Path(root); build=root/build_name; source=root/source_name; run=root/run_name
    files={}; expected_checks=[]; errors=[]
    def add(path,kind,expected=None):
        path=Path(path); key=str(path)
        if key not in files:
            if not path.is_file():
                errors.append({'path':key,'error':'required file missing'});return None
            before=path.stat(); digest=hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda:f.read(4*1024*1024),b''):digest.update(b)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError('Changed during inventory: '+key)
            files[key]={'path':key,'resolved_path':str(path.resolve()),'bytes':after.st_size,
                        'sha256':digest.hexdigest(),'categories':[]}
        row=files[key]
        if kind not in row['categories']:row['categories'].append(kind)
        if expected is not None:
            ok=row['sha256']==expected.lower();expected_checks.append({'path':key,'expected_sha256':expected,'matched':ok})
            if not ok:errors.append({'path':key,'error':'recorded digest mismatch'})
        return row
    def tree(path,kind):
        path=Path(path)
        if not path.is_dir():errors.append({'path':str(path),'error':'required directory missing'});return
        for p in sorted(path.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts:add(p,kind)
    def pins(path,kind):
        add(path,'provenance_record')
        result=[]
        for line in Path(path).read_text().splitlines():
            m=re.fullmatch(r'([a-fA-F0-9]{64})\s+[*]?(.*)',line)
            if m:
                p=Path(m[2]);p=p if p.is_absolute() else Path(path).parent/p
                result.append(add(p,kind,m[1]))
        return result
    worker=json.loads((run/'worker.json').read_text());server=json.loads((run/'server.json').read_text())
    backend=server['backends'][0];env=backend['env']
    binary=Path(backend['executable'])
    assert binary==build/'rek-native-clone'
    for n in ['worker.json','server.json','identity.json']:add(run/n,'runtime_configuration')
    identity=json.loads((run/'identity.json').read_text())
    for p,sha in identity['filePins'].items():add(p,'run_identity_pin',sha)
    assert (build/'exit-code.txt').read_text().strip()=='0'
    add(build/'exit-code.txt','build_receipt');fresh=pins(build/'outputs.sha256','built_artifact')
    reused=pins(build/'reused-inputs.sha256','reused_object')
    tree(source,'current_source_snapshot');tree(root/app_name,'viewer_source_and_assets')
    tree(root/tools_name,'build_and_launch_tools')
    for k in ['model_path','physics_export_path','controller_encoder_path','controller_decoder_path','render_model_path']:
        add(worker[k],'runtime_asset_'+k)
    tree(worker['assets_path'],'motion_and_physics_asset_bundle')
    tree(worker['motion_features_path'],'motion_foot_feature_bundle')
    catalog_path=Path(env['REK_MUJOCO_KERNEL_CATALOG']);add(catalog_path,'kernel_catalog')
    catalog=json.loads(catalog_path.read_text())
    for module in catalog['modules']:
        for key in ['source','ptx','meta']:
            add(module[key],'warp_kernel_'+key,module[key+'Sha256'])
    add(env['REK_MUJOCO_CONDITIONAL_PTX'],'conditional_ptx',env['REK_MUJOCO_CONDITIONAL_SHA256'])

    native=Path('/home/spark-advantage/rek-training/native-mujoco-gpu-20260914')
    base=Path('/home/spark-advantage/rek-training/physical-bot1-integration-20260921-r1/build-r1')
    measurement=Path('/home/spark-advantage/rek-training/physical-measurement-integration-20260921-r1')
    native_g1=native/'native-source/ocean/rek_g1'
    for p in [base/'build-script.sh',base/'reused-object-pins.sha256',base/'source-tree-hashes.sha256',
              base/'snapshot.json',base/'build-commands.txt',
              native/'build-native-v2/build-source-manifest.txt',native/'build-native-v2-command.txt',
              native_g1/'native5/build_native.sh',measurement/'build-r2.sh',measurement/'build-provenance.txt']:
        add(p,'historical_build_provenance')
    provenance=[]
    for row in reused:
        if row is None:continue
        obj=Path(row['path']);name=obj.name
        if name=='measurement.o':p=measurement/'source-r2/ocean/rek_g1/native5/measurement.cu';rel='ocean/rek_g1/native5/measurement.cu'
        elif name=='cJSON.o':p=native/'native-source/vendor/cJSON.c';rel='vendor/cJSON.c'
        elif name.startswith('mujoco_'):rel='ocean/rek_g1/native5/mujoco_gpu/'+name[7:-2]+'.cpp';p=native/'native-source'/rel
        elif name in ['physics.o','sonic_controller.o']:rel='ocean/rek_g1/native5/'+name[:-2]+'.cu';p=native/'native-source'/rel
        else:
            stem=name[:-2].removesuffix('.device').removesuffix('.host')
            rel='ocean/rek_g1/'+stem+('.cu' if stem.endswith('_cuda') else '.c');p=native/'native-source'/rel
        old=add(p,'reused_translation_unit_source');current=source/rel
        new=add(current,'current_reused_translation_unit') if current.is_file() else None
        provenance.append({'object':row['path'],'object_sha256':row['sha256'],'recorded_build_translation_unit':str(p),
            'translation_unit_sha256':old['sha256'] if old else None,'current_source_path':str(current) if new else None,
            'translation_unit_bytes_equal_current':new['sha256']==old['sha256'] if new and old else None,
            'scope':'Object digest reverified against reused-inputs; historical build script supplies source mapping. No new compilation or header-closure equivalence claim.'})

    # ldd executes the trusted platform loader in trace mode, never worker main.
    ld=subprocess.run(['ldd',str(binary)],capture_output=True,text=True,check=True)
    if 'not found' in ld.stdout:errors.append({'error':'unresolved shared library','detail':ld.stdout})
    shared=[]
    for line in ld.stdout.splitlines():
        m=re.search(r'(?:=>\s+)?(/\S+)\s+\(',line)
        if m:shared.append(add(m[1],'resolved_shared_library'))
    vendors=[]
    for p in sorted(Path('/usr/share/glvnd/egl_vendor.d').glob('*.json')):
        add(p,'egl_vendor_configuration');vendors.append(json.loads(p.read_text()))
    ldconfig=subprocess.run(['ldconfig','-p'],capture_output=True,text=True,check=True).stdout
    for vendor in vendors:
        library=vendor.get('ICD',{}).get('library_path')
        if library:
            matches=[line.rsplit(' => ',1)[1] for line in ldconfig.splitlines() if line.strip().startswith(library+' ') and ' => ' in line]
            for path in matches:
                add(path,'egl_dynamic_vendor_library')
                vendor_ld=subprocess.run(['ldd',path],capture_output=True,text=True,check=True)
                for line in vendor_ld.stdout.splitlines():
                    m=re.search(r'(?:=>\s+)?(/\S+)\s+\(',line)
                    if m:add(m[1],'egl_dynamic_vendor_dependency')
    node=shutil.which('node');nvcc='/usr/local/cuda/bin/nvcc';compiler=shutil.which('g++')
    versions={}
    for name,path,args in [('node',node,['--version']),('nvcc',nvcc,['--version']),('g++',compiler,['--version'])]:
        if path:
            add(path,'host_toolchain');versions[name]=subprocess.run([path,*args],capture_output=True,text=True,check=True).stdout.strip()
    for path in ['/etc/os-release','/proc/driver/nvidia/version']:
        if Path(path).is_file():versions[path]=Path(path).read_text().strip()
    categories={}
    for row in files.values():
        for cat in row['categories']:
            d=categories.setdefault(cat,{'files':0,'bytes':0});d['files']+=1;d['bytes']+=row['bytes']
    return {'schema':'rek.native_clone.dependency_inventory.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'root':str(root),'bindings':{'build':build_name,'source':source_name,'run':run_name,'app':app_name,'tools':tools_name},
        'architecture':platform.machine(),'versions':versions,'environment':env,'worker_config':worker,
        'binary':files[str(binary)],'files':list(files.values()),'recorded_pin_checks':expected_checks,
        'reused_object_source_provenance':provenance,'category_totals_nonexclusive':categories,
        'unique_paths':len(files),'bytes_across_paths':sum(r['bytes'] for r in files.values()),'errors':errors,
        'closure_status':'required paths and recorded pins verified' if not errors else 'incomplete',
        'limitations':['This is a host-bound dependency inventory, not a portable or clean-room build.',
            'No GPU, worker execution, compilation, dependency installation, or large dependency copying performed.',
            'Reused object source file equality does not establish identical historical transitive headers/toolchain.',
            'GLVND vendor libraries are recorded statically; runtime-selected driver paths may require additional host libraries.',
            'Generated Warp PTX/source paths reside in the existing user cache and must be retained or explicitly relocated with updated configuration.',
            'No full CUDA toolkit, operating-system package, or kernel-driver redistribution closure is claimed.']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    for n in ['build','source','run','app','tools']:p.add_argument('--'+n,required=True)
    a=p.parse_args();print(json.dumps(collect(a.root,a.build,a.source,a.run,a.app,a.tools),indent=2))
