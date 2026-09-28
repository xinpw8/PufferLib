from pathlib import Path
import json,hashlib,difflib,datetime
root=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=root/'original';staged=root/'source-r1'
entries=[];diff=[]
for p in sorted(staged.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(staged);old=original/rel
    if old.is_file() and sha(old)==sha(p):continue
    entries.append({'relative_path':rel.as_posix(),'source_path':str(p),'old_sha256':sha(old) if old.exists() else None,'new_sha256':sha(p),'bytes':p.stat().st_size})
    diff.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],p.read_text().splitlines(True),fromfile='original/'+rel.as_posix(),tofile='source-r1/'+rel.as_posix()))
snapshot=json.loads((root/'SNAPSHOT.json').read_text())
worktree=Path(snapshot['source_root'])
changed=[r['path'] for r in snapshot['source_files'] if sha(worktree/r['path'])!=r['sha256']]
assert not changed,changed
sources=[]
for name in ['SonicPolicyRunner.txt','RobotInputController.txt','SonicPolicyRunner_NestedType_StateRingBuffer.txt','RobotInputController_NestedType__WaitForSubclassDependencies_d__82.txt']:
    p=Path(r'C:\rekagent\work\controller-audit-isil\IsilDump\REKApp\REKApp')/name
    sources.append({'path':str(p),'sha256':sha(p)})
map={'schema':'rek.native_clone.controller_source_map.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshot_sha256':sha(root/'SNAPSHOT.json'),
    'entries':entries,'worktree_110_files_unchanged':not changed,'recovered_source_pins':sources,
    'rebuild_translation_units':['ocean/rek_g1/sonic_motion_composer_native.c via g1_cuda_device.cu','ocean/rek_g1/g1_semantic_scheduler_cuda.cu','ocean/rek_g1/native5/motion_assets.cu','ocean/rek_g1/native5/robot_state.cu'],
    'runtime_integration_owned_by_parent':True}
(root/'SOURCE-MAP.json').write_text(json.dumps(map,indent=2)+'\n')
(root/'controller.patch').write_text(''.join(diff),newline='\n')
comparison=json.loads((root/'HISTORY-COMPARISON.json').read_text())
validation={'schema':'rek.native_clone.controller_validation.v1','source_map_sha256':sha(root/'SOURCE-MAP.json'),'patch_sha256':sha(root/'controller.patch'),
    'cpu_reset_checks':216,'cpu_existing_scheduler_checks':21839,'cpu_existing_dispatch_equivalence_ticks':1200,
    'original_decoder_values_compared':comparison['history_values_compared'],'original_decoder_bit_mismatches':comparison['history_bit_mismatch_count'],
    'original_heading_values_compared':comparison['heading_values_compared'],'original_heading_bit_mismatches':comparison['heading_bit_mismatch_count'],
    'heading_exact_parity':False,'heading_max_absolute_angle_error_radians':1.1920928955078125e-7,
    'heading_max_absolute_quaternion_component_error':5.960464477539063e-8,
    'original_reset_component_reference':'Original composer640-row comparison previously passed92800 mapped joint values and measured layer/reset states; this integration uses that same Reset extension.',
    'callback_order_proven':False,'callback_order_required_explicit':True,'gpu_runs':0,'production_edits':0,
    'files':[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file() and p.parts[len(root.parts)] in ['tests','build-cpu-r1'] and p.name!='output-hashes.sha256']}
(root/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps({'changed_source_files':len(entries),'source_map_sha256':sha(root/'SOURCE-MAP.json'),'validation_sha256':sha(root/'VALIDATION.json'),'unchanged_worktree_files':len(snapshot['source_files'])},indent=2))
