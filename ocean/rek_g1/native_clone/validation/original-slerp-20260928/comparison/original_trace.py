"""Strict adapter for the independent original-binary harness, with explicit order conversion.

This module never runs the game or changes native inputs. The loaded-array hashes
must demonstrate the declared map before a normalized comparison is allowed.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct

from compare_traces import require, sha, fp, vector, fields, ORDERS, OFFSETS, ordered_bits, parse_integer

GAME='6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412'
META='e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd'
INTEROP='faa94fb58e24fda95e2c06810e28b9eb2d6d9f9f8327541976a0dc1011f646d2'

def stable_records(path):
    path=Path(path);before=path.stat();raw=path.read_bytes();after=path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'original trace changed during read')
    require(raw.endswith(b'\n'),'unterminated original trace')
    return raw,[json.loads(line,parse_int=parse_integer) for line in raw.splitlines()]

def boolint(value):
    require(type(value) is bool,'original Boolean expected')
    return int(value)

def fixture_agreement(original_path,root):
    """Bind input bytes and semantic protocol even though the two manifests differ."""
    original=json.loads(Path(original_path).read_bytes());native=json.loads((root/'fixture.json').read_bytes())
    require(original['schema']=='rek.original_composer.fixture.v1','wrong original fixture schema')
    want={'orders':native['orders'],'repeats':native['repeats'],'warmup_ticks':native['warmup_advances'],
          'ticks':native['ticks'],'walk_tick':native['forward_before_tick'],'release_tick':native['idle_before_tick'],
          'offsets':native['reference_offsets'],'controller_rate_hz':native['controller_rate_hz'],
          'locomotion_speed_scale':native['locomotion_scale']}
    require(original['protocol']==want,'explicit fixture protocol differs')
    require(original['joint_config']['sha256']==sha(root/'assets/sonic_config'),'joint config identity differs')
    native_by_id={c['clip_id']:c for c in native['clips']}
    require([c['clip_id'] for c in original['clips']]==[377,370],'clip identity/order differs')
    translations={'blendInTime':'blend_in_seconds','blendOutTime':'blend_out_seconds','endFrame':'end_frame',
                  'loop':'loop','mirror':'mirror','playbackSpeed':'playback_speed','startFrame':'start_frame','yawBlend':'yaw_blend'}
    for clip in original['clips']:
        n=native_by_id[clip['clip_id']]
        require(clip['frames']==n['frames'] and clip['npz']['sha256']==n['npz_sha256'],'clip fixture identity differs')
        npz_name='idle_processed.npz' if clip['clip_id']==377 else 'walking_processed.npz'
        require(sha(root/'assets'/npz_name)==n['npz_sha256'],'local NPZ changed')
        require(clip['foot_features']['sha256']==sha(root/'assets'/f"motion_{clip['clip_id']}_foot_features.f32le"),'foot feature identity differs')
        require(clip['config']['path_id']==n['config_path_id'],'config identity differs')
        for okey,nkey in translations.items():
            require(struct.pack('<f',clip['config'][okey])==struct.pack('<f',n['config'][nkey]),'composer config differs: '+okey)
    return original,{'native_fixture_sha256':sha(root/'fixture.json'),'original_fixture_sha256':sha(original_path),
                     'protocol_equal':True,'npz_config_and_foot_features_equal':True,
                     'scope':'historical explicitly supplied assets, not a verified authoritative server asset set'}

def canonical_row(row):
    r=copy.deepcopy(row)
    for name in ('pre','post_advance','post_consume'):
        state=r[name];state['action_playing']=boolint(state['action_playing'])
        for side in ('current','from'):
            layer=state[side]
            # The frozen harness derives clip_id from config pointer, so only
            # has_config is knowable. It does not expose the clip pointer itself.
            layer['has_config']=int(layer['clip_id'] is not None)
            layer['clip_id']=0 if layer['clip_id'] is None else layer['clip_id']
            for key in ('active','heading_valid','heading_resync'):layer[key]=boolint(layer[key])
            require(type(layer['loop']) is bool and type(layer['mirror']) is bool,'original loop/mirror shape')
    for ref in r['references']:
        ref['dof_position_mujoco']=ref['dof_position_raw']
        # The internal canonical key is historical; its raw comparison is
        # explicitly labeled raw-index. No basis change occurs here.
        require(type(ref['velocity_available']) is bool and type(ref['root_position_available']) is bool,'availability flags missing')
        for available,key,width in ((ref['velocity_available'],'reference_velocity_raw',29),
                                    (ref['root_position_available'],'reference_root_position_raw',3)):
            require((ref[key] is not None)==available,'availability/value contradiction')
            if available:vector(ref[key],width)
    fields(r,('has_clip',))
    return r

def load_original(path,fixture_path,root):
    path=Path(path);root=Path(root);raw,records=stable_records(path);header=records[0];footer=records[-1]
    fixture,agreement=fixture_agreement(fixture_path,root)
    require(header['event']=='header' and header['schema']=='rek.original_composer.trace.v1','wrong original trace header')
    require(header['mode']=='sequence','smoke trace is not complete comparison')
    require((header['game_sha256'],header['metadata_sha256'],header['interop_sha256'])==(GAME,META,INTEROP),'original binary/interop pins differ')
    require(header['fixture_sha256']==agreement['original_fixture_sha256'],'original trace fixture pin differs')
    require(footer['event']=='oracle_end' and footer['success'] is True and footer['reason']=='sequence_complete' and footer['rows']==640 and footer['error'] is None,'original trace incomplete')
    require(footer['no_robot_runner_or_physics_created_by_harness'] is True,'wrong detached harness scope')
    require(footer['whole_process_physics_steps_not_instrumented'] is True,'unexpected whole-process physics claim')
    entries=[r for r in records if r['event']=='entrypoints'];require(len(entries)==1,'missing unique original method attestation')
    methods=entries[0]['methods'];names={(x['type'],x['method']) for x in methods}
    required={('REKApp.SonicMotionComposer',name) for name in ('Init','Reset','PlayAction','PlayActionImmediate','GetReferenceFrame','TryGetReferenceVelocity','TryGetReferenceRootPos','Advance','ConsumeHeadingDelta','get_HeadingClipOwnership','SetLocomotionSpeed','RegisterFootFeatures','BuildClip','LoadClip')}
    required.add(('REKApp.NpzReader','Read'));require(names==required,'original method coverage differs')
    require(all(x['module'].replace('\\','/').split('/')[-1].lower()=='gameassembly.dll' and int(x['rva'],16)>0 and len(x['first32_sha256'])==64 for x in methods),'invalid original method attestation')
    calls=footer['completed_wrapper_calls']
    for name,count in {'Init':4,'Reset':4,'PlayActionImmediate':4,'PlayAction':12,'SetLocomotionSpeed':644,
                       'Advance':708,'ConsumeHeadingDelta':708,'GetReferenceFrame':3200,
                       'TryGetReferenceVelocity':3200,'TryGetReferenceRootPos':3200,'RegisterFootFeatures':8}.items():
        require(calls.get(name)==count,'original call count differs: '+name)
    initialized=[r for r in records if r['event']=='lifecycle' and r.get('phase')=='initialized_detached']
    require([r['trial'] for r in initialized]==list(range(4)),'missing detached trial initialization')
    mapping=json.loads((root/'assets/sonic_config').read_bytes())['mujoco_to_isaaclab']
    require(sorted(mapping)==list(range(29)),'invalid config map')
    for r in initialized:
        require(r['active'] is False and r['enabled'] is False and r['automatic_Start_not_invoked'] is True,'component lifecycle differs')
        require(r['mujoco_to_isaaclab']==mapping,'actual original map differs from pinned configuration')
    inv=[mapping.index(i) for i in range(29)]
    clips=[r for r in records if r['event']=='clip_provisioned'];require([r['clip_id'] for r in clips]==[377,370]*4,'clip provisioning coverage differs')
    expected_clips={r['clip_id']:r for r in fixture['clips']};loaded=[]
    for r in clips:
        f=expected_clips[r['clip_id']]
        require(r['npz_sha256']==f['npz']['sha256'] and r['foot_feature_sha256']==f['foot_features']['sha256'] and r['config']==f['config'],'original provisioned asset/config differs')
        require(r['frames']==f['frames'] and fp(r['fps'])[0]==50 and r['root_quaternion_order']=='wxyz','original clip shape differs')
        b=(root/'assets'/f"motion_{r['clip_id']}_dof_position.f32le").read_bytes()
        require(len(b)==f['frames']*29*4,'native clip shape changed')
        reordered=b''.join(b[(frame*29+j)*4:(frame*29+j+1)*4] for frame in range(f['frames']) for j in mapping)
        expected=hashlib.sha256(reordered).hexdigest()
        loaded.append({'clip_id':r['clip_id'],'original_loaded_dof_sha256':r['loaded_dof_sha256'],
                       'expected_map_applied_dof_sha256':expected,'raw_native_dof_sha256':hashlib.sha256(b).hexdigest(),
                       'map_applied_bytes_match':r['loaded_dof_sha256']==expected,
                       'original_loaded_root_quat_sha256':r['loaded_root_quat_sha256']})
    rows=[canonical_row(r) for r in records if r['event']=='row'];require(len(rows)==640,'original row count')
    for i,r in enumerate(rows):
        require((r['order'],r['repeat'],r['tick'],r['trial'])==(ORDERS[i//320],(i//160)%2,i%160,i//160),'original row identity/alignment differs')
    ends=[r for r in records if r['event']=='trial_end']
    require([r['trial'] for r in ends]==list(range(4)) and all(r['rows']==160 and r['active'] is False and r['enabled'] is False for r in ends),'original trial terminal differs')
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'header':header,
            'fixture_agreement':agreement,'rows':rows,'records':records,'declared_map':mapping,'inverse_map':inv,
            'loaded_clip_checks':loaded,'all_loaded_dof_arrays_match_declared_map':all(r['map_applied_bytes_match'] for r in loaded),
            'map_evidence':'Recovered SonicMotionComposer.BuildClip lines20535,20619–20659; GetReferenceFrame copy at2401. ISIL-to-current-binary generation provenance is unverified; actual loaded-array hashes independently check the mapping.',
            'scope':'Detached original compiled client component. Whole-process physics steps were not instrumented.'}

def permute(vector_value,indices):
    return {key:[vector_value[key][i] for i in indices] for key in ('values','bits')}

def normalized_rows(rows,inverse):
    result=copy.deepcopy(rows)
    for row in result:
        for ref in row['references']:ref['dof_position_mujoco']=permute(ref['dof_position_raw'],inverse)
    return result

def auxiliary_comparison(native,original,inverse,mapping_verified):
    """Measure estimates separately; never promote them to API implementation parity."""
    stats={'values':0,'different_bits':0,'maximum_absolute_error':0.,'sum_squared_error':0.,'maximum_ulp_difference':0}
    available=0;root_available=0;first=None
    for n,o in zip(native,original):
        for nr,ref in zip(n['references'],o['references']):
            available+=int(ref['velocity_available']);root_available+=int(ref['root_position_available'])
            if not ref['velocity_available'] or not mapping_verified:continue
            estimate=vector(nr['native_reference_velocity_estimate'],29)
            original_velocity=vector(permute(ref['reference_velocity_raw'],inverse),29)
            for j,((a,ab),(b,bb)) in enumerate(zip(estimate,original_velocity)):
                error=abs(a-b);stats['values']+=1;stats['different_bits']+=int(ab!=bb)
                stats['maximum_absolute_error']=max(stats['maximum_absolute_error'],error)
                stats['sum_squared_error']+=error*error;stats['maximum_ulp_difference']=max(stats['maximum_ulp_difference'],abs(ordered_bits(ab)-ordered_bits(bb)))
                if ab!=bb and first is None:first={k:n[k] for k in ('order','repeat','tick','trial')}|{'frames_ahead':ref['frames_ahead'],'joint':j,'native_estimate':a,'original_api':b}
    stats['rms_error']=math.sqrt(stats.pop('sum_squared_error')/stats['values']) if stats['values'] else None
    stats['first_difference']=first
    return {'references':3200,'original_velocity_available_references':available,
            'native_velocity_status':'finite-difference estimate only; no native implementation of original velocity API',
            'velocity_estimate_vs_original_api':stats if stats['values'] else None,
            'velocity_basis_transform_applied':mapping_verified,
            'original_root_position_available_references':root_available,'native_root_position_status':'unsupported; not numerically compared',
            'joint_order_transform_note':'Velocity uses the same explicit original-to-input-column map, only when loaded joint arrays establish that map.'}
