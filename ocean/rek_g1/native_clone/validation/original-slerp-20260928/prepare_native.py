"""Stage immutable prior composer fixture plus instrumentation; does not execute it."""
from pathlib import Path
import difflib,hashlib,json,shutil

root=Path(__file__).resolve().parent
prior=Path(r'C:\rekagent\work\rek-original-functions-20260927-r1\comparison')
target=root/'native';target.mkdir(exist_ok=False)
for name in ['source','assets']:shutil.copytree(prior/name,target/name)
for name in ['fixture.json','fixture_config.h','native_trace.c']:shutil.copyfile(prior/name,target/name)
original=(target/'native_trace.c').read_text();(target/'native_trace.original.c').write_bytes((prior/'native_trace.c').read_bytes())
text=original
def edit(old,new):
    global text
    assert text.count(old)==1,old;text=text.replace(old,new)
edit('static FILE* out;','#include "slerp_boundary.h"\n\nstatic FILE* out;')
edit('if(argc!=3){fprintf(stderr,"usage: native-trace ASSETS NEW_JSONL\\n");return 2;}',
     'if(argc!=4&&argc!=5){fprintf(stderr,"usage: native-trace ASSETS NEW_JSONL NEW_CALL_LOG [ORACLE_TABLE]\\n");return 2;}\n    boundary_open(argv[3],argc==5?argv[4]:NULL);')
edit('SonicMotionComposerNativeBackends b={sonic_motion_composer_libm_candidate_quaternion_slerp,',
     'SonicMotionComposerNativeBackends b={boundary_slerp,')
edit('SonicMotionEntryMatcherNative m;','boundary_trial=order*2+rep;boundary_tick=-1;boundary_phase="initialization";\n        SonicMotionEntryMatcherNative m;')
edit('for(int t=0;t<17;t++){','boundary_phase="warmup";\n        for(int t=0;t<17;t++){')
edit('if(order==0)ok(', 'boundary_phase="reset";\n        if(order==0)ok(')
edit('if(t==50)ok(', 'boundary_tick=t;boundary_phase="action";\n            if(t==50)ok(')
edit('ok(sonic_motion_composer_native_build_reference_rows(&c,&timing,NULL,&ref));','boundary_phase="references";\n            ok(sonic_motion_composer_native_build_reference_rows(&c,&timing,NULL,&ref));')
edit('ok(sonic_motion_composer_native_advance(&c,&adv));fprintf(out,','boundary_phase="advance";\n            ok(sonic_motion_composer_native_advance(&c,&adv));fprintf(out,')
edit('if(ferror(out)||fclose(out))return 2;','if(ferror(out)||fclose(out))return 2;\n    boundary_close();')
(target/'native_trace.c').write_text(text,newline='\n')
shutil.copyfile(root/'slerp_boundary.h',target/'slerp_boundary.h')
(target/'instrumentation.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='native_trace.original.c',tofile='native_trace.c')),newline='\n')
pins=[{'path':p.relative_to(target).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(target.rglob('*')) if p.is_file()]
(target/'SOURCE.json').write_text(json.dumps({'schema':'rek.slerp_boundary.native_stage.v1','prior':str(prior),'execution_performed':False,'files':pins},indent=2)+'\n')
print('Staged native source/assets only; execution not performed.')
