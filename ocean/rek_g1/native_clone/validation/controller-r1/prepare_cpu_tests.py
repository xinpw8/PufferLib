from pathlib import Path
import json,struct,shutil
root=Path(__file__).parent
tests=root/'tests';tests.mkdir(exist_ok=True)
old=Path(r'C:\rekagent\work\rek-residual-parity-20260927-r1\motion-reset-idle-variant\source-r2\test_idle_reset_cpu.cpp').read_text()
prefix=old.split('void identical(')[0]
body=r'''
void test_ordered(Assets& assets) {
    for(int route: {0,1,2,7})for(auto order: {REK_G1_RESET_RUNNER_THEN_INPUT,REK_G1_RESET_INPUT_THEN_RUNNER}) {
        Fixture f(assets);
        require(sonic_motion_composer_native_play_action(&f.composers[0],&assets.routes[route].clip,&assets.routes[route].config)==0,"set prior route");
        f.composers[0].current_layer.cursor=2.75f;
        f.composers[0].current_layer.speed=0.625f;
        f.composers[0].pending_heading_delta=0.3f;
        f.rows[0].direct_native_active=1;
        auto expected=f.composers[0];
        if(order==REK_G1_RESET_INPUT_THEN_RUNNER)
            require(sonic_motion_composer_native_play_action(&expected,&assets.routes[0].clip,&assets.routes[0].config)==0,"original Input PlayIdle");
        require(sonic_motion_composer_native_reset(&expected)==0,"original Runner Reset");
        float q[29],ref[4],head[4],base[8]={0.8f,0,0,0.6f,1,0,0,0};
        require(sonic_motion_composer_native_reference_frame(&expected,0,&assets.mirror,q,ref)==0,"reference at runner callback");
        require(rek_g1_initial_heading(&expected.backends,base,ref,head),"runner InitHeading");
        if(order==REK_G1_RESET_RUNNER_THEN_INPUT)
            require(sonic_motion_composer_native_play_action(&expected,&assets.routes[0].clip,&assets.routes[0].config)==0,"original Input PlayIdle after runner");
        f.flags[1]=0;const auto other=f.composers[1];const auto otherrow=f.rows[1];
        reset_ordered_kernel(&f.buffers,f.flags,base,order,1,Fixture::count);
        require(f.rows[0].status==0,"ordered reset status");
        require(!memcmp(&expected,&f.composers[0],sizeof(expected)),"ordered reset retains exact callback state");
        require(!memcmp(head,f.heading,sizeof(head)),"heading computed at runner callback");
        require(!memcmp(&other,&f.composers[1],sizeof(other)),"unselected composer untouched");
        require(!memcmp(&otherrow,&f.rows[1],sizeof(otherrow)),"unselected scheduler untouched");
        require(f.rows[0].direct_native_active==1,"body reset preserves dispatch mode");
        if(order==REK_G1_RESET_RUNNER_THEN_INPUT){
            require(f.composers[0].from_layer.active,"ordinary idle crossfade retained");
            require(f.composers[0].from_layer.clip.dof_position_mujoco==assets.routes[route].clip.dof_position_mujoco,"prior clip retained");
            require(f.composers[0].from_layer.cursor==0,"prior clip reset cursor retained");
            require(f.composers[0].from_layer.speed==0.625f,"prior speed retained");
        }else require(!f.composers[0].from_layer.active,"runner last clears outgoing active only");
        reset_ordered_kernel(&f.buffers,f.flags,base,order,0,Fixture::count);
        require(f.rows[0].direct_native_active==0,"explicit mode-clear supported");
    }
    Fixture bad(assets);auto previous=bad.composers[0];bad.flags[1]=0;
    reset_ordered_kernel(&bad.buffers,bad.flags,bad.heading,static_cast<RekG1CudaResetOrder>(0),1,Fixture::count);
    require(bad.rows[0].status==400,"invalid order fails closed");
    require(!memcmp(&previous,&bad.composers[0],sizeof(previous)),"invalid order preserves composer");
    for(int i=0;i<33;++i)require(bad.masks[i]==0,"invalid order no action mask");
    Fixture nonfinite(assets);previous=nonfinite.composers[0];float base[8]={NAN,0,0,0,1,0,0,0};
    reset_ordered_kernel(&nonfinite.buffers,nonfinite.flags,base,REK_G1_RESET_RUNNER_THEN_INPUT,1,Fixture::count);
    require(nonfinite.rows[0].status==311,"nonfinite base rejected");
    require(!memcmp(&previous,&nonfinite.composers[0],sizeof(previous)),"failed heading preserves composer");
}
}
int main(){Assets assets;test_ordered(assets);printf("{\"checks\":%d,\"failures\":0,\"gpu_calls\":0}\n",checks);}
'''
(tests/'test_ordered_reset.cpp').write_text(prefix+body,newline='\n')
# Preserve the existing actual scheduler dispatch regression unchanged.
shutil.copyfile(Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile\ocean\rek_g1\test_g1_semantic_direct_cpu.cpp'),tests/'test_g1_semantic_direct_cpu.cpp')
fixture=Path(r'C:\rekagent\work\rek-native-clone-20260927-r1\original-history\fixture\fixtures.json')
shutil.copyfile(fixture,tests/'fixtures.json')
f=json.loads(fixture.read_text());names=list(f['history']['channels'])
with (tests/'history-input.bin').open('wb') as out:
    out.write(struct.pack('<64f',*f['tokens']));out.write(struct.pack('<i',len(f['operations'])))
    for op in f['operations']:
        out.write(struct.pack('<i',{'clear':0,'push':1,'query':2}[op['kind']]))
        if op['kind']=='push':
            snap=f['snapshots'][op['snapshot_id']]
            values=[v for n in names for v in snap[n]]
            assert len(values)==93
            out.write(struct.pack('<93f',*values))
    out.write(struct.pack('<i',len(f['quaternions'])))
    for q in f['quaternions']:out.write(struct.pack('<4f',*q['wxyz']))
print('prepared exact shared history and actual kernel tests')
