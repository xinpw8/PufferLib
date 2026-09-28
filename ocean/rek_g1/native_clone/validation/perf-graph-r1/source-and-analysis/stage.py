from pathlib import Path
import hashlib,difflib,json
root=Path(__file__).parent
source=root.parent/'source/ocean/rek_g1/native5/eval_worker.cpp'
before=source.read_bytes();(root/'eval_worker.original.cpp').write_bytes(before)
s=before.decode().replace('\r\n','\n')
def replace(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
replace('#include "runtime_api.h"','#include "runtime_api.h"\n#include "eval_step_graph.h"')
replace('    bool failed=false;','    bool failed=false;\n    bool use_step_graph=false;\n    EvalStepGraph step_graph;')
replace('        arenas=integer(config,"arenas",4);', '''        if(const auto* graph=field(config,"cuda_graph_step")) {
            if(!cJSON_IsBool(graph))throw std::runtime_error("cuda_graph_step must be boolean");
            use_step_graph=cJSON_IsTrue(graph);
        }
        arenas=integer(config,"arenas",4);''')
replace('        cuda_ok(cudaMemsetAsync(direct_enabled,0,arenas*2,stream));','''        // Destroy a prior capture before any reset or later resource replacement.
        step_graph.clear();
        cuda_ok(cudaMemsetAsync(direct_enabled,0,arenas*2,stream));''')
replace('        runtime_ok(rek_native5_check_status(runtime,stream));tick=0;scripted_move=0;','''        runtime_ok(rek_native5_check_status(runtime,stream));tick=0;scripted_move=0;
        if(use_step_graph) {
            // Recording enqueues no simulated step. Inputs were bound once in
            // the constructor; copies and high-level inference stay outside.
            step_graph.capture(stream,[&]{runtime_ok(rek_native5_step(runtime,stream));});
            std::fprintf(stderr,"eval_worker cuda_graph_step=1 captured_runtime_step_only=1\\n");
        }''')
replace('                runtime_ok(rek_native5_step(runtime,stream));tick++;','''                if(use_step_graph)step_graph.launch(stream);
                else runtime_ok(rek_native5_step(runtime,stream));
                tick++;''')
replace('        renderer.reset();for(auto* p:policies)rek_native_policy_destroy(p);','''        if(stream)cudaStreamSynchronize(stream);
        step_graph.clear();
        renderer.reset();for(auto* p:policies)rek_native_policy_destroy(p);''')
(root/'eval_worker.cpp').write_text(s,newline='\n')
assert source.read_bytes()==before
(root/'worker.diff').write_text(''.join(difflib.unified_diff(before.decode().replace('\r\n','\n').splitlines(True),s.splitlines(True),fromfile='original/eval_worker.cpp',tofile='candidate/eval_worker.cpp')),newline='\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(root/'SOURCE.json').write_text(json.dumps({'schema':'rek.native_clone.optional_step_graph.v1','original':{'path':str(source),'sha256':sha(source)},'candidate':{'path':str(root/'eval_worker.cpp'),'sha256':sha(root/'eval_worker.cpp')},'header':{'path':str(root/'eval_step_graph.h'),'sha256':sha(root/'eval_step_graph.h')},'default_enabled':False,'gpu_tested':False},indent=2)+'\n')
print((root/'SOURCE.json').read_text())
