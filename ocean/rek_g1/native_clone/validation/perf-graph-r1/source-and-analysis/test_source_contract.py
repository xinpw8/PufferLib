from pathlib import Path
import hashlib,json
root=Path(__file__).parent
s=(root/'eval_worker.cpp').read_text();original=(root/'eval_worker.original.cpp').read_text()
checks=0
def check(value):
    global checks
    assert value
    checks+=1
check('bool use_step_graph=false;' in s)
check('if(!cJSON_IsBool(graph))' in s)
check(s.count('step_graph.capture(stream,[&]{runtime_ok(rek_native5_step(runtime,stream));});')==1)
check(s.count('step_graph.launch(stream)')==1)
check('if(use_step_graph)step_graph.launch(stream);\n                else runtime_ok(rek_native5_step(runtime,stream));\n                tick++;' in s)
for begin,end in [('for(int a=0;a<arenas;a++){','                runtime_ok(rek_native5_step(runtime,stream));tick++;'),('                try{\n                    runtime_ok(rek_native5_check_status','        }else if(op=="reset")reset();')]:
    region=original.split(begin,1)[1].split(end,1)[0]
    check(region in s)
check(s.index('rek_native5_bind_direct_commands')<s.index('        reset();'))
check(s.index('step_graph.clear();',s.index('~Worker'))<s.index('rek_native5_close(runtime)',s.index('~Worker')))
check(s.count('rek_native5_step(runtime,stream)')==2)
print(json.dumps({'checks':checks,'failures':0,'scope':'static bounded worker integration contracts; real graph equivalence still required'}))
