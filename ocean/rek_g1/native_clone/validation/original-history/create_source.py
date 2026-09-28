from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parent
OLD=Path(r'C:\rekagent\work\rek-original-functions-20260927-r1\oracle')
def write(name, text):
    with (ROOT/name).open('x',encoding='utf8',newline='\n') as f:f.write(text)
p=(OLD/'Plugin.cs').read_text()
prefix=p[:p.index('    internal void OnUpdate()')]
prefix=prefix.replace('RekComposerOracle','RekHistoryOracle').replace('openai.rek.detached.composer.oracle','openai.rek.detached.history.oracle').replace('REK Original Composer Oracle','REK Original History Oracle')
prefix=prefix.replace('static readonly int[] Offsets = { 0, 1, 5, 10, 15 };','static readonly string[] Channels={"angularVelocity","jointPositions","jointVelocities","actions","gravityDir"};')
prefix=prefix.replace('rek.original_composer.fixture.v1','rek.original_history.fixture.v1').replace('rek.original_composer.trace.v1','rek.original_history.trace.v1').replace('Original compiled client composer with explicit supplied fixture, no authoritative server or physics claim','Original compiled client history and decoder packing with explicit pretransformed snapshots; no motor, physics or authoritative server claim')
bound=p[p.index('    byte[] ReadBound'):p.index('    (MocapClipConfig config')]
bound=bound.replace('var names=new[]{"Init","Reset","PlayAction","PlayActionImmediate","GetReferenceFrame","TryGetReferenceVelocity","TryGetReferenceRootPos","Advance","ConsumeHeadingDelta","get_HeadingClipOwnership","SetLocomotionSpeed","RegisterFootFeatures","BuildClip","LoadClip"};','var names=new[]{"BuildObsPlans","FillHistoryChannel","BuildDecoderObs","CalcHeadingMj","YawQuatMj","QuatMulMj"};')
bound=bound.replace('typeof(SonicMotionComposer),typeof(NpzReader)','typeof(SonicPolicyRunner),typeof(SonicPolicyRunner.StateRingBuffer)').replace('type==typeof(NpzReader)?new[]{"Read"}:names','type==typeof(SonicPolicyRunner.StateRingBuffer)?new[]{"Push","GetLatest","Clear"}:names')
tail=p[p.index('    static object Pack(float v)'):]
tail=tail.replace('no_robot_runner_or_physics_created_by_harness=true','no_robot_or_physics_created_by_harness=true,disabled_runner_dependencies_manually_supplied=true,model_loading_and_automatic_runner_lifecycle_not_invoked=true').replace('Composer oracle closed','History oracle closed')
body=r'''
    internal void OnUpdate()
    {
        if(finished||executing)return;
        unityThread=Environment.CurrentManagedThreadId;
        if(age.Elapsed.TotalSeconds>55){Finish(false,"startup_deadline",null);return;}
        if(Time.frameCount<3)return;
        executing=true;
        try {
            Emit(new {@event="scene_component_inventory",phase="before_harness_trials",counts=ComponentCounts()});
            ResolveEntrypoints();Emit(new {@event="entrypoints",unity_thread=unityThread,frame=Time.frameCount,methods=entrypoints});
            string text=new UTF8Encoding(false,true).GetString(ReadBound(Fixture.GetProperty("joint_config")));
            for(int repeat=0;repeat<2;repeat++)RunHistory(repeat,text);
            RunQuaternionFunctions();
            Require(rows==28,"expected28queryrows");Finish(true,"history_sequence_complete",null);
        } catch(Exception ex){Finish(false,"execution_failed",ex);}
    }

    static float[] Floats(JsonElement e)=>e.EnumerateArray().Select(x=>x.GetSingle()).ToArray();
    static object? Maybe(Il2CppStructArray<float>? a)=>a==null?null:Pack(a.ToArray());
    static object Snapshot(SonicPolicyRunner.StateSnapshot s)=>new {
        angularVelocity=Maybe(s.angularVelocity),jointPositions=Maybe(s.jointPositions),
        jointVelocities=Maybe(s.jointVelocities),actions=Maybe(s.actions),gravityDir=Maybe(s.gravityDir)};
    static SonicPolicyRunner.StateSnapshot NewSnapshot(JsonElement e)=>new() {
        angularVelocity=new(Floats(e.GetProperty("angularVelocity"))),jointPositions=new(Floats(e.GetProperty("jointPositions"))),
        jointVelocities=new(Floats(e.GetProperty("jointVelocities"))),actions=new(Floats(e.GetProperty("actions"))),gravityDir=new(Floats(e.GetProperty("gravityDir")))};

    void RunHistory(int repeat,string text)
    {
        var go=new GameObject("REK_HISTORY_DETACHED_"+repeat);owned.Add(go);go.SetActive(false);
        var r=go.AddComponent<SonicPolicyRunner>();r.enabled=false;
        // This original Unity parser is the same parser used by InitializeInternal.
        // Do not call InitializeInternal: it creates model/Robot/DDS dependencies.
        r.config=Call("JsonUtility.FromJson<SonicConfig>",()=>JsonUtility.FromJson<SonicPolicyRunner.SonicConfig>(text));
        Require(r.config!=null&&r.config.joints.Length==29,"original parsed config shape");
        r.numJoints=29;r.maxHistoryFrames=10;
        r.stateRing=new SonicPolicyRunner.StateRingBuffer(21);Count("StateRingBuffer.ctor");
        r.historyReadBuffer=new Il2CppReferenceArray<SonicPolicyRunner.StateSnapshot>(10);
        r.tokenBuffer=new Il2CppStructArray<float>(Floats(Fixture.GetProperty("tokens")));
        Call("BuildObsPlans",()=>r.BuildObsPlans());
        Require(r.encoderObsSize==1762&&r.decoderObsSize==994&&r.decoderPlan.Length==6,"original plan shape");
        Emit(new {@event="lifecycle",repeat,active=go.activeSelf,enabled=r.enabled,is_ready=r.IsReady,
            full_Init_and_Start_not_called=true,automatic_runner_callbacks_not_invoked=true,
            joint_order="explicit fixture already transformed; no map or normalization applied by harness",
            config_sha256=Str(Fixture.GetProperty("joint_config"),"sha256"),r.numJoints,r.maxHistoryFrames,
            capacity=r.stateRing.Capacity,encoder_size=r.encoderObsSize,decoder_size=r.decoderObsSize,
            decoder_plan=r.decoderPlan.Select(x=>new{x.name,x.offset,x.dim,fill_present=x.fill!=null}).ToArray(),
            tokens=Pack(r.tokenBuffer.ToArray())});
        int operation=0;
        foreach(var op in Fixture.GetProperty("operations").EnumerateArray()) {
            Require(!go.activeSelf&&!r.enabled,"detached runner activated");
            string kind=Str(op,"kind");
            if(kind=="push") {
                int id=op.GetProperty("snapshot_id").GetInt32();var snap=NewSnapshot(Fixture.GetProperty("snapshots")[id]);
                Call("Push",()=>r.stateRing.Push(snap));
                Emit(new{@event="push",repeat,operation,snapshot_id=id,count=r.stateRing.count,write_index=r.stateRing.writeIdx,input=Snapshot(snap)});
            } else if(kind=="clear") {
                Call("Clear",()=>r.stateRing.Clear());
                Emit(new{@event="clear",repeat,operation,count=r.stateRing.count,write_index=r.stateRing.writeIdx});
            } else if(kind=="query") {
                var result1=new Il2CppReferenceArray<SonicPolicyRunner.StateSnapshot>(10);
                var result2=new Il2CppReferenceArray<SonicPolicyRunner.StateSnapshot>(5);
                Call("GetLatest",()=>r.stateRing.GetLatest(10,1,result1));
                Call("GetLatest",()=>r.stateRing.GetLatest(5,2,result2));
                var dest=new Il2CppStructArray<float>(Enumerable.Repeat(-777.25f,994).ToArray());
                Call("BuildDecoderObs",()=>r.BuildDecoderObs(dest));
                // Direct original channel fill starts with sentinels, so skipped null padding
                // can be distinguished from BuildDecoderObs's preceding Array.Clear.
                var fills=new List<object>();
                foreach(var obs in r.config.decoder.observations.Where(x=>x.type=="history")) {
                    int n=obs.dim_per_frame*obs.num_frames;var d=new Il2CppStructArray<float>(Enumerable.Repeat(-333.5f,n+4).ToArray());
                    Call("FillHistoryChannel",()=>r.FillHistoryChannel(obs,d,2));
                    fills.Add(new{obs.name,offset=2,sentinel=Pack(-333.5f),values=Pack(d.ToArray())});
                }
                Emit(new{@event="history_query",repeat,operation,label=Str(op,"label"),count=r.stateRing.count,write_index=r.stateRing.writeIdx,
                    latest_step1=result1.Select(Snapshot).ToArray(),latest_step2=result2.Select(Snapshot).ToArray(),
                    decoder=Pack(dest.ToArray()),direct_channel_fills=fills});rows++;
            } else throw new InvalidDataException("unknown operation "+kind);
            operation++;
        }
        Emit(new{@event="trial_end",repeat,active=go.activeSelf,enabled=r.enabled,is_ready=r.IsReady});
        UObj.Destroy(go);
    }

    void RunQuaternionFunctions()
    {
        foreach(var f in Fixture.GetProperty("quaternions").EnumerateArray()) {
            var q=new Il2CppStructArray<float>(Floats(f.GetProperty("wxyz")));
            float h=Call("CalcHeadingMj",()=>SonicPolicyRunner.CalcHeadingMj(q));
            var y=new Il2CppStructArray<float>(4);Call("YawQuatMj",()=>SonicPolicyRunner.YawQuatMj(h,y));
            var product=new Il2CppStructArray<float>(4);Call("QuatMulMj",()=>SonicPolicyRunner.QuatMulMj(y,q,product));
            Emit(new{@event="quaternion_function",label=Str(f,"label"),input_wxyz=Pack(q.ToArray()),heading=Pack(h),yaw_wxyz=Pack(y.ToArray()),yaw_times_input_wxyz=Pack(product.ToArray())});
        }
    }
'''
write('Plugin.cs',prefix+body+bound+tail)
proj=(OLD/'RekComposerOracle.csproj').read_text().replace('RekComposerOracle','RekHistoryOracle')
proj=proj.replace('    <Reference Include="REKApp">','    <Reference Include="UnityEngine.JSONSerializeModule"><HintPath>$(RekInteropDir)\\UnityEngine.JSONSerializeModule.dll</HintPath><Private>false</Private></Reference>\n    <Reference Include="REKApp">')
write('RekHistoryOracle.csproj',proj)
fixture=ROOT/'fixture';fixture.mkdir()
cfg=(OLD/'fixture/sonic_config.json').read_bytes();(fixture/'sonic_config.json').write_bytes(cfg)
channels={'angularVelocity':3,'jointPositions':29,'jointVelocities':29,'actions':29,'gravityDir':3}
snapshots=[]
for frame in range(31):
    snapshots.append({name:[((-1)**(frame+c+j))*(c*64+frame+1+j/64) for j in range(width)] for c,(name,width) in enumerate(channels.items())})
ops=[{'kind':'query','label':'empty'}]
for i in range(27):
    ops.append({'kind':'push','snapshot_id':i})
    if i+1 in [1,2,3,9,10,11,20,21,22,27]:ops.append({'kind':'query','label':f'push_{i+1}'})
ops += [{'kind':'clear'},{'kind':'query','label':'cleared'}]
for i in range(27,31):
    ops.append({'kind':'push','snapshot_id':i})
    if i in [27,30]:ops.append({'kind':'query','label':f'after_clear_{i-26}'})
assert sum(x['kind']=='query' for x in ops)==14
f={'schema':'rek.original_history.fixture.v1','joint_config':{'file':'sonic_config.json','bytes':len(cfg),'sha256':hashlib.sha256(cfg).hexdigest()},
   'history':{'capacity':21,'channels':channels,'decoder_frames':10,'decoder_step':1,'diagnostic_frames':5,'diagnostic_step':2},
   'tokens':[(-1)**i*(i/8+0.0625) for i in range(64)],'snapshots':snapshots,'operations':ops,
   'quaternions':[{'label':'identity','wxyz':[1,0,0,0]},{'label':'positive_yaw','wxyz':[0.8,0,0,0.6]},{'label':'negative_yaw','wxyz':[0.8,0,0,-0.6]},{'label':'tilted_asymmetric','wxyz':[0.5,-0.5,0.5,0.5]}],
   'scope':'Explicit pretransformed asymmetric channels. No physical state sampling, joint permutation, model inference, physics or authoritative server claim.'}
(fixture/'fixtures.json').write_text(json.dumps(f,indent=2)+'\n')
helper=(OLD/'prepare_inputs.py').read_text().replace('rek.original_composer.fixture.v1','rek.original_history.fixture.v1').replace("    for clip in data['clips']: bindings.extend([clip['npz'], clip['foot_features']])\n",'')
# The reviewed runtime preparation creates an empty input directory. Accept only verified empty.
helper=helper.replace("    assert not target.exists(), 'input directory must be fresh'","    assert not target.exists() or (target.is_dir() and not any(target.iterdir())), 'input directory must be absent or verified empty'").replace('    target.mkdir()','    target.mkdir(exist_ok=True)')
write('prepare_inputs.py',helper)
