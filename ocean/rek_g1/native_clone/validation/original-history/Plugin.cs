using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using BepInEx;
using BepInEx.Unity.IL2CPP;
using Il2CppInterop.Common;
using Il2CppInterop.Runtime;
using Il2CppInterop.Runtime.Runtime;
using Il2CppInterop.Runtime.InteropTypes.Arrays;
using REKApp;
using UnityEngine;
using UObj = UnityEngine.Object;

namespace RekHistoryOracle;

[BepInPlugin("openai.rek.detached.history.oracle", "REK Original History Oracle", "0.1.0")]
[BepInProcess("REK.exe")]
public sealed class Plugin : BasePlugin
{
    const string GameHash = "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412";
    const string MetadataHash = "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd";
    const string InteropHash = "faa94fb58e24fda95e2c06810e28b9eb2d6d9f9f8327541976a0dc1011f646d2";
    internal static Plugin? Active;
    StreamWriter? writer;
    JsonDocument? configDoc, fixtureDoc;
    string output = "", partial = "", runId = "", fixtureRoot = "";
    bool finished, executing;
    int unityThread, rows;
    readonly Stopwatch age = Stopwatch.StartNew();
    readonly Dictionary<string,int> calls = new();
    readonly List<UObj> owned = new();
    readonly List<object> entrypoints = new();
    static readonly string[] Channels={"angularVelocity","jointPositions","jointVelocities","actions","gravityDir"};
    static readonly JsonSerializerOptions Json = new() { WriteIndented = false };
    new JsonElement Config => configDoc!.RootElement;
    JsonElement Fixture => fixtureDoc!.RootElement;
    public static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    static string FileHash(string p) => Hash(File.ReadAllBytes(p));
    static void Require(bool ok,string message) { if(!ok) throw new InvalidOperationException(message); }
    static string Full(string p) => Path.GetFullPath(p).TrimEnd(Path.DirectorySeparatorChar,Path.AltDirectorySeparatorChar);
    static string Str(JsonElement j,string key) => j.GetProperty(key).GetString() ?? throw new InvalidDataException(key);
    void Emit(object value) { writer!.WriteLine(JsonSerializer.Serialize(value,Json)); writer.Flush(); }
    void Count(string method) { calls[method]=calls.GetValueOrDefault(method)+1; }
    void Call(string name,System.Action fn) { Guard(); fn(); Count(name); }
    T Call<T>(string name,Func<T> fn) { Guard();var result=fn();Count(name);return result; }
    void Guard() { Require(Environment.CurrentManagedThreadId==unityThread,"not Unity callback thread");Require(age.Elapsed.TotalSeconds<60,"oracle60second deadline"); }

    public override void Load()
    {
        // Four independent opt-ins prevent accidental activation in an existing installation.
        if(Environment.GetEnvironmentVariable("REK_COMPOSER_ORACLE_ENABLE")!="detached-v1" ||
           !Environment.GetCommandLineArgs().Contains("--rek-composer-oracle")) return;
        try {
            var p=Environment.GetEnvironmentVariable("REK_COMPOSER_ORACLE_CONFIG");Require(!string.IsNullOrWhiteSpace(p),"missing explicit config");
            configDoc=JsonDocument.Parse(File.ReadAllBytes(p!));Require(Str(Config,"schema")=="rek.original_composer.run.v1","config schema");
            Require(Full(Str(Config,"expected_game_root")).Equals(Full(Paths.GameRootPath),StringComparison.OrdinalIgnoreCase),"isolated game-root mismatch");
            runId=Str(Config,"run_id");Require(runId.Length is >=8 and <=100,"run id");
            using var marker=JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Paths.GameRootPath,"REK_COMPOSER_ORACLE_ISOLATED.json")));
            Require(Str(marker.RootElement,"run_id")==runId && Str(marker.RootElement,"config_sha256")==FileHash(p!),"isolated marker/config mismatch");
            output=Full(Str(Config,"output_directory"));Require(!Directory.Exists(output),"output must be fresh");Directory.CreateDirectory(output);
            partial=Path.Combine(output,"oracle.jsonl.partial");writer=new StreamWriter(new FileStream(partial,FileMode.CreateNew,FileAccess.Write,FileShare.Read),new UTF8Encoding(false));
            var game=Path.Combine(Paths.GameRootPath,"GameAssembly.dll");var meta=Path.Combine(Paths.GameRootPath,"REK_Data","il2cpp_data","Metadata","global-metadata.dat");
            Require(FileHash(game)==GameHash && FileHash(meta)==MetadataHash,"game/metadata hash mismatch");
            Require(FileHash(typeof(SonicMotionComposer).Assembly.Location)==InteropHash,"actual loaded interop hash mismatch");
            var fp=Str(Config,"fixture_manifest");Require(FileHash(fp)==Str(Config,"fixture_sha256"),"fixture manifest hash");fixtureRoot=Path.GetDirectoryName(Full(fp))!;fixtureDoc=JsonDocument.Parse(File.ReadAllBytes(fp));
            Require(Str(Fixture,"schema")=="rek.original_history.fixture.v1","fixture schema");
            Require(Str(Config,"mode") is "smoke" or "sequence","unknown mode");
            Emit(new { @event="header",schema="rek.original_history.trace.v1",run_id=runId,mode=Str(Config,"mode"),process_id=Environment.ProcessId,
                game_root=Paths.GameRootPath,game_sha256=GameHash,metadata_sha256=MetadataHash,interop_sha256=InteropHash,
                plugin_sha256=FileHash(Assembly.GetExecutingAssembly().Location),config_sha256=FileHash(p!),fixture_sha256=FileHash(fp),
                claim="Original compiled client history and decoder packing with explicit pretransformed snapshots; no motor, physics or authoritative server claim",utc=DateTime.UtcNow.ToString("O") });
            Active=this;AddComponent<OracleBehaviour>();
        } catch(Exception ex) {
            Log.LogError(ex);
            if(writer!=null) { Finish(false,"preflight_failed",ex); }
            // An unverified isolated marker/root never grants permission to quit any process.
        }
    }


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
    byte[] ReadBound(JsonElement binding)
    {
        string p=Full(Path.Combine(fixtureRoot,Str(binding,"file")));
        Require(p.StartsWith(fixtureRoot+Path.DirectorySeparatorChar,StringComparison.OrdinalIgnoreCase),"fixture path escape");
        byte[] b=File.ReadAllBytes(p);Require(b.Length==binding.GetProperty("bytes").GetInt32()&&Hash(b)==Str(binding,"sha256"),"fixture input hash/size mismatch");return b;
    }

    unsafe void ResolveEntrypoints()
    {
        var names=new[]{"BuildObsPlans","FillHistoryChannel","BuildDecoderObs","CalcHeadingMj","YawQuatMj","QuatMulMj"};
        foreach(var type in new[]{typeof(SonicPolicyRunner),typeof(SonicPolicyRunner.StateRingBuffer)}) {
            RuntimeHelpers.RunClassConstructor(type.TypeHandle);
            foreach(string name in type==typeof(SonicPolicyRunner.StateRingBuffer)?new[]{"Push","GetLatest","Clear"}:names) {
                var method=type.GetMethod(name,BindingFlags.Public|BindingFlags.NonPublic|BindingFlags.Instance|BindingFlags.Static|BindingFlags.DeclaredOnly) ?? throw new MissingMethodException(type.FullName,name);
                var field=Il2CppInteropUtils.GetIl2CppMethodInfoPointerFieldForGeneratedMethod(method) ?? throw new InvalidOperationException("not a generated native wrapper: "+name);
                var info=(IntPtr)field.GetValue(null)!;Require(info!=IntPtr.Zero,"null native MethodInfo");
                IntPtr ptr=(UnityVersionHandler.Wrap((Il2CppMethodInfo*)info) ?? throw new InvalidOperationException("missing method wrapper")).MethodPointer;
                var owner=Process.GetCurrentProcess().Modules.Cast<ProcessModule>().SingleOrDefault(m=>ptr.ToInt64()>=m.BaseAddress.ToInt64()&&ptr.ToInt64()<m.BaseAddress.ToInt64()+m.ModuleMemorySize);
                Require(owner!=null && string.Equals(Path.GetFileName(owner.FileName),"GameAssembly.dll",StringComparison.OrdinalIgnoreCase),"method outside original GameAssembly: "+name);
                byte[] bytes=new byte[32];System.Runtime.InteropServices.Marshal.Copy(ptr,bytes,0,bytes.Length);
                entrypoints.Add(new {type=type.FullName,method=name,wrapper=method.ToString(),method_info=$"0x{info.ToInt64():x}",entrypoint=$"0x{ptr.ToInt64():x}",rva=$"0x{ptr.ToInt64()-owner!.BaseAddress.ToInt64():x}",module=owner.FileName,first32_sha256=Hash(bytes)});
            }
        }
    }

    static object Pack(float v){Require(float.IsFinite(v),"nonfinite original output");return new {value=v,bits=$"0x{BitConverter.SingleToInt32Bits(v):x8}"};}
    static object Pack(float[] v){Require(v.All(float.IsFinite),"nonfinite original array output");return new {values=v,bits=v.Select(x=>$"0x{BitConverter.SingleToInt32Bits(x):x8}").ToArray()};}
    static string HashJagged(Il2CppReferenceArray<Il2CppStructArray<float>> a){using var s=new MemoryStream();foreach(var row in a)foreach(float v in row)s.Write(BitConverter.GetBytes(v));return Hash(s.ToArray());}
    static object ComponentCounts() => new {loaded_robot_components=Resources.FindObjectsOfTypeAll<Robot>().Length,loaded_runner_components=Resources.FindObjectsOfTypeAll<SonicPolicyRunner>().Length,loaded_composer_components=Resources.FindObjectsOfTypeAll<SonicMotionComposer>().Length,scope="loaded components, includes inactive; no singleton getters"};
    void Finish(bool success,string reason,Exception? error)
    {
        if(finished)return;finished=true;
        try { if(writer!=null){Emit(new { @event="oracle_end",success,reason,rows,completed_wrapper_calls=calls,elapsed_seconds=age.Elapsed.TotalSeconds,error=error?.ToString(),no_robot_or_physics_created_by_harness=true,disabled_runner_dependencies_manually_supplied=true,model_loading_and_automatic_runner_lifecycle_not_invoked=true,whole_process_physics_steps_not_instrumented=true,counts=unityThread==0?null:ComponentCounts() });writer.Dispose();writer=null;File.Move(partial,Path.Combine(output,success?"oracle.jsonl":"oracle.failed.jsonl"));} }
        finally { Active=null;Log.LogInfo($"History oracle closed: {reason}, success={success}");Application.Quit(success?0:2); }
    }
}

public sealed class OracleBehaviour : MonoBehaviour
{
    public OracleBehaviour(IntPtr pointer):base(pointer){}
    public void Update()=>Plugin.Active?.OnUpdate();
}
