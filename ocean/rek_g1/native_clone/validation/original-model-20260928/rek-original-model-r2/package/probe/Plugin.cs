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
using Mujoco;
using REKApp;
using UnityEngine;
using UObj = UnityEngine.Object;

namespace RekOriginalModelInventory;

[BepInPlugin("openai.rek.original.model.inventory", "REK Original Model Asset Inventory", "0.1.0")]
[BepInProcess("REK.exe")]
public sealed class Plugin : BasePlugin
{
    const string GameHash="6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412";
    const string MetadataHash="e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd";
    const string InteropHash="faa94fb58e24fda95e2c06810e28b9eb2d6d9f9f8327541976a0dc1011f646d2";
    internal static Plugin? Active;
    StreamWriter? writer;
    JsonDocument? configDoc,fixtureDoc;
    string output="",partial="",runId="";
    bool finished,executing;
    int unityThread,components,transforms,serializedBytes;
    readonly Stopwatch age=Stopwatch.StartNew();
    readonly Dictionary<string,int> calls=new();
    static readonly HashSet<string> AllowedRekTypes=new(){"REKApp.Robot","REKApp.SonicPolicyRunner","REKApp.SonicMotionComposer","REKApp.RobotConfig","REKApp.RobotInputController","REKApp.RobotInputControllerCommand","REKApp.RobotInputControllerVR"};
    new JsonElement Config=>configDoc!.RootElement;
    JsonElement Fixture=>fixtureDoc!.RootElement;
    static string Hash(byte[] b)=>Convert.ToHexString(SHA256.HashData(b)).ToLowerInvariant();
    static string FileHash(string p){using var f=File.OpenRead(p);using var h=SHA256.Create();return Convert.ToHexString(h.ComputeHash(f)).ToLowerInvariant();}
    static string Full(string p)=>Path.GetFullPath(p).TrimEnd(Path.DirectorySeparatorChar,Path.AltDirectorySeparatorChar);
    static string Str(JsonElement j,string key)=>j.GetProperty(key).GetString()??throw new InvalidDataException(key);
    static void Require([System.Diagnostics.CodeAnalysis.DoesNotReturnIf(false)] bool ok,string message){if(!ok)throw new InvalidOperationException(message);}
    void Guard(){Require(Environment.CurrentManagedThreadId==unityThread,"not Unity callback thread");Require(age.Elapsed.TotalSeconds<60,"inventory deadline");}
    void Count(string name)=>calls[name]=calls.GetValueOrDefault(name)+1;
    void Emit(object value){writer!.WriteLine(JsonSerializer.Serialize(value));writer.Flush();}

    public override void Load()
    {
        // Retain the four independent opt-ins used by the reviewed isolated image.
        if(Environment.GetEnvironmentVariable("REK_COMPOSER_ORACLE_ENABLE")!="detached-v1" || !Environment.GetCommandLineArgs().Contains("--rek-composer-oracle"))return;
        try {
            var path=Environment.GetEnvironmentVariable("REK_COMPOSER_ORACLE_CONFIG");Require(!string.IsNullOrWhiteSpace(path),"explicit config missing");
            configDoc=JsonDocument.Parse(File.ReadAllBytes(path!));Require(Str(Config,"schema")=="rek.original_model_inventory.run.v1","run schema");
            Require(Full(Str(Config,"expected_game_root")).Equals(Full(Paths.GameRootPath),StringComparison.OrdinalIgnoreCase),"isolated game root");
            runId=Str(Config,"run_id");Require(runId.Length is >=8 and <=100,"run id");
            using var marker=JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Paths.GameRootPath,"REK_COMPOSER_ORACLE_ISOLATED.json")));
            Require(Str(marker.RootElement,"run_id")==runId && Str(marker.RootElement,"config_sha256")==FileHash(path!),"marker/config mismatch");
            Require(FileHash(Path.Combine(Paths.GameRootPath,"GameAssembly.dll"))==GameHash,"GameAssembly mismatch");
            Require(FileHash(Path.Combine(Paths.GameRootPath,"REK_Data","il2cpp_data","Metadata","global-metadata.dat"))==MetadataHash,"metadata mismatch");
            Require(FileHash(typeof(RobotCatalog).Assembly.Location)==InteropHash,"actual REK interop mismatch");
            string fp=Str(Config,"fixture_manifest");Require(FileHash(fp)==Str(Config,"fixture_sha256"),"fixture hash");
            fixtureDoc=JsonDocument.Parse(File.ReadAllBytes(fp));Require(Str(Fixture,"schema")=="rek.original_model_inventory.fixture.v1","fixture schema");
            Require(Str(Fixture,"catalog_resource")=="Workshop/RobotCatalog" && Str(Fixture,"robot_id")=="g1","fixed selection");
            foreach(var f in Fixture.GetProperty("game_files").EnumerateArray()){
                string rel=Str(f,"file");Require(!Path.IsPathRooted(rel)&&!rel.Split('/','\\').Contains(".."),"bound path");
                string full=Full(Path.Combine(Paths.GameRootPath,rel));Require(full.StartsWith(Full(Paths.GameRootPath)+Path.DirectorySeparatorChar,StringComparison.OrdinalIgnoreCase),"path escape");
                Require(new FileInfo(full).Length==f.GetProperty("bytes").GetInt64()&&FileHash(full)==Str(f,"sha256"),"bound asset mismatch: "+rel);
            }
            Require(FileHash(typeof(MjScene).Assembly.Location)==Str(Fixture,"mujoco_interop_sha256"),"actual MuJoCo interop mismatch");
            Require(FileHash(typeof(GameObject).Assembly.Location)==Str(Fixture,"unity_core_interop_sha256"),"actual Unity core interop mismatch");
            Require(FileHash(typeof(JsonUtility).Assembly.Location)==Str(Fixture,"unity_json_interop_sha256"),"actual Unity JSON interop mismatch");
            output=Full(Str(Config,"output_directory"));Require(!Directory.Exists(output),"fresh output required");Directory.CreateDirectory(output);
            partial=Path.Combine(output,"inventory.jsonl.partial");writer=new StreamWriter(new FileStream(partial,FileMode.CreateNew,FileAccess.Write,FileShare.Read),new UTF8Encoding(false));
            Emit(new{@event="header",schema="rek.original_model_inventory.trace.v1",run_id=runId,utc=DateTime.UtcNow.ToString("O"),game_sha256=GameHash,metadata_sha256=MetadataHash,interop_sha256=InteropHash,
                plugin_sha256=FileHash(Assembly.GetExecutingAssembly().Location),fixture_sha256=FileHash(fp),config_sha256=FileHash(path!),claims=new[]{"Original catalog and serialized prefab inspection only","No model generation, robot spawn, dynamics or authoritative-server claim"}});
            Active=this;AddComponent<InventoryBehaviour>();
        }catch(Exception e){Log.LogError(e);if(writer!=null)Finish(false,"preflight_failed",e);}
    }

    internal void OnUpdate()
    {
        if(finished||executing)return;unityThread=Environment.CurrentManagedThreadId;
        if(age.Elapsed.TotalSeconds>55){Finish(false,"startup_deadline",null);return;}
        if(Time.frameCount<3)return;executing=true;
        try {
            Guard();Emit(new{@event="loaded_counts",phase="before_load",counts=Counts()});
            Emit(new{@event="entrypoint",method=Attest(typeof(RobotCatalog),"TryGetById"),invoked_by_harness=false});
            var catalog=Resources.Load<RobotCatalog>("Workshop/RobotCatalog");Count("Resources.Load<RobotCatalog>");Require(catalog!=null,"catalog missing");
            var entries=catalog.robots;Require(entries!=null && entries.Count is >0 and <=64,"catalog count bound");
            Emit(new{@event="catalog",name=catalog.name,instance_id=catalog.GetInstanceID(),instance_id_scope="process-local, not serialized pathID",serialized=Serialize(catalog)});
            RobotCatalogEntry? selectedEntry=null;int selectedMatches=0;
            for(int i=0;i<entries.Count;i++){
                var e=entries[i];var p=e.previewPrefab;
                if(e.id==Str(Fixture,"robot_id")){selectedEntry=e;selectedMatches++;}
                Emit(new{@event="catalog_entry",index=i,id=e.id,display_name=e.displayName,type_label=e.typeLabel,prefab=p==null?null:new{name=p.name,instance_id=p.GetInstanceID(),scene_valid=p.scene.IsValid(),active_self=p.activeSelf,active_in_hierarchy=p.activeInHierarchy}});
            }
            Require(selectedMatches==1 && selectedEntry!=null,"catalog must contain exactly one requested id");
            Require(selectedEntry.previewPrefab!=null,"g1 prefab missing");
            Count("catalog_enumeration_exact_id_selection");
            Emit(new{@event="selection",method="direct_observed_catalog_enumeration",robot_id=Str(Fixture,"robot_id"),exact_matches=selectedMatches,original_TryGetById_invoked=false,original_TryGetById_semantics_validated=false});
            var prefab=selectedEntry.previewPrefab;Require(prefab.name==Str(Fixture,"expected_prefab_name"),"unexpected prefab name");
            Require(!prefab.scene.IsValid(),"catalog prefab is a scene instance");
            Walk(prefab.transform,"",0);
            int configCount=0;
            foreach(var c in Resources.FindObjectsOfTypeAll<RobotConfig>()){
                Guard();Require(++configCount<=64,"robot config count bound");
                Emit(new{@event="loaded_robot_config",instance_id=c.GetInstanceID(),name=c.name,serialized=Serialize(c)});
            }
            int clipCount=0;
            foreach(var c in Resources.FindObjectsOfTypeAll<MocapClipConfig>()){
                Guard();Require(++clipCount<=256,"clip config count bound");
                Emit(new{@event="loaded_clip_config",instance_id=c.GetInstanceID(),name=c.name,serialized=Serialize(c)});
            }
            int settingsCount=0;
            foreach(var s in Resources.FindObjectsOfTypeAll<MjGlobalSettings>()){
                Guard();Require(++settingsCount<=16,"settings count bound");
                Emit(new{@event="loaded_global_settings",instance_id=s.GetInstanceID(),name=s.name,scene_valid=s.gameObject.scene.IsValid(),scene_name=s.gameObject.scene.name,active_self=s.gameObject.activeSelf,serialized=Serialize(s)});
            }
            Emit(new{@event="settings_scope",loaded_count=settingsCount,missing_level1_settings_is_not_default_proof=true,no_scene_loaded_by_harness=true});
            Emit(new{@event="loaded_counts",phase="after_load",counts=Counts()});
            Finish(true,"inventory_complete",null);
        }catch(Exception e){Finish(false,"inspection_failed",e);}
    }

    void Walk(Transform t,string parent,int depth)
    {
        Guard();Require(depth<=64&&++transforms<=2048,"hierarchy bound");var go=t.gameObject;
        string path=parent+"/"+go.name;
        var pos=t.localPosition;var rot=t.localRotation;var scale=t.localScale;
        Emit(new{@event="transform",path,instance_id=t.GetInstanceID(),gameobject_instance_id=go.GetInstanceID(),active_self=go.activeSelf,active_in_hierarchy=go.activeInHierarchy,local_position=new[]{pos.x,pos.y,pos.z},local_rotation_xyzw=new[]{rot.x,rot.y,rot.z,rot.w},local_scale=new[]{scale.x,scale.y,scale.z}});
        foreach(var c in go.GetComponents<Component>()){
            Guard();Require(++components<=8192,"component bound");
            if(c==null){Emit(new{@event="missing_component",path});continue;}
            string type=c.GetIl2CppType().FullName;
            bool selected=type.StartsWith("Mujoco.",StringComparison.Ordinal)||AllowedRekTypes.Contains(type);
            Emit(new{@event="component",path,type,instance_id=c.GetInstanceID(),serialized_scope=selected?"Unity JsonUtility serialized fields":"identity only",serialized=selected?Serialize(c):null});
        }
        for(int i=0;i<t.childCount;i++)Walk(t.GetChild(i),path,depth+1);
    }

    object Serialize(Il2CppSystem.Object value)
    {
        Guard();string text=JsonUtility.ToJson(value,false);Count("JsonUtility.ToJson");
        byte[] b=Encoding.UTF8.GetBytes(text);Require(b.Length<=1024*1024,"serialized object byte bound");serializedBytes+=b.Length;Require(serializedBytes<=16*1024*1024,"serialized total bound");
        using var j=JsonDocument.Parse(text);
        return new{utf8_bytes=b.Length,sha256=Hash(b),value=j.RootElement.Clone(),pointer_scope="instanceID references require correlation; no serialized file/pathID claim"};
    }

    unsafe object Attest(Type type,string name)
    {
        RuntimeHelpers.RunClassConstructor(type.TypeHandle);
        var m=type.GetMethod(name,BindingFlags.Public|BindingFlags.NonPublic|BindingFlags.Instance|BindingFlags.DeclaredOnly,null,new[]{typeof(string),typeof(RobotCatalogEntry).MakeByRefType()},null)??throw new MissingMethodException(type.FullName,name);
        Require(m.ReturnType==typeof(bool),"unexpected TryGetById return type");
        var f=Il2CppInteropUtils.GetIl2CppMethodInfoPointerFieldForGeneratedMethod(m)??throw new InvalidOperationException("native wrapper missing");
        IntPtr info=(IntPtr)f.GetValue(null)!;Require(info!=IntPtr.Zero,"method info null");
        IntPtr ptr=(UnityVersionHandler.Wrap((Il2CppMethodInfo*)info)??throw new InvalidOperationException("method wrapper missing")).MethodPointer;
        var owner=Process.GetCurrentProcess().Modules.Cast<ProcessModule>().SingleOrDefault(x=>ptr.ToInt64()>=x.BaseAddress.ToInt64()&&ptr.ToInt64()<x.BaseAddress.ToInt64()+x.ModuleMemorySize);
        Require(owner!=null&&!string.IsNullOrWhiteSpace(owner.FileName)&&Full(owner.FileName).Equals(Full(Path.Combine(Paths.GameRootPath,"GameAssembly.dll")),StringComparison.OrdinalIgnoreCase),"entrypoint outside pinned isolated GameAssembly");
        byte[] b=new byte[32];System.Runtime.InteropServices.Marshal.Copy(ptr,b,0,b.Length);
        return new{type=type.FullName,method=name,wrapper=m.ToString(),module=owner!.FileName,rva=$"0x{ptr.ToInt64()-owner.BaseAddress.ToInt64():x}",first32_sha256=Hash(b)};
    }
    static object CountType<T>() where T:Component
    {
        var all=Resources.FindObjectsOfTypeAll<T>();int scene=0,active=0;
        foreach(var c in all)if(c.gameObject.scene.IsValid()){scene++;if(c.gameObject.activeInHierarchy)active++;}
        return new{loaded=all.Length,scene_instances=scene,active_scene_instances=active,asset_or_non_scene=all.Length-scene};
    }
    static object Counts()=>new{robots=CountType<Robot>(),runners=CountType<SonicPolicyRunner>(),composers=CountType<SonicMotionComposer>(),mjscenes=CountType<MjScene>(),scope="asset counts may rise on Resources.Load; no singleton getters"};
    void Finish(bool success,string reason,Exception? error)
    {
        if(finished)return;finished=true;
        try{if(writer!=null){Emit(new{@event="inventory_end",success,reason,components,transforms,serialized_bytes=serializedBytes,completed_wrapper_calls=calls,error=error?.ToString(),elapsed_seconds=age.Elapsed.TotalSeconds,no_instantiation_model_generation_dynamics_or_scene_load_by_harness=true,whole_process_physics_steps_not_instrumented=true});writer.Dispose();writer=null;File.Move(partial,Path.Combine(output,success?"inventory.jsonl":"inventory.failed.jsonl"));}}
        finally{Active=null;Log.LogInfo($"Inventory closed: {reason}, success={success}");Application.Quit(success?0:2);}
    }
}
public sealed class InventoryBehaviour:MonoBehaviour
{
    public InventoryBehaviour(IntPtr pointer):base(pointer){}
    public void Update()=>Plugin.Active?.OnUpdate();
}
