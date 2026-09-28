using System.Diagnostics;
using System.Globalization;
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

namespace RekSlerpOracle;

[BepInPlugin("openai.rek.detached.slerp.oracle","REK Original Slerp Oracle","0.1.0")]
[BepInProcess("REK.exe")]
public sealed class Plugin:BasePlugin
{
    const string GameHash="6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412";
    const string MetaHash="e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd";
    const string PlayerHash="277953a7035b1633c239904853bfbea7b2948937ef5567e70c1911c260dd1414";
    const string RekInteropHash="faa94fb58e24fda95e2c06810e28b9eb2d6d9f9f8327541976a0dc1011f646d2";
    const string UnityInteropHash="3ac45305a21f5107c0a9e813c48503cf27944c15ab9f6495396f20f61840104a";
    internal static Plugin? Active;
    readonly Stopwatch timer=Stopwatch.StartNew();
    JsonDocument? config,fixture;
    StreamWriter? writer;
    string output="",partial="";
    bool verifiedIsolation,finished,executing;
    int unityThread,rows;
    static string Hash(string p)=>Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(p))).ToLowerInvariant();
    static string Full(string p)=>Path.GetFullPath(p).TrimEnd(Path.DirectorySeparatorChar,Path.AltDirectorySeparatorChar);
    static string Str(JsonElement j,string k)=>j.GetProperty(k).GetString()??throw new InvalidDataException(k);
    static void Require(bool b,string why){if(!b)throw new InvalidDataException(why);}
    void Emit(object value){writer!.WriteLine(JsonSerializer.Serialize(value));writer.Flush();}
    void Guard(){Require(Environment.CurrentManagedThreadId==unityThread,"Unity callback thread");Require(timer.Elapsed.TotalSeconds<90,"oracle deadline");}
    public override void Load()
    {
        if(Environment.GetEnvironmentVariable("REK_SLERP_ORACLE_ENABLE")!="detached-v1" ||
           !Environment.GetCommandLineArgs().Contains("--rek-slerp-oracle"))return;
        try{
            var cp=Environment.GetEnvironmentVariable("REK_SLERP_ORACLE_CONFIG");Require(!string.IsNullOrWhiteSpace(cp),"config required");
            config=JsonDocument.Parse(File.ReadAllBytes(cp!));var c=config.RootElement;
            Require(Str(c,"schema")=="rek.original_slerp.run.v1","run schema");
            Require(Full(Str(c,"expected_game_root")).Equals(Full(Paths.GameRootPath),StringComparison.OrdinalIgnoreCase),"private game root");
            using var marker=JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Paths.GameRootPath,"REK_SLERP_ORACLE_ISOLATED.json")));
            Require(Str(marker.RootElement,"run_id")==Str(c,"run_id")&&Str(marker.RootElement,"config_sha256")==Hash(cp!),"private marker/config identity");
            verifiedIsolation=true;
            Require(Hash(Path.Combine(Paths.GameRootPath,"GameAssembly.dll"))==GameHash,"GameAssembly identity");
            Require(Hash(Path.Combine(Paths.GameRootPath,"UnityPlayer.dll"))==PlayerHash,"UnityPlayer identity");
            Require(Hash(Path.Combine(Paths.GameRootPath,"REK_Data/il2cpp_data/Metadata/global-metadata.dat"))==MetaHash,"metadata identity");
            Require(Hash(typeof(SonicMotionComposer).Assembly.Location)==RekInteropHash,"REK interop identity");
            Require(Hash(typeof(Quaternion).Assembly.Location)==UnityInteropHash,"Unity interop identity");
            string fp=Str(c,"fixture_manifest");Require(Hash(fp)==Str(c,"fixture_sha256"),"fixture identity");
            fixture=JsonDocument.Parse(File.ReadAllBytes(fp));var f=fixture.RootElement;
            Require(Str(f,"schema")=="rek.original_slerp.fixture.v1"&&f.GetProperty("repeat_count").GetInt32()==2,"fixture contract");
            Require(f.GetProperty("cases").GetArrayLength() is >0 and <=100000,"bounded case count");
            output=Full(Str(c,"output_directory"));Require(!Directory.Exists(output),"fresh output required");Directory.CreateDirectory(output);
            partial=Path.Combine(output,"oracle.jsonl.partial");writer=new(new FileStream(partial,FileMode.CreateNew,FileAccess.Write,FileShare.Read),new UTF8Encoding(false));
            Emit(new{@event="header",schema="rek.original_slerp.trace.v1",run_id=Str(c,"run_id"),game_root=Paths.GameRootPath,
                game_sha256=GameHash,metadata_sha256=MetaHash,unity_player_sha256=PlayerHash,rek_interop_sha256=RekInteropHash,
                unity_interop_sha256=UnityInteropHash,plugin_sha256=Hash(Assembly.GetExecutingAssembly().Location),config_sha256=Hash(cp!),fixture_sha256=Hash(fp),
                input_order="a_wxyz,b_wxyz,t",input_normalization=false,claim="Pure original compiled client Slerp calls; no physics or server claim"});
            Active=this;AddComponent<OracleBehaviour>();
        }catch(Exception ex){Log.LogError(ex);if(verifiedIsolation)Finish(false,ex);}
    }
    unsafe object Attest(Type type,string name,params Type[] parameterTypes)
    {
        RuntimeHelpers.RunClassConstructor(type.TypeHandle);
        var method=type.GetMethod(name,BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic|BindingFlags.DeclaredOnly,null,parameterTypes,null)??throw new MissingMethodException(type.FullName,name);
        var field=Il2CppInteropUtils.GetIl2CppMethodInfoPointerFieldForGeneratedMethod(method)??throw new InvalidDataException("not original wrapper");
        var info=(IntPtr)field.GetValue(null)!;Require(info!=IntPtr.Zero,"native MethodInfo");
        var pointer=(UnityVersionHandler.Wrap((Il2CppMethodInfo*)info)??throw new InvalidDataException("method wrapper")).MethodPointer;
        var owner=Process.GetCurrentProcess().Modules.Cast<ProcessModule>().SingleOrDefault(m=>pointer.ToInt64()>=m.BaseAddress.ToInt64()&&pointer.ToInt64()<m.BaseAddress.ToInt64()+m.ModuleMemorySize);
        Require(owner!=null&&Full(owner.FileName).Equals(Full(Path.Combine(Paths.GameRootPath,"GameAssembly.dll")),StringComparison.OrdinalIgnoreCase),"original module path");
        byte[] first=new byte[32];System.Runtime.InteropServices.Marshal.Copy(pointer,first,0,32);
        return new{type=type.FullName,method=name,parameter_types=method.GetParameters().Select(p=>p.ParameterType.FullName).ToArray(),module=owner!.FileName,rva=$"0x{pointer.ToInt64()-owner.BaseAddress.ToInt64():x}",
            first32_sha256=Convert.ToHexString(SHA256.HashData(first)).ToLowerInvariant()};
    }
    static string[] Bits(float[] values){Require(values.All(float.IsFinite),"finite output");return values.Select(v=>$"0x{BitConverter.SingleToInt32Bits(v):x8}").ToArray();}
    static float[] Decode(JsonElement array)
    {
        var words=array.EnumerateArray().Select(v=>v.GetString()??"").ToArray();Require(words.Length==9,"nine input words");
        var result=new float[9];
        for(int i=0;i<9;i++){
            Require(words[i].Length==10&&words[i].StartsWith("0x",StringComparison.Ordinal),"canonical input word");
            uint bits=uint.Parse(words[i][2..],NumberStyles.AllowHexSpecifier,CultureInfo.InvariantCulture);
            Require(words[i]==$"0x{bits:x8}","canonical lowercase input");result[i]=BitConverter.Int32BitsToSingle(unchecked((int)bits));
            Require(float.IsFinite(result[i]),"finite input");
        }
        return result;
    }
    internal void OnUpdate()
    {
        if(finished||executing||Time.frameCount<3)return;executing=true;unityThread=Environment.CurrentManagedThreadId;
        try{
            Guard();Emit(new{@event="entrypoints",methods=new[]{Attest(typeof(SonicMotionComposer),"SlerpWxyz",typeof(Il2CppStructArray<float>),typeof(Il2CppStructArray<float>),typeof(float),typeof(Il2CppStructArray<float>)),Attest(typeof(Quaternion),"Slerp",typeof(Quaternion),typeof(Quaternion),typeof(float))}});
            for(int repeat=0;repeat<2;repeat++){
                int index=0;
                foreach(var c in fixture!.RootElement.GetProperty("cases").EnumerateArray()){
                    Guard();Require(c.GetProperty("id").GetInt32()==index,"case order");float[] q=Decode(c.GetProperty("input_bits"));
                    var a=new Il2CppStructArray<float>(q.Take(4).ToArray());var b=new Il2CppStructArray<float>(q.Skip(4).Take(4).ToArray());
                    var result=new Il2CppStructArray<float>(4);SonicMotionComposer.SlerpWxyz(a,b,q[8],result);
                    Quaternion unity=Quaternion.Slerp(new Quaternion(q[1],q[2],q[3],q[0]),new Quaternion(q[5],q[6],q[7],q[4]),q[8]);
                    Emit(new{@event="slerp",repeat,case_id=index,input_bits=Bits(q),rek_wxyz_bits=Bits(result.ToArray()),unity_wxyz_bits=Bits(new[]{unity.w,unity.x,unity.y,unity.z})});rows++;index++;
                }
            }
            Finish(true,null);
        }catch(Exception ex){Finish(false,ex);}
    }
    void Finish(bool success,Exception? error)
    {
        if(finished)return;finished=true;
        try{if(writer!=null){Emit(new{@event="oracle_end",success,rows,elapsed_seconds=timer.Elapsed.TotalSeconds,error=error?.ToString(),
            composer_or_robot_instances_created=false,whole_process_physics_not_instrumented=true});writer.Dispose();writer=null;File.Move(partial,Path.Combine(output,success?"oracle.jsonl":"oracle.failed.jsonl"));}}
        finally{Active=null;if(verifiedIsolation)Application.Quit(success?0:2);}
    }
}
public sealed class OracleBehaviour:MonoBehaviour
{
    public OracleBehaviour(IntPtr pointer):base(pointer){}
    public void Update()=>Plugin.Active?.OnUpdate();
}
