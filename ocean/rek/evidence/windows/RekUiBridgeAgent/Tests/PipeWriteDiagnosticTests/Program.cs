using System.Runtime.CompilerServices;
using System.Text.Json;
using RekUiBridgeAgent;

const string forbidden = "fixture-private-payload-should-never-be-logged";
var checks = 0;
void Check(bool ok, string label) { checks++; if (!ok) throw new Exception(label); }
Exception? captured = null;
try { JsonSerializer.Serialize(new FailingPayload()); }
catch (Exception exception) { captured = exception; }
Check(captured is IndexOutOfRangeException, "real serializer getter exception preserved");
captured!.Data["private"] = forbidden;
var diagnostic = PipeWriteFailureDiagnostic.Describe(captured, "serialize");
Check(diagnostic.Contains("stage=serialize;type=System.IndexOutOfRangeException"), "stage and type");
Check(diagnostic.Contains("FailingPayload.get_Value@IL_"), "actual throwing method");
Check(diagnostic.Contains("System.Text.Json"), "serializer identified");
Check(!diagnostic.Contains(forbidden), "message and Data excluded");
Check(!diagnostic.Contains(".cs:") && !diagnostic.Contains("Program.cs"), "source path and lines excluded");
Check(diagnostic.Split('|').Length <= 12, "frame count bounded");
var unthrown = PipeWriteFailureDiagnostic.Describe(new NullReferenceException(forbidden), "pipe_write");
Check(unthrown.Contains("stage=pipe_write;type=System.NullReferenceException;methods="), "missing stack handled");
Check(!unthrown.Contains(forbidden), "unthrown message excluded");
Console.WriteLine($"PASS pipe_write_diagnostic_checks={checks} actual_serializer_exception=true payload_values_logged=false");

sealed class FailingPayload
{
    public int Value { [MethodImpl(MethodImplOptions.NoInlining)] get => throw new IndexOutOfRangeException("fixture-private-payload-should-never-be-logged"); }
}
