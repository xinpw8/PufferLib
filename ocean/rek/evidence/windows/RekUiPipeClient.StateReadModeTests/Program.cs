using System.Text.Json;

const string Legacy = "5754ea9f818ba27b24c6f8be25978ab59a772ab23d4f260e6f257aaff627d67c";
const string Current = "6c88b9e3718f5549c146865a4a336515385e5b019fd8321a6f5e1fec4253a5e4";
var tests = 0;
void Check(bool condition, string name)
{
    tests++;
    if (!condition) throw new Exception(name);
}
void Reject(params string[] arguments)
{
    Check(!StateReadModeOptions.TryExtract(arguments, out _, out _, out var error)
        && error is not null, "invalid option rejected before connection");
}
foreach (var legacyArgs in new[] { Array.Empty<string>(), new[] { "state" },
    new[] { "state", "result.jsonl", "15" }, new[] { "policy-relay", Current },
    new[] { "schedule", "result.jsonl" } })
{
    Check(StateReadModeOptions.TryExtract(legacyArgs, out var remaining, out var hash, out var error)
        && remaining.SequenceEqual(legacyArgs) && hash is null && error is null, "legacy argv unchanged");
}
foreach (var argv in new[] { new[] { "state", "--bridge-sha256=" + Current },
    new[] { "state", "--bridge-sha256=" + Current, "result.jsonl", "15" },
    new[] { "state", "result.jsonl", "15", "--bridge-sha256=" + Current } })
{
    Check(StateReadModeOptions.TryExtract(argv, out var remaining, out var hash, out var error)
        && hash == Current && error is null
        && remaining.SequenceEqual(argv.Where(a => !a.StartsWith("--bridge-sha256="))),
        "explicit state override preserves positional arguments");
}
foreach (var mode in new[] { "policy-relay", "enter-private", "exit-lost", "schedule", "trial",
    "controller", "g1-held", "unknown", "State" })
    Reject(mode, "--bridge-sha256=" + Current);
foreach (var value in new[] { "", "a", new string('a', 63), new string('a', 65),
    new string('g', 64), Current.ToUpperInvariant(), " " + Current, Current + " " })
    Reject("state", "--bridge-sha256=" + value);
Reject("state", "--bridge-sha256", Current);
Reject("state", "--bridge-sha256x=" + Current);
Reject("state", "--bridge-sha256=" + Current, "--bridge-sha256=" + Current);
Reject("--bridge-sha256=" + Current);
Check(StateReadModeOptions.ExpectedHash(false, null, Legacy) == Legacy, "legacy state pin preserved");
Check(StateReadModeOptions.ExpectedHash(true, null, Legacy) == Legacy, "legacy mutation pin preserved");
Check(StateReadModeOptions.ExpectedHash(false, Current, Legacy) == Current, "explicit read-only pin selected exactly");
Check(StateReadModeOptions.ExpectedHash(false, Current, Legacy) != Legacy, "override does not allow old hash instead");
foreach (var invalid in new[] { (true, Current), (false, "bad") })
{
    var rejected = false;
    try { StateReadModeOptions.ExpectedHash(invalid.Item1, invalid.Item2, Legacy); }
    catch (InvalidDataException) { rejected = true; }
    Check(rejected, "leased or malformed override rejected by validation helper");
}
Console.WriteLine(JsonSerializer.Serialize(new { tests, passed = tests,
    game_connections = 0, lease_commands = 0, input_commands = 0 }));
