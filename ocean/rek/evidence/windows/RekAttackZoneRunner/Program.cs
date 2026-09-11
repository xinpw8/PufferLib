using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using RekUiBridgeAgent;

const string ArtifactSchema = "rek.attack_zone_schedule_artifact.v1";

try
{
    if (args.Length == 0)
        return Usage();
    return args[0] switch
    {
        "generate" => Generate(args),
        "command" => Command(args),
        "validate-coverage" => ValidateCoverage(args),
        "self-test" => SelfTest(args),
        "execute" => Execute(args, resume: false),
        "resume" => Execute(args, resume: true),
        _ => Usage(),
    };
}
catch (Exception exception)
{
    Console.Error.WriteLine($"{exception.GetType().Name}: {exception.Message}");
    return 1;
}

static int Execute(string[] arguments, bool resume)
{
    if (arguments.Length is < 3 or > 5 ||
        (arguments.Length >= 4 && !int.TryParse(arguments[3], out _)) ||
        (arguments.Length == 5 && !int.TryParse(arguments[4], out _)))
    {
        return Usage();
    }
    var artifactPath = Path.GetFullPath(arguments[1]);
    var artifact = LoadArtifact(artifactPath);
    var maximumTrials = arguments.Length >= 4
        ? int.Parse(arguments[3])
        : artifact.Entries.Count;
    var trialTimeoutSeconds = arguments.Length == 5
        ? int.Parse(arguments[4])
        : 180;
    return AttackZonePipeExecutor.ExecuteAsync(
            artifact,
            artifactPath,
            arguments[2],
            resume,
            maximumTrials,
            trialTimeoutSeconds)
        .GetAwaiter()
        .GetResult();
}

static int Generate(string[] arguments)
{
    if (arguments.Length is < 5 or > 6 ||
        !int.TryParse(arguments[3], out var runOrdinal) ||
        !int.TryParse(arguments[4], out var repetitions))
    {
        return Usage();
    }
    var runId = arguments[1];
    var seed = arguments[2];
    var entries = AttackZoneTrialContract.BuildRandomizedSchedule(
        runId,
        runOrdinal,
        seed,
        repetitions);
    var canonical = AttackZoneTrialContract.BuildScheduleCanonicalJson(
        runId,
        runOrdinal,
        seed,
        repetitions,
        entries);
    var scheduleSha256 = AttackZoneTrialContract.ComputeScheduleSha256(
        runId,
        runOrdinal,
        seed,
        repetitions,
        entries);
    using var schedule = JsonDocument.Parse(canonical);
    var artifact = JsonSerializer.Serialize(new
    {
        artifact_schema = ArtifactSchema,
        schedule_sha256 = scheduleSha256,
        schedule = schedule.RootElement,
    });
    if (arguments.Length == 6)
    {
        var outputPath = Path.GetFullPath(arguments[5]);
        var parent = Path.GetDirectoryName(outputPath) ??
            throw new InvalidOperationException("output path had no parent directory");
        Directory.CreateDirectory(parent);
        var temporaryPath = outputPath + $".tmp-{Guid.NewGuid():N}";
        PublishNewText(outputPath, temporaryPath, artifact + Environment.NewLine);
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            status = "created",
            path = outputPath,
            schedule_sha256 = scheduleSha256,
            entry_count = entries.Count,
            independent_run_id = runId,
            independent_run_ordinal = runOrdinal,
        }));
    }
    else
    {
        Console.WriteLine(artifact);
    }
    return 0;
}

static int Command(string[] arguments)
{
    if (arguments.Length != 8 ||
        !int.TryParse(arguments[2], out var scheduleOrdinal) ||
        !int.TryParse(arguments[6], out var actionSequence))
    {
        return Usage();
    }
    var artifact = LoadArtifact(arguments[1]);
    if (scheduleOrdinal < 0 || scheduleOrdinal >= artifact.Entries.Count)
        throw new InvalidDataException("schedule ordinal was outside the frozen schedule");
    var entry = artifact.Entries[scheduleOrdinal];
    if (entry.ScheduleOrdinal != scheduleOrdinal)
        throw new InvalidDataException("schedule ordinal was not canonical");
    var target = AttackZoneTrialContract.CreateTarget(
        entry,
        artifact.ScheduleSha256,
        artifact.Seed,
        arguments[3],
        arguments[4],
        arguments[5],
        actionSequence);
    if (!AttackZoneTrialContract.TryValidateTarget(target, out _, out var reason))
        throw new InvalidDataException(reason);
    using var targetDocument = JsonDocument.Parse(
        AttackZoneTrialContract.SerializeTarget(target));
    var command = JsonSerializer.Serialize(new
    {
        type = "command",
        request_id = arguments[7],
        command = "StartAttackZoneTrial",
        target = targetDocument.RootElement,
    });
    Console.WriteLine(command);
    return 0;
}

static int ValidateCoverage(string[] arguments)
{
    if (arguments.Length < 2)
        return Usage();
    var allEntries = new List<AttackZoneScheduleEntry>();
    var hashes = new List<string>();
    foreach (var path in arguments.Skip(1))
    {
        var artifact = LoadArtifact(path);
        allEntries.AddRange(artifact.Entries);
        hashes.Add(artifact.ScheduleSha256);
    }
    var validation = AttackZoneTrialContract.ValidateIndependentCoverage(allEntries);
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        complete = validation.Complete,
        required_independent_runs_per_cell =
            AttackZoneTrialContract.MinimumIndependentRunsPerCell,
        supplied_schedule_count = arguments.Length - 1,
        supplied_schedule_sha256 = hashes,
        missing_cell_count = validation.MissingCells.Count,
        missing_cells = validation.MissingCells.Select(value => new
        {
            cell_key = value.CellKey,
            independent_run_count = value.IndependentRunCount,
            required_independent_run_count = value.RequiredIndependentRunCount,
        }).ToArray(),
    }));
    return validation.Complete ? 0 : 2;
}

static int SelfTest(string[] arguments)
{
    if (arguments.Length != 1)
        return Usage();
    var directory = Path.Combine(
        Path.GetTempPath(),
        $"rek-attack-zone-runner-test-{Guid.NewGuid():N}");
    Directory.CreateDirectory(directory);
    var destination = Path.Combine(directory, "schedule.json");
    var temporary = Path.Combine(directory, "schedule.tmp");
    try
    {
        File.WriteAllText(destination, "preserve", new UTF8Encoding(false));
        var rejected = false;
        try
        {
            PublishNewText(destination, temporary, "replacement");
        }
        catch (IOException)
        {
            rejected = true;
        }
        if (!rejected || File.ReadAllText(destination) != "preserve")
            throw new InvalidDataException("pre-existing schedule was not preserved fail-closed");
        if (File.Exists(temporary))
            throw new InvalidDataException("failed publication left a temporary schedule");
        AttackZonePipeExecutor.RunFileSafetySelfTest();
        Console.WriteLine(
            "PASS no-replace publication; resume rejects gaps, partials, transcript tampering, and manifest runner/schema drift");
        return 0;
    }
    finally
    {
        if (File.Exists(temporary))
            File.Delete(temporary);
        if (File.Exists(destination))
            File.Delete(destination);
        Directory.Delete(directory);
    }
}

static void PublishNewText(string destination, string temporary, string content)
{
    if (File.Exists(destination))
        throw new IOException($"artifact destination already exists: {destination}");
    try
    {
        File.WriteAllText(temporary, content, new UTF8Encoding(false));
        File.Move(temporary, destination);
    }
    finally
    {
        if (File.Exists(temporary))
            File.Delete(temporary);
    }
}

static LoadedArtifact LoadArtifact(string path)
{
    var fullPath = Path.GetFullPath(path);
    using var document = JsonDocument.Parse(File.ReadAllText(fullPath));
    var root = document.RootElement;
    RequireExactProperties(root, "artifact_schema", "schedule_sha256", "schedule");
    RequireString(root, "artifact_schema", ArtifactSchema);
    var declaredHash = RequiredString(root, "schedule_sha256");
    var schedule = root.GetProperty("schedule");
    RequireExactProperties(schedule,
        "attack_zone_trial_schema",
        "protocol_sha256",
        "schedule_schema",
        "randomization_algorithm",
        "randomization_seed_hex",
        "independent_run_id",
        "independent_run_ordinal",
        "repetitions_per_cell",
        "required_independent_runs_per_cell",
        "entries");
    RequireString(schedule, "attack_zone_trial_schema", AttackZoneTrialContract.Schema);
    RequireString(schedule, "protocol_sha256", AttackZoneTrialContract.ExpectedSha256);
    RequireString(schedule, "schedule_schema", AttackZoneTrialContract.ScheduleSchema);
    RequireString(schedule, "randomization_algorithm",
        AttackZoneTrialContract.RandomizationAlgorithm);
    var seed = RequiredString(schedule, "randomization_seed_hex");
    var runId = RequiredString(schedule, "independent_run_id");
    var runOrdinal = RequiredInt32(schedule, "independent_run_ordinal");
    var repetitions = RequiredInt32(schedule, "repetitions_per_cell");
    if (RequiredInt32(schedule, "required_independent_runs_per_cell") !=
        AttackZoneTrialContract.MinimumIndependentRunsPerCell)
    {
        throw new InvalidDataException("independent-run quota mismatch");
    }
    var regenerated = AttackZoneTrialContract.BuildRandomizedSchedule(
        runId, runOrdinal, seed, repetitions);
    var canonical = AttackZoneTrialContract.BuildScheduleCanonicalJson(
        runId, runOrdinal, seed, repetitions, regenerated);
    var observedHash = Convert.ToHexString(SHA256.HashData(
        Encoding.UTF8.GetBytes(schedule.GetRawText()))).ToLowerInvariant();
    var regeneratedHash = AttackZoneTrialContract.ComputeScheduleSha256(
        runId, runOrdinal, seed, repetitions, regenerated);
    if (!string.Equals(observedHash, declaredHash, StringComparison.Ordinal) ||
        !string.Equals(regeneratedHash, declaredHash, StringComparison.Ordinal) ||
        !string.Equals(canonical, schedule.GetRawText(), StringComparison.Ordinal))
    {
        throw new InvalidDataException("schedule content or SHA-256 was not canonical");
    }
    return new LoadedArtifact(
        declaredHash,
        seed,
        runId,
        runOrdinal,
        repetitions,
        regenerated);
}

static string RequiredString(JsonElement parent, string name)
{
    var value = parent.GetProperty(name);
    if (value.ValueKind != JsonValueKind.String || string.IsNullOrEmpty(value.GetString()))
        throw new InvalidDataException($"expected nonempty string {name}");
    return value.GetString()!;
}

static int RequiredInt32(JsonElement parent, string name)
{
    var value = parent.GetProperty(name);
    if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt32(out var result))
        throw new InvalidDataException($"expected integer {name}");
    return result;
}

static void RequireString(JsonElement parent, string name, string expected)
{
    if (RequiredString(parent, name) != expected)
        throw new InvalidDataException($"expected {name}={expected}");
}

static void RequireExactProperties(JsonElement element, params string[] expected)
{
    if (element.ValueKind != JsonValueKind.Object)
        throw new InvalidDataException("expected JSON object");
    var names = new HashSet<string>(StringComparer.Ordinal);
    foreach (var property in element.EnumerateObject())
    {
        if (!names.Add(property.Name))
            throw new InvalidDataException("duplicate JSON property");
    }
    if (names.Count != expected.Length || expected.Any(value => !names.Contains(value)))
        throw new InvalidDataException("unexpected JSON object shape");
}

static int Usage()
{
    Console.Error.WriteLine(
        "usage:\n" +
        "  RekAttackZoneRunner generate RUN_ID RUN_ORDINAL SEED_HEX REPETITIONS [OUTPUT]\n" +
        "  RekAttackZoneRunner command SCHEDULE ORDINAL SESSION_SHA ROUND_SHA TRIAL_ID ACTION_SEQUENCE REQUEST_ID\n" +
        "  RekAttackZoneRunner validate-coverage SCHEDULE [SCHEDULE ...]\n" +
        "  RekAttackZoneRunner execute SCHEDULE NEW_OUTPUT_DIRECTORY [MAX_TRIALS] [TRIAL_TIMEOUT_SECONDS]\n" +
        "  RekAttackZoneRunner resume SCHEDULE EXISTING_OUTPUT_DIRECTORY [MAX_TRIALS] [TRIAL_TIMEOUT_SECONDS]\n" +
        "  RekAttackZoneRunner self-test");
    return 64;
}

internal sealed record LoadedArtifact(
    string ScheduleSha256,
    string Seed,
    string IndependentRunId,
    int IndependentRunOrdinal,
    int RepetitionsPerCell,
    IReadOnlyList<AttackZoneScheduleEntry> Entries);
