using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

internal static class Program
{
    internal const string Desktop = "RekPolicyEval";
    internal const string Station = "WinSta0";
    internal const string InputDesktop = "Default";
    internal const string Host = "D21";
    internal const string RekExecutable = @"C:\Program Files (x86)\Steam\steamapps\common\REK Alpha Test\REK.exe";
    internal const string ExpectedProof = "windows_native=1;station=WinSta0;desktop=RekPolicyEval;input_desktop=Default;host=D21";
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    private sealed record DesktopState(string Host, string Station, string Desktop, string InputDesktop);
    private sealed record ForegroundState(string Hwnd, uint ProcessId);
    private sealed record ProbeReport(bool Passed, int ProcessId, DesktopState Before, DesktopState After,
        string Proof, bool WindowCreated, string WindowHandle, int MessagePumpIterations);

    public static int Main(string[] args)
    {
        try
        {
            if (!OperatingSystem.IsWindows()) throw new InvalidOperationException("Windows is required.");
            if (args.Length == 3 && args[0] == "--probe-child" && args[1] == "--report")
                return ProbeChild(args[2]);
            if (args.Length == 3 && args[0] == "--self-test" && args[1] == "--output-directory")
                return Launch(true, args[2], null, null);
            if (args.Length == 7 && args[0] == "--launch-rek" && args[1] == "--exe" &&
                args[3] == "--sha256" && args[5] == "--output-directory")
                return Launch(false, args[6], args[2], args[4]);
            Console.Error.WriteLine("Usage: --self-test --output-directory ABSOLUTE_NEW_DIRECTORY\n" +
                "       --launch-rek --exe EXACT_REK_PATH --sha256 EXPECTED_SHA256 --output-directory ABSOLUTE_NEW_DIRECTORY");
            return 2;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"{ex.GetType().Name}: {ex.Message}");
            return 1;
        }
    }

    private static int Launch(bool selfTest, string outputDirectory, string? executable, string? expectedHash)
    {
        var contracts = TestContracts();
        using var singleLauncher = new Mutex(false, @"Local\RekIsolatedDesktopLauncher");
        if (!singleLauncher.WaitOne(0)) throw new InvalidOperationException("Another isolated launcher owns this desktop.");
        try
        {
            var before = ReadDesktopState();
            RequireLauncherState(before);
            var foregroundBefore = ReadForeground();
            if (foregroundBefore.Hwnd == "0x0") throw new InvalidOperationException("No foreground window can be observed.");
            RequireNewOutputDirectory(outputDirectory);

            string childExe;
            string childArgs;
            string workingDirectory;
            string? actualHash = null;
            FileStream? pinnedFile = null;
            if (selfTest)
            {
                childExe = Environment.ProcessPath ?? throw new InvalidOperationException("Missing launcher executable path.");
                if (!string.Equals(Path.GetFileName(childExe), "RekIsolatedDesktopLauncher.exe", StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException("Run the built apphost executable, not dotnet DLL, for self-test.");
                childArgs = "--probe-child --report " + Quote(Path.Combine(outputDirectory, "probe.json"));
                workingDirectory = AppContext.BaseDirectory;
            }
            else
            {
                RequireRekPathAndHashSyntax(executable!, expectedHash!);
                RequireNoRek();
                // Keep this read handle open through CreateProcess so the hashed executable cannot be rewritten.
                pinnedFile = new FileStream(executable!, FileMode.Open, FileAccess.Read, FileShare.Read);
                actualHash = Convert.ToHexString(SHA256.HashData(pinnedFile)).ToLowerInvariant();
                if (!string.Equals(actualHash, expectedHash, StringComparison.OrdinalIgnoreCase))
                {
                    pinnedFile.Dispose();
                    throw new InvalidOperationException("REK executable SHA-256 does not match the supplied pin.");
                }
                childExe = Path.GetFullPath(executable!);
                childArgs = "-screen-fullscreen 0 -screen-width 1280 -screen-height 720";
                workingDirectory = Path.GetDirectoryName(childExe)!;
            }

            using (pinnedFile)
            {
                RequireDesktopAbsent();
                var desktopHandle = Native.CreateDesktopW(Desktop, IntPtr.Zero, IntPtr.Zero, 0,
                    Native.DesktopReadObjects | Native.DesktopCreateWindow | Native.DesktopWriteObjects, IntPtr.Zero);
                if (desktopHandle == IntPtr.Zero) throw Win32("CreateDesktopW");
                Native.ProcessInformation process = default;
                var created = false;
                try
                {
                    if (ObjectName(desktopHandle) != Desktop) throw new InvalidOperationException("Created desktop name mismatch.");
                    RequireLauncherState(ReadDesktopState());
                    if (!selfTest) RequireNoRek();
                    var startup = new Native.StartupInfo
                    {
                        Size = (uint)Marshal.SizeOf<Native.StartupInfo>(),
                        Desktop = Station + "\\" + Desktop,
                        Flags = Native.StartfForceOffFeedback
                    };
                    var command = new StringBuilder(Quote(childExe) + " " + childArgs);
                    // CREATE_NO_WINDOW prevents the probe console from appearing. The GUI game uses its assigned desktop.
                    if (!Native.CreateProcessW(childExe, command, IntPtr.Zero, IntPtr.Zero, false,
                        Native.CreateNoWindow, IntPtr.Zero, workingDirectory, ref startup, out process))
                        throw Win32("CreateProcessW");
                    created = true;
                    Native.CloseHandle(process.Thread);
                    process.Thread = IntPtr.Zero;
                    WriteNew(Path.Combine(outputDirectory, "launch.json"), new
                    {
                        mode = selfTest ? "self_test" : "launch_rek", pid = process.ProcessId,
                        executable = childExe, executable_sha256 = actualHash, arguments = childArgs,
                        requested_desktop = startup.Desktop, before, foreground_before = foregroundBefore,
                        utc = DateTimeOffset.UtcNow, contract_checks = contracts,
                        steam_relaunch_containment_proven = false, account_identity_verified_by_launcher = false
                    });

                    var samples = 0;
                    var foregroundChanges = 0;
                    var desktopChanges = 0;
                    ForegroundState? firstChangedForeground = null;
                    string? firstDesktopFailure = null;
                    uint wait;
                    do
                    {
                        wait = Native.WaitForSingleObject(process.Process, 50);
                        if (wait == Native.WaitFailed) throw Win32("WaitForSingleObject");
                        samples++;
                        var foreground = ReadForeground();
                        if (foreground != foregroundBefore)
                        {
                            foregroundChanges++;
                            firstChangedForeground ??= foreground;
                        }
                        try { RequireLauncherState(ReadDesktopState()); }
                        catch (Exception ex) { desktopChanges++; firstDesktopFailure ??= ex.Message; }
                    } while (wait == Native.WaitTimeout);
                    if (wait != Native.WaitObject0) throw new InvalidOperationException("Unexpected child wait result.");
                    if (!Native.GetExitCodeProcess(process.Process, out var exitCode)) throw Win32("GetExitCodeProcess");
                    var foregroundAfter = ReadForeground();
                    var after = ReadDesktopState();
                    ProbeReport? probe = null;
                    if (selfTest)
                    {
                        probe = JsonSerializer.Deserialize<ProbeReport>(File.ReadAllText(Path.Combine(outputDirectory, "probe.json")));
                        if (probe is null || probe.ProcessId != process.ProcessId || probe.Proof != ExpectedProof)
                            throw new InvalidOperationException("Probe identity/proof mismatch.");
                    }
                    var passed = exitCode == 0 && foregroundChanges == 0 && desktopChanges == 0 &&
                        foregroundAfter == foregroundBefore && (!selfTest || probe!.Passed);
                    var report = new
                    {
                        passed, mode = selfTest ? "self_test" : "launch_rek", pid = process.ProcessId,
                        child_exit_code = exitCode, contract_checks = contracts, samples,
                        foreground_changes = foregroundChanges, desktop_changes = desktopChanges,
                        foreground_before = foregroundBefore, foreground_after = foregroundAfter,
                        first_changed_foreground = firstChangedForeground, first_desktop_failure = firstDesktopFailure,
                        before, after, probe,
                        desktop_retained_until_child_exit = true,
                        polling_limit = "50 ms samples are evidence, not proof against shorter transients; no corrective focus actions.",
                        utc = DateTimeOffset.UtcNow
                    };
                    WriteNew(Path.Combine(outputDirectory, "result.json"), report);
                    Console.WriteLine(JsonSerializer.Serialize(report, JsonOptions));
                    return passed ? 0 : 1;
                }
                finally
                {
                    // Even an evidence-write failure must not release our desktop handle while the direct child lives.
                    if (created && process.Process != IntPtr.Zero)
                    {
                        Native.WaitForSingleObject(process.Process, Native.Infinite);
                        Native.CloseHandle(process.Process);
                    }
                    if (process.Thread != IntPtr.Zero) Native.CloseHandle(process.Thread);
                    Native.CloseDesktop(desktopHandle);
                }
            }
        }
        finally { singleLauncher.ReleaseMutex(); }
    }

    private static int ProbeChild(string reportPath)
    {
        RequireLocalAbsolutePath(reportPath);
        var before = ReadDesktopState();
        RequireProbeState(before);
        // The only window created by this program belongs to this verified non-input desktop.
        var window = Native.CreateWindowExW(Native.WsExNoActivate | Native.WsExToolWindow, "STATIC",
            "REK isolated desktop probe", Native.WsOverlappedWindow | Native.WsVisible,
            16, 16, 320, 120, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero);
        if (window == IntPtr.Zero) throw Win32("CreateWindowExW probe");
        var iterations = 0;
        try
        {
            var timer = Stopwatch.StartNew();
            while (timer.ElapsedMilliseconds < 2000)
            {
                RequireProbeState(ReadDesktopState());
                while (Native.PeekMessageW(out var message, window, 0, 0, Native.PmRemove))
                {
                    Native.TranslateMessage(ref message);
                    Native.DispatchMessageW(ref message);
                }
                iterations++;
                Thread.Sleep(10);
            }
            var after = ReadDesktopState();
            RequireProbeState(after);
            WriteNew(reportPath, new ProbeReport(true, Environment.ProcessId, before, after,
                ExpectedProof, true, "0x" + window.ToInt64().ToString("x"), iterations));
            return 0;
        }
        finally { Native.DestroyWindow(window); }
    }

    private static DesktopState ReadDesktopState()
    {
        var input = Native.OpenInputDesktop(0, false, Native.DesktopReadObjects);
        if (input == IntPtr.Zero) throw Win32("OpenInputDesktop");
        try
        {
            return new DesktopState(Environment.MachineName, ObjectName(Native.GetProcessWindowStation()),
                ObjectName(Native.GetThreadDesktop(Native.GetCurrentThreadId())), ObjectName(input));
        }
        finally { Native.CloseDesktop(input); }
    }

    private static string ObjectName(IntPtr handle)
    {
        if (handle == IntPtr.Zero) throw new InvalidOperationException("Missing desktop/window-station handle.");
        var text = new StringBuilder(256);
        if (!Native.GetUserObjectInformationW(handle, Native.UoiName, text, (uint)(text.Capacity * 2), out _))
            throw Win32("GetUserObjectInformationW");
        return text.ToString();
    }

    private static ForegroundState ReadForeground()
    {
        var window = Native.GetForegroundWindow();
        Native.GetWindowThreadProcessId(window, out var pid);
        return new ForegroundState("0x" + window.ToInt64().ToString("x"), pid);
    }

    private static void RequireLauncherState(DesktopState state)
    {
        if (state != new DesktopState(Host, Station, InputDesktop, InputDesktop))
            throw new InvalidOperationException("Launcher requires D21, WinSta0, thread desktop Default and input desktop Default.");
    }

    private static void RequireProbeState(DesktopState state)
    {
        if (state != new DesktopState(Host, Station, Desktop, InputDesktop))
            throw new InvalidOperationException("Child requires D21, WinSta0, RekPolicyEval and distinct input desktop Default.");
    }

    private static void RequireNoRek()
    {
        var processes = Process.GetProcessesByName("REK");
        try
        {
            if (processes.Length != 0) throw new InvalidOperationException("An existing REK process is present; it will not be modified.");
        }
        finally { foreach (var process in processes) process.Dispose(); }
    }

    private static void RequireDesktopAbsent()
    {
        var existing = Native.OpenDesktopW(Desktop, 0, false, Native.DesktopReadObjects);
        if (existing != IntPtr.Zero)
        {
            Native.CloseDesktop(existing);
            throw new InvalidOperationException("RekPolicyEval already exists; refusing to reuse another desktop owner.");
        }
        var error = Marshal.GetLastWin32Error();
        if (error != 2) throw new Win32Exception(error, "Cannot prove RekPolicyEval is absent.");
    }

    private static void RequireRekPathAndHashSyntax(string executable, string hash)
    {
        RequireLocalAbsolutePath(executable);
        if (!string.Equals(Path.GetFullPath(executable), RekExecutable, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Executable must be the exact installed REK.exe path.");
        if (hash.Length != 64 || !hash.All(Uri.IsHexDigit))
            throw new InvalidOperationException("An explicit 64-digit executable SHA-256 pin is required.");
    }

    private static void RequireLocalAbsolutePath(string path)
    {
        if (!Path.IsPathFullyQualified(path) || path.StartsWith(@"\\", StringComparison.Ordinal) ||
            path.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                .Any(segment => segment.Contains("OneDrive", StringComparison.OrdinalIgnoreCase)))
            throw new InvalidOperationException("Use an absolute local path outside OneDrive.");
    }

    private static void RequireNewOutputDirectory(string path)
    {
        RequireLocalAbsolutePath(path);
        if (Directory.Exists(path) || File.Exists(path)) throw new InvalidOperationException("Output directory must not already exist.");
        Directory.CreateDirectory(path);
    }

    private static void WriteNew<T>(string path, T value)
    {
        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
        JsonSerializer.Serialize(stream, value, JsonOptions);
    }

    private static string Quote(string value)
    {
        if (value.Contains('"') || value.Contains('\r') || value.Contains('\n') || value.EndsWith('\\'))
            throw new InvalidOperationException("Unsupported command-line path characters.");
        return "\"" + value + "\"";
    }

    private static Exception Win32(string operation) => new Win32Exception(Marshal.GetLastWin32Error(), operation);

    private static int TestContracts()
    {
        var checks = 0;
        void Accept(Action action) { action(); checks++; }
        void Reject(Action action)
        {
            try { action(); }
            catch (InvalidOperationException) { checks++; return; }
            throw new InvalidOperationException("Contract regression: invalid input accepted.");
        }
        Accept(() => RequireLauncherState(new(Host, Station, InputDesktop, InputDesktop)));
        Accept(() => RequireProbeState(new(Host, Station, Desktop, InputDesktop)));
        Reject(() => RequireProbeState(new(Host, Station, InputDesktop, InputDesktop)));
        Reject(() => RequireProbeState(new(Host, Station, Desktop, Desktop)));
        Reject(() => RequireProbeState(new("OTHER", Station, Desktop, InputDesktop)));
        Reject(() => RequireProbeState(new(Host, "Service-0", Desktop, InputDesktop)));
        Reject(() => RequireLauncherState(new(Host, Station, InputDesktop, "Winlogon")));
        Accept(() => RequireRekPathAndHashSyntax(RekExecutable, new string('a', 64)));
        Reject(() => RequireRekPathAndHashSyntax(@"C:\other\REK.exe", new string('a', 64)));
        Reject(() => RequireRekPathAndHashSyntax(RekExecutable, "bad"));
        Reject(() => RequireRekPathAndHashSyntax(RekExecutable, new string('g', 64)));
        Reject(() => RequireLocalAbsolutePath(@"C:\Users\Daniel\OneDrive\probe.json"));
        Reject(() => RequireLocalAbsolutePath(@"\\server\share\probe.json"));
        Reject(() => RequireLocalAbsolutePath("relative.json"));
        Accept(() => { if (Quote(@"C:\space path\probe.json") != "\"C:\\space path\\probe.json\"") throw new InvalidOperationException("Quote regression."); });
        Reject(() => Quote("bad\"path"));
        return checks;
    }

    private static class Native
    {
        internal const uint DesktopReadObjects = 0x1, DesktopCreateWindow = 0x2, DesktopWriteObjects = 0x80;
        internal const uint StartfForceOffFeedback = 0x80, CreateNoWindow = 0x08000000;
        internal const uint WaitObject0 = 0, WaitTimeout = 258, WaitFailed = 0xffffffff, Infinite = 0xffffffff;
        internal const uint WsExNoActivate = 0x08000000, WsExToolWindow = 0x80;
        internal const uint WsOverlappedWindow = 0x00cf0000, WsVisible = 0x10000000, PmRemove = 1;
        internal const int UoiName = 2;

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        internal struct StartupInfo
        {
            public uint Size;
            public string? Reserved, Desktop, Title;
            public uint X, Y, XSize, YSize, XCountChars, YCountChars, FillAttribute, Flags;
            public ushort ShowWindow, Reserved2Size;
            public IntPtr Reserved2, StdInput, StdOutput, StdError;
        }
        [StructLayout(LayoutKind.Sequential)]
        internal struct ProcessInformation { public IntPtr Process, Thread; public uint ProcessId, ThreadId; }
        [StructLayout(LayoutKind.Sequential)]
        internal struct Message
        {
            public IntPtr Window;
            public uint Id;
            public UIntPtr WParam;
            public IntPtr LParam;
            public uint Time;
            public int X, Y;
            public uint Private;
        }

        [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        internal static extern IntPtr CreateDesktopW(string name, IntPtr device, IntPtr mode, uint flags, uint access, IntPtr security);
        [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        internal static extern IntPtr OpenDesktopW(string name, uint flags, [MarshalAs(UnmanagedType.Bool)] bool inherit, uint access);
        [DllImport("user32.dll", SetLastError = true)]
        internal static extern IntPtr OpenInputDesktop(uint flags, [MarshalAs(UnmanagedType.Bool)] bool inherit, uint access);
        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)] internal static extern bool CloseDesktop(IntPtr desktop);
        [DllImport("user32.dll")] internal static extern IntPtr GetProcessWindowStation();
        [DllImport("user32.dll")] internal static extern IntPtr GetThreadDesktop(uint threadId);
        [DllImport("kernel32.dll")] internal static extern uint GetCurrentThreadId();
        [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)] internal static extern bool GetUserObjectInformationW(IntPtr handle, int index, StringBuilder text, uint bytes, out uint needed);
        [DllImport("user32.dll")] internal static extern IntPtr GetForegroundWindow();
        [DllImport("user32.dll")] internal static extern uint GetWindowThreadProcessId(IntPtr window, out uint processId);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)] internal static extern bool CreateProcessW(string application, StringBuilder command,
            IntPtr processAttributes, IntPtr threadAttributes, [MarshalAs(UnmanagedType.Bool)] bool inherit, uint flags,
            IntPtr environment, string directory, ref StartupInfo startup, out ProcessInformation process);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)] internal static extern bool GetExitCodeProcess(IntPtr process, out uint exitCode);
        [DllImport("kernel32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)] internal static extern bool CloseHandle(IntPtr handle);
        [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        internal static extern IntPtr CreateWindowExW(uint extendedStyle, string className, string title, uint style,
            int x, int y, int width, int height, IntPtr parent, IntPtr menu, IntPtr instance, IntPtr parameter);
        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)] internal static extern bool DestroyWindow(IntPtr window);
        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)] internal static extern bool PeekMessageW(out Message message, IntPtr window, uint min, uint max, uint remove);
        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)] internal static extern bool TranslateMessage(ref Message message);
        [DllImport("user32.dll")] internal static extern IntPtr DispatchMessageW(ref Message message);
    }
}
