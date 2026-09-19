using System.Runtime.InteropServices;
using System.Text;

namespace RekUiBridgeAgent;

// Read-only Win32 queries, repeated at each policy mutation boundary. No desktop
// switching, focus, window manipulation, global input, or environment proof.
public static class NativeWindowsDesktopIsolation
{
    public static bool TryVerify(out string? proof, out string reason)
        => TryVerify(Read, out proof, out reason);

    public static bool TryVerify(Func<WindowsDesktopFacts> read, out string? proof, out string reason)
    {
        proof = null;
        try { reason = PolicyExecutionIsolationContract.WindowsRejectReason(read()) ?? ""; }
        catch { reason = "windows_desktop_api_failure"; }
        if (reason.Length != 0) return false;
        proof = PolicyExecutionIsolationContract.WindowsProof;
        return true;
    }

    public static WindowsDesktopFacts Read()
    {
        if (!OperatingSystem.IsWindows()) return new(false, false, null, null, null, null);
        var host = Environment.MachineName;
        var ntdll = GetModuleHandleW("ntdll.dll");
        if (ntdll == IntPtr.Zero) return new(false, false, host, null, null, null);
        var wine = GetProcAddress(ntdll, "wine_get_version");
        if (wine != IntPtr.Zero) return new(true, false, host, null, null, null);
        if (Marshal.GetLastWin32Error() != 127) return new(false, false, host, null, null, null);
        var station = GetProcessWindowStation();
        var desktop = GetThreadDesktop(GetCurrentThreadId());
        if (!TryName(station, out var stationName) || !TryName(desktop, out var desktopName))
            return new(false, true, host, null, null, null);
        var input = OpenInputDesktop(0, false, 0x0001); // DESKTOP_READOBJECTS only.
        if (input == IntPtr.Zero) return new(false, true, host, stationName, desktopName, null);
        bool read; bool closed; string? inputName;
        try { read = TryName(input, out inputName); }
        finally { closed = CloseDesktop(input); }
        return new(read && closed, true, host, stationName, desktopName, inputName);
    }

    private static bool TryName(IntPtr handle, out string? name)
    {
        name = null;
        if (handle == IntPtr.Zero) return false;
        var text = new StringBuilder(256);
        if (!GetUserObjectInformationW(handle, 2, text, (uint)(text.Capacity * 2), out var needed) ||
            needed < 2 || needed > text.Capacity * 2 || (needed & 1) != 0) return false;
        name = text.ToString();
        return name.Length > 0 && needed == (name.Length + 1) * 2;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, ExactSpelling = true, SetLastError = true)]
    private static extern IntPtr GetModuleHandleW(string module);
    [DllImport("kernel32.dll", CharSet = CharSet.Ansi, ExactSpelling = true, SetLastError = true)]
    private static extern IntPtr GetProcAddress(IntPtr module, string name);
    [DllImport("kernel32.dll", ExactSpelling = true)] private static extern uint GetCurrentThreadId();
    [DllImport("user32.dll", ExactSpelling = true, SetLastError = true)] private static extern IntPtr GetProcessWindowStation();
    [DllImport("user32.dll", ExactSpelling = true, SetLastError = true)] private static extern IntPtr GetThreadDesktop(uint thread);
    [DllImport("user32.dll", ExactSpelling = true, SetLastError = true)]
    private static extern IntPtr OpenInputDesktop(uint flags, [MarshalAs(UnmanagedType.Bool)] bool inherit, uint access);
    [DllImport("user32.dll", ExactSpelling = true, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)] private static extern bool CloseDesktop(IntPtr desktop);
    [DllImport("user32.dll", CharSet = CharSet.Unicode, ExactSpelling = true, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)] private static extern bool GetUserObjectInformationW(
        IntPtr handle, int index, StringBuilder data, uint bytes, out uint needed);
}
