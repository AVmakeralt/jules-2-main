using System;
using System.Runtime.InteropServices;

public static class JulesNative
{
    private const string Dll = "jules";

    public enum JulesError : uint
    {
        Success = 0,
        InvalidArg = 1,
        RuntimeError = 2,
        OutOfMemory = 3,
        NotFound = 4,
        UnknownError = 255,
    }

    public enum JulesMemoryPool : uint
    {
        Core = 0,
        Extra = 1,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct JulesMlMemorySnapshot
    {
        public UIntPtr min_bytes;
        public UIntPtr extra_bytes;
        public UIntPtr core_used_bytes;
        public UIntPtr extra_used_bytes;
        public UIntPtr total_used_bytes;
        public UIntPtr total_cap_bytes;
        public UIntPtr headroom_bytes;
    }

    [DllImport(Dll)] public static extern IntPtr jules_init();
    [DllImport(Dll)] public static extern void jules_destroy(IntPtr ctx);
    [DllImport(Dll)] public static extern uint jules_version();
    [DllImport(Dll)] public static extern IntPtr jules_error_string(JulesError code);

    [DllImport(Dll, CharSet = CharSet.Ansi)]
    public static extern JulesError jules_run_file_ffi(string path);

    [DllImport(Dll, CharSet = CharSet.Ansi)]
    public static extern JulesError jules_check_code_ffi(string source);

    [DllImport(Dll)] public static extern JulesError jules_ml_memory_configure(UIntPtr min_bytes, UIntPtr extra_bytes);
    [DllImport(Dll)] public static extern JulesError jules_ml_memory_acquire(UIntPtr bytes, JulesMemoryPool pool);
    [DllImport(Dll)] public static extern JulesError jules_ml_memory_release(UIntPtr bytes, JulesMemoryPool pool);
    [DllImport(Dll)] public static extern JulesError jules_ml_memory_reset_usage();
    [DllImport(Dll)] public static extern JulesError jules_ml_memory_snapshot(out JulesMlMemorySnapshot snapshot);

    public static string ErrorToString(JulesError err)
        => Marshal.PtrToStringAnsi(jules_error_string(err)) ?? "Unknown";
}
