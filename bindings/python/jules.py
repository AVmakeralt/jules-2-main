import ctypes
from ctypes import c_char_p, c_uint32


class JulesMlMemorySnapshot(ctypes.Structure):
    _fields_ = [
        ("min_bytes", ctypes.c_size_t),
        ("extra_bytes", ctypes.c_size_t),
        ("core_used_bytes", ctypes.c_size_t),
        ("extra_used_bytes", ctypes.c_size_t),
        ("total_used_bytes", ctypes.c_size_t),
        ("total_cap_bytes", ctypes.c_size_t),
        ("headroom_bytes", ctypes.c_size_t),
    ]


class Jules:
    def __init__(self, lib_path: str = "libjules.so"):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.jules_version.restype = c_uint32
        self.lib.jules_run_file_ffi.argtypes = [c_char_p]
        self.lib.jules_run_file_ffi.restype = ctypes.c_uint
        self.lib.jules_check_code_ffi.argtypes = [c_char_p]
        self.lib.jules_check_code_ffi.restype = ctypes.c_uint
        self.lib.jules_error_string.argtypes = [ctypes.c_uint]
        self.lib.jules_error_string.restype = c_char_p
        self.lib.jules_ml_memory_configure.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        self.lib.jules_ml_memory_configure.restype = ctypes.c_uint
        self.lib.jules_ml_memory_acquire.argtypes = [ctypes.c_size_t, ctypes.c_uint]
        self.lib.jules_ml_memory_acquire.restype = ctypes.c_uint
        self.lib.jules_ml_memory_release.argtypes = [ctypes.c_size_t, ctypes.c_uint]
        self.lib.jules_ml_memory_release.restype = ctypes.c_uint
        self.lib.jules_ml_memory_reset_usage.argtypes = []
        self.lib.jules_ml_memory_reset_usage.restype = ctypes.c_uint
        self.lib.jules_ml_memory_snapshot.argtypes = [ctypes.POINTER(JulesMlMemorySnapshot)]
        self.lib.jules_ml_memory_snapshot.restype = ctypes.c_uint

    def version(self) -> int:
        return int(self.lib.jules_version())

    def run_file(self, path: str) -> int:
        return int(self.lib.jules_run_file_ffi(path.encode("utf-8")))

    def check_code(self, source: str) -> int:
        return int(self.lib.jules_check_code_ffi(source.encode("utf-8")))

    def error_string(self, code: int) -> str:
        msg = self.lib.jules_error_string(code)
        return msg.decode("utf-8") if msg else "Unknown"

    def configure_ml_memory(self, min_bytes: int, extra_bytes: int) -> int:
        return int(self.lib.jules_ml_memory_configure(min_bytes, extra_bytes))

    def acquire_ml_memory(self, bytes_to_acquire: int, pool: str) -> int:
        pool_code = 0 if pool.lower() == "core" else 1
        return int(self.lib.jules_ml_memory_acquire(bytes_to_acquire, pool_code))

    def release_ml_memory(self, bytes_to_release: int, pool: str) -> int:
        pool_code = 0 if pool.lower() == "core" else 1
        return int(self.lib.jules_ml_memory_release(bytes_to_release, pool_code))

    def reset_ml_memory_usage(self) -> int:
        return int(self.lib.jules_ml_memory_reset_usage())

    def ml_memory_snapshot(self) -> dict:
        snapshot = JulesMlMemorySnapshot()
        code = int(self.lib.jules_ml_memory_snapshot(ctypes.byref(snapshot)))
        return {
            "code": code,
            "min_bytes": int(snapshot.min_bytes),
            "extra_bytes": int(snapshot.extra_bytes),
            "core_used_bytes": int(snapshot.core_used_bytes),
            "extra_used_bytes": int(snapshot.extra_used_bytes),
            "total_used_bytes": int(snapshot.total_used_bytes),
            "total_cap_bytes": int(snapshot.total_cap_bytes),
            "headroom_bytes": int(snapshot.headroom_bytes),
        }
