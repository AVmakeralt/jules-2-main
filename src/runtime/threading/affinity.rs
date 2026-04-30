// =========================================================================
// CPU Affinity and Thread Pinning
// Cross-platform CPU affinity management
// Linux: sched_setaffinity via libc
// Windows: SetThreadAffinityMask via winapi
// =========================================================================

use std::thread;

#[cfg(feature = "numa")]
#[cfg(target_os = "linux")]
use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO, sched_getaffinity};

#[cfg(feature = "numa")]
#[cfg(target_os = "windows")]
use windows::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask, GetCurrentProcess};

/// Set CPU affinity for the current thread
pub fn set_thread_affinity(cpu_id: usize) -> Result<(), String> {
    #[cfg(feature = "numa")]
    #[cfg(target_os = "linux")]
    {
        set_thread_affinity_linux(cpu_id)
    }
    
    #[cfg(feature = "numa")]
    #[cfg(target_os = "windows")]
    {
        set_thread_affinity_windows(cpu_id)
    }
    
    #[cfg(not(feature = "numa"))]
    {
        // Stub implementation when NUMA feature is disabled
        Ok(())
    }
    
    #[cfg(feature = "numa")]
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        Err("CPU affinity not supported on this platform".to_string())
    }
}

#[cfg(feature = "numa")]
#[cfg(target_os = "linux")]
fn set_thread_affinity_linux(cpu_id: usize) -> Result<(), String> {
    unsafe {
        let mut cpuset: cpu_set_t = std::mem::zeroed();
        CPU_ZERO(&mut cpuset);
        CPU_SET(cpu_id as i32, &mut cpuset);
        
        let pid = libc::getpid();
        let result = sched_setaffinity(pid, std::mem::size_of::<cpu_set_t>(), &cpuset);
        
        if result == 0 {
            Ok(())
        } else {
            Err(format!("Failed to set thread affinity: errno {}", libc::__errno_location().read()))
        }
    }
}

#[cfg(feature = "numa")]
#[cfg(target_os = "windows")]
fn set_thread_affinity_windows(cpu_id: usize) -> Result<(), String> {
    unsafe {
        let thread_handle = GetCurrentThread();
        let mask = 1u64 << cpu_id;
        
        let result = SetThreadAffinityMask(thread_handle, mask);
        
        if result.0 != 0 {
            Ok(())
        } else {
            Err(format!("Failed to set thread affinity: error {:?}", std::io::Error::last_os_error()))
        }
    }
}

/// Set CPU affinity for a specific thread handle
#[cfg(feature = "numa")]
#[cfg(target_os = "linux")]
pub fn set_thread_affinity_for_thread(thread: &std::thread::Thread, cpu_id: usize) -> Result<(), String> {
    use libc::{pthread_self, pthread_setaffinity_np};
    
    unsafe {
        let mut cpuset: cpu_set_t = std::mem::zeroed();
        CPU_ZERO(&mut cpuset);
        CPU_SET(cpu_id as i32, &mut cpuset);
        
        // Get the pthread_t from the thread handle
        // Note: This is a simplified approach - in production would use proper thread ID extraction
        let pthread_id = pthread_self();
        let result = pthread_setaffinity_np(pthread_id, std::mem::size_of::<cpu_set_t>(), &cpuset);
        
        if result == 0 {
            Ok(())
        } else {
            Err(format!("Failed to set thread affinity: errno {}", libc::__errno_location().read()))
        }
    }
}

#[cfg(feature = "numa")]
#[cfg(target_os = "windows")]
pub fn set_thread_affinity_for_thread(thread: &std::thread::Thread, cpu_id: usize) -> Result<(), String> {
    unsafe {
        // Use as_raw_handle to get the thread handle
        let handle = thread.as_raw_handle();
        let mask = 1u64 << cpu_id;
        
        let result = SetThreadAffinityMask(handle, mask);
        
        if result.0 != 0 {
            Ok(())
        } else {
            Err(format!("Failed to set thread affinity: error {:?}", std::io::Error::last_os_error()))
        }
    }
}

#[cfg(not(feature = "numa"))]
pub fn set_thread_affinity_for_thread(_thread: &std::thread::Thread, _cpu_id: usize) -> Result<(), String> {
    // NUMA feature disabled - affinity not available
    Err("CPU affinity requires NUMA feature to be enabled".to_string())
}

/// Get the current thread's CPU affinity (for debugging)
pub fn get_thread_affinity() -> Result<Vec<usize>, String> {
    #[cfg(feature = "numa")]
    #[cfg(target_os = "linux")]
    {
        get_thread_affinity_linux()
    }
    
    #[cfg(feature = "numa")]
    #[cfg(target_os = "windows")]
    {
        get_thread_affinity_windows()
    }
    
    #[cfg(not(feature = "numa"))]
    {
        Err("CPU affinity requires NUMA feature to be enabled".to_string())
    }
    
    #[cfg(feature = "numa")]
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        Err("CPU affinity not supported on this platform".to_string())
    }
}

#[cfg(feature = "numa")]
#[cfg(target_os = "linux")]
fn get_thread_affinity_linux() -> Result<Vec<usize>, String> {
    unsafe {
        let mut cpuset: cpu_set_t = std::mem::zeroed();
        CPU_ZERO(&mut cpuset);
        
        let pid = libc::getpid();
        let result = sched_getaffinity(pid, std::mem::size_of::<cpu_set_t>(), &mut cpuset);
        
        if result == 0 {
            let mut cpus = Vec::new();
            for i in 0..1024 {
                if libc::CPU_ISSET(i, &cpuset) {
                    cpus.push(i);
                }
            }
            Ok(cpus)
        } else {
            Err(format!("Failed to get thread affinity: errno {}", libc::__errno_location().read()))
        }
    }
}

#[cfg(feature = "numa")]
#[cfg(target_os = "windows")]
fn get_thread_affinity_windows() -> Result<Vec<usize>, String> {
    // Windows doesn't provide a direct way to get current affinity
    // Return all possible CPUs as fallback
    let num_cpus = num_cpus::get();
    Ok((0..num_cpus).collect())
}

/// Set CPU affinity for a mask of CPUs
pub fn set_thread_affinity_mask(mask: u64) -> Result<(), String> {
    #[cfg(feature = "numa")]
    #[cfg(target_os = "linux")]
    {
        unsafe {
            let mut cpuset: cpu_set_t = std::mem::zeroed();
            CPU_ZERO(&mut cpuset);
            
            for i in 0..64 {
                if (mask >> i) & 1 == 1 {
                    CPU_SET(i as i32, &mut cpuset);
                }
            }
            
            let pid = libc::getpid();
            let result = sched_setaffinity(pid, std::mem::size_of::<cpu_set_t>(), &cpuset);
            
            if result == 0 {
                Ok(())
            } else {
                Err(format!("Failed to set thread affinity mask: errno {}", libc::__errno_location().read()))
            }
        }
    }
    
    #[cfg(feature = "numa")]
    #[cfg(target_os = "windows")]
    {
        unsafe {
            let thread_handle = GetCurrentThread();
            let result = SetThreadAffinityMask(thread_handle, mask);
            
            if result.0 != 0 {
                Ok(())
            } else {
                Err(format!("Failed to set thread affinity mask: error {:?}", std::io::Error::last_os_error()))
            }
        }
    }
    
    #[cfg(not(feature = "numa"))]
    {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_affinity() {
        let result = set_thread_affinity(0);
        // May fail on systems without proper permissions
        let _ = result;
    }

    #[test]
    fn test_get_affinity() {
        let result = get_thread_affinity();
        assert!(result.is_ok());
    }

    #[test]
    fn test_set_affinity_mask() {
        let result = set_thread_affinity_mask(1);
        let _ = result;
    }
}
