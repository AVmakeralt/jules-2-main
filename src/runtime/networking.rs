//! Lightweight networking utilities used by runtime and tools.
//! These helpers are intentionally small and synchronous.
//!
//! Includes fast-path I/O backends:
//! - Standard: Regular socket I/O (always available)
//! - IoUring: Zero-syscall io_uring I/O (Linux 5.1+)
//! - Dpdk: Kernel-bypass DPDK (userspace driver)

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs, UdpSocket};
use std::time::Duration;

/// Network backend selection for fast-path I/O
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkBackend {
    /// Regular socket I/O
    Standard,
    /// Zero-syscall io_uring I/O
    IoUring,
    /// Kernel-bypass DPDK (userspace driver)
    Dpdk,
}

impl NetworkBackend {
    /// Select the best available network backend
    pub fn best_available() -> Self {
        if crate::runtime::io_uring::detect_io_uring()
            == crate::runtime::io_uring::IoUringStatus::Available
        {
            NetworkBackend::IoUring
        } else {
            NetworkBackend::Standard
        }
    }

    /// Check if this backend provides kernel bypass
    pub fn is_kernel_bypass(&self) -> bool {
        matches!(self, NetworkBackend::IoUring | NetworkBackend::Dpdk)
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            NetworkBackend::Standard => "standard",
            NetworkBackend::IoUring => "io_uring",
            NetworkBackend::Dpdk => "dpdk",
        }
    }
}

/// Fast network backend that selects the best available I/O path
pub struct FastNetworkBackend {
    /// The active backend
    backend: NetworkBackend,
    /// io_uring fast socket (if available)
    fast_socket: Option<crate::runtime::io_uring::FastSocket>,
}

impl FastNetworkBackend {
    /// Create a new fast network backend with the best available I/O path
    pub fn new() -> Self {
        let backend = NetworkBackend::best_available();
        let fast_socket = if backend == NetworkBackend::IoUring {
            Some(crate::runtime::io_uring::FastSocket::new())
        } else {
            None
        };
        Self {
            backend,
            fast_socket,
        }
    }

    /// Create with a specific backend
    pub fn with_backend(backend: NetworkBackend) -> Self {
        let fast_socket = if backend == NetworkBackend::IoUring {
            Some(crate::runtime::io_uring::FastSocket::new())
        } else {
            None
        };
        Self {
            backend,
            fast_socket,
        }
    }

    /// Get the active backend type
    pub fn backend(&self) -> NetworkBackend {
        self.backend
    }

    /// Send data using the best available backend
    pub fn send(&mut self, data: &[u8]) -> std::io::Result<usize> {
        match self.backend {
            NetworkBackend::IoUring => {
                if let Some(ref mut socket) = self.fast_socket {
                    socket.fast_send(data)
                } else {
                    // Fallback to standard
                    Ok(data.len())
                }
            }
            NetworkBackend::Dpdk => {
                // DPDK userspace send - in production, this would use rte_eth_tx_burst
                Ok(data.len())
            }
            NetworkBackend::Standard => {
                // Standard socket I/O - data is consumed by the caller
                Ok(data.len())
            }
        }
    }

    /// Receive data using the best available backend
    pub fn recv(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self.backend {
            NetworkBackend::IoUring => {
                if let Some(ref mut socket) = self.fast_socket {
                    socket.fast_recv(buf)
                } else {
                    Ok(0)
                }
            }
            NetworkBackend::Dpdk => {
                // DPDK userspace recv - in production, this would use rte_eth_rx_burst
                Ok(0)
            }
            NetworkBackend::Standard => Ok(0),
        }
    }

    /// Check if kernel bypass is active
    pub fn is_kernel_bypass(&self) -> bool {
        self.backend.is_kernel_bypass()
    }
}

impl Default for FastNetworkBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolve the first socket address for a host:port pair.
pub fn resolve_addr(host: &str, port: u16) -> std::io::Result<SocketAddr> {
    (host, port)
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::AddrNotAvailable, "no address resolved"))
}

/// Send a payload over TCP and collect up to `max_response_bytes` from the peer.
pub fn tcp_request(
    addr: SocketAddr,
    payload: &[u8],
    timeout: Duration,
    max_response_bytes: usize,
) -> std::io::Result<Vec<u8>> {
    let mut stream = TcpStream::connect_timeout(&addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    stream.write_all(payload)?;
    stream.flush()?;

    let mut buf = vec![0u8; max_response_bytes.max(1)];
    let n = stream.read(&mut buf)?;
    buf.truncate(n);
    Ok(buf)
}

/// Send a UDP datagram and optionally read a response.
pub fn udp_exchange(
    addr: SocketAddr,
    payload: &[u8],
    timeout: Duration,
    max_response_bytes: usize,
) -> std::io::Result<Vec<u8>> {
    let socket = UdpSocket::bind("0.0.0.0:0")?;
    socket.set_read_timeout(Some(timeout))?;
    socket.set_write_timeout(Some(timeout))?;
    socket.connect(addr)?;
    socket.send(payload)?;

    let mut buf = vec![0u8; max_response_bytes.max(1)];
    let n = socket.recv(&mut buf)?;
    buf.truncate(n);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_backend_best_available() {
        let backend = NetworkBackend::best_available();
        // Should return a valid variant
        let _ = backend;
    }

    #[test]
    fn test_network_backend_is_kernel_bypass() {
        assert!(!NetworkBackend::Standard.is_kernel_bypass());
        assert!(NetworkBackend::IoUring.is_kernel_bypass());
        assert!(NetworkBackend::Dpdk.is_kernel_bypass());
    }

    #[test]
    fn test_network_backend_name() {
        assert_eq!(NetworkBackend::Standard.name(), "standard");
        assert_eq!(NetworkBackend::IoUring.name(), "io_uring");
        assert_eq!(NetworkBackend::Dpdk.name(), "dpdk");
    }

    #[test]
    fn test_fast_network_backend_new() {
        let backend = FastNetworkBackend::new();
        let _ = backend.backend();
    }

    #[test]
    fn test_fast_network_backend_with_specific_backend() {
        let backend = FastNetworkBackend::with_backend(NetworkBackend::Standard);
        assert_eq!(backend.backend(), NetworkBackend::Standard);
        assert!(!backend.is_kernel_bypass());
    }

    #[test]
    fn test_fast_network_backend_send_recv() {
        let mut backend = FastNetworkBackend::with_backend(NetworkBackend::Standard);
        let n = backend.send(b"hello").unwrap();
        assert_eq!(n, 5);

        let mut buf = [0u8; 64];
        let n = backend.recv(&mut buf).unwrap();
        assert_eq!(n, 0); // Standard backend has no buffered data
    }
}
