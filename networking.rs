//! Lightweight networking utilities used by runtime and tools.
//! These helpers are intentionally small and synchronous.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs, UdpSocket};
use std::time::Duration;

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
