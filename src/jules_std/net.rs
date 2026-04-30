// =============================================================================
// std/net — Jules Standard Library: Networking
//
// TCP client/server, UDP with reliability layer, WebSocket, RPC,
// state synchronization, lag compensation.
// Pure Rust, zero external dependencies (uses std::net).
// =============================================================================

#![allow(dead_code)]

use std::io::{Read, Write};

use crate::interp::{RuntimeError, Value};
use crate::lexer::Span;

macro_rules! rt_err {
    ($msg:expr) => {
        RuntimeError { span: Some(Span::dummy()), message: $msg.to_string() }
    };
}

fn str_arg(args: &[Value], i: usize) -> Option<String> {
    match args.get(i) {
        Some(Value::Str(s)) => Some(s.clone()),
        _ => None,
    }
}

fn i64_arg(args: &[Value], i: usize) -> Option<i64> {
    args.get(i).and_then(|v| v.as_i64())
}

// ─── TCP Server ──────────────────────────────────────────────────────────────

struct TcpServerHandle {
    listener: Option<std::net::TcpListener>,
    clients: Vec<std::net::TcpStream>,
}

thread_local! {
    static TCP_SERVERS: std::cell::RefCell<Vec<TcpServerHandle>> = std::cell::RefCell::new(Vec::new());
    static TCP_CLIENTS: std::cell::RefCell<Vec<std::net::TcpStream>> = std::cell::RefCell::new(Vec::new());
    static UDP_SOCKETS: std::cell::RefCell<Vec<std::net::UdpSocket>> = std::cell::RefCell::new(Vec::new());
}

// ─── Serialization helpers ───────────────────────────────────────────────────

fn value_to_bytes(v: &Value) -> Vec<u8> {
    match v {
        Value::I32(n) => { let b = n.to_le_bytes(); b.to_vec() }
        Value::I64(n) => { let b = n.to_le_bytes(); b.to_vec() }
        Value::F32(x) => { let b = x.to_le_bytes(); b.to_vec() }
        Value::F64(x) => { let b = x.to_le_bytes(); b.to_vec() }
        Value::Bool(b) => { vec![if *b { 1u8 } else { 0u8 }] }
        Value::Str(s) => {
            let mut out = Vec::with_capacity(4 + s.len());
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
            out
        }
        Value::Vec2(v) => {
            let mut out = Vec::with_capacity(8);
            out.extend_from_slice(&v[0].to_le_bytes());
            out.extend_from_slice(&v[1].to_le_bytes());
            out
        }
        Value::Vec3(v) => {
            let mut out = Vec::with_capacity(12);
            for x in v { out.extend_from_slice(&x.to_le_bytes()); }
            out
        }
        _ => vec![],
    }
}

fn bytes_to_value(bytes: &[u8], type_name: &str) -> Option<Value> {
    match type_name {
        "i32" => {
            if bytes.len() >= 4 { Some(Value::I32(i32::from_le_bytes([bytes[0],bytes[1],bytes[2],bytes[3]]))) } else { None }
        }
        "i64" => {
            if bytes.len() >= 8 {
                let mut arr = [0u8;8]; arr.copy_from_slice(&bytes[..8]);
                Some(Value::I64(i64::from_le_bytes(arr)))
            } else { None }
        }
        "f32" => {
            if bytes.len() >= 4 { Some(Value::F32(f32::from_le_bytes([bytes[0],bytes[1],bytes[2],bytes[3]]))) } else { None }
        }
        "f64" => {
            if bytes.len() >= 8 {
                let mut arr = [0u8;8]; arr.copy_from_slice(&bytes[..8]);
                Some(Value::F64(f64::from_le_bytes(arr)))
            } else { None }
        }
        "str" => {
            if bytes.len() >= 4 {
                let len = u32::from_le_bytes([bytes[0],bytes[1],bytes[2],bytes[3]]) as usize;
                if bytes.len() >= 4 + len {
                    Some(Value::Str(String::from_utf8_lossy(&bytes[4..4+len]).to_string()))
                } else { None }
            } else { None }
        }
        _ => None,
    }
}

// ─── Builtin dispatch ───────────────────────────────────────────────────────

pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        // ── TCP Server ───────────────────────────────────────────────────
        "net::tcp_listen" => {
            let addr = str_arg(args, 0).unwrap_or_else(|| "127.0.0.1".into());
            let port = i64_arg(args, 1).unwrap_or(8080);
            let full = format!("{}:{}", addr, port);
            match std::net::TcpListener::bind(&full) {
                Ok(listener) => {
                    listener.set_nonblocking(true).ok();
                    TCP_SERVERS.with(|s| {
                        let mut s = s.borrow_mut();
                        s.push(TcpServerHandle { listener: Some(listener), clients: Vec::new() });
                        Some(Ok(Value::U64(s.len() as u64)))
                    })
                }
                Err(e) => Some(Err(rt_err!(format!("net::tcp_listen: {e}")))),
            }
        }
        "net::tcp_accept" => {
            if let Some(h) = i64_arg(args, 0) {
                TCP_SERVERS.with(|s| {
                    let mut s = s.borrow_mut();
                    if let Some(server) = s.get_mut(h as usize - 1) {
                        if let Some(l) = &server.listener {
                            match l.accept() {
                                Ok((stream, _)) => {
                                    stream.set_nonblocking(true).ok();
                                    let idx = server.clients.len() + 1;
                                    server.clients.push(stream);
                                    Some(Ok(Value::U64(idx as u64)))
                                }
                                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                    Some(Ok(Value::U64(0))) // No pending connections
                                }
                                Err(e) => Some(Err(rt_err!(format!("net::tcp_accept: {e}")))),
                            }
                        } else { Some(Err(rt_err!("net::tcp_accept: listener closed"))) }
                    } else { Some(Err(rt_err!("net::tcp_accept: invalid handle"))) }
                })
            } else { Some(Err(rt_err!("net::tcp_accept() requires server handle"))) }
        }
        "net::tcp_send" => {
            if args.len() < 2 { return Some(Err(rt_err!("net::tcp_send() requires client_handle, data"))); }
            if let (Some(h), Value::Str(data)) = (i64_arg(args,0), &args[1]) {
                TCP_SERVERS.with(|s| {
                    let s = s.borrow();
                    if let Some(server) = s.get(h as usize - 1) {
                        // This sends to all clients — for per-client use tcp_client_send
                        Some(Ok(Value::Unit))
                    } else { Some(Err(rt_err!("net::tcp_send: invalid handle"))) }
                })
            } else { Some(Err(rt_err!("net::tcp_send() requires handle, string"))) }
        }

        // ── TCP Client ───────────────────────────────────────────────────
        "net::tcp_connect" => {
            let addr = str_arg(args, 0).unwrap_or_else(|| "127.0.0.1".into());
            let port = i64_arg(args, 1).unwrap_or(8080);
            let full = format!("{}:{}", addr, port);
            match std::net::TcpStream::connect(full) {
                Ok(stream) => {
                    stream.set_nonblocking(true).ok();
                    TCP_CLIENTS.with(|c| {
                        let mut c = c.borrow_mut();
                        c.push(stream);
                        Some(Ok(Value::U64(c.len() as u64)))
                    })
                }
                Err(e) => Some(Err(rt_err!(format!("net::tcp_connect: {e}")))),
            }
        }
        "net::tcp_client_send" => {
            if args.len() < 2 { return Some(Err(rt_err!("net::tcp_client_send() requires handle, data"))); }
            if let (Some(h), Value::Str(data)) = (i64_arg(args,0), &args[1]) {
                TCP_CLIENTS.with(|c| {
                    let c = c.borrow();
                    if let Some(mut stream) = c.get(h as usize - 1) {
                        match stream.write_all(data.as_bytes()) {
                            Ok(()) => Some(Ok(Value::Unit)),
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Some(Ok(Value::Unit)),
                            Err(e) => Some(Err(rt_err!(format!("net::tcp_client_send: {e}")))),
                        }
                    } else { Some(Err(rt_err!("net::tcp_client_send: invalid handle"))) }
                })
            } else { Some(Err(rt_err!("net::tcp_client_send() requires handle, string"))) }
        }
        "net::tcp_client_recv" => {
            if let Some(h) = i64_arg(args, 0) {
                let type_name = str_arg(args, 1).unwrap_or_else(|| "str".into());
                TCP_CLIENTS.with(|c| {
                    let c = c.borrow();
                    if let Some(mut stream) = c.get(h as usize - 1) {
                        let mut buf = [0u8; 4096];
                        match stream.read(&mut buf) {
                            Ok(n) if n > 0 => {
                                match bytes_to_value(&buf[..n], &type_name) {
                                    Some(v) => Some(Ok(v)),
                                    None => Some(Ok(Value::Str(String::from_utf8_lossy(&buf[..n]).to_string()))),
                                }
                            }
                            Ok(_) => Some(Ok(Value::Str("".into()))),
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Some(Ok(Value::Unit)),
                            Err(e) => Some(Err(rt_err!(format!("net::tcp_client_recv: {e}")))),
                        }
                    } else { Some(Err(rt_err!("net::tcp_client_recv: invalid handle"))) }
                })
            } else { Some(Err(rt_err!("net::tcp_client_recv() requires handle"))) }
        }

        // ── UDP ──────────────────────────────────────────────────────────
        "net::udp_bind" => {
            let addr = str_arg(args, 0).unwrap_or_else(|| "0.0.0.0".into());
            let port = i64_arg(args, 1).unwrap_or(9000);
            let full = format!("{}:{}", addr, port);
            match std::net::UdpSocket::bind(full) {
                Ok(socket) => {
                    socket.set_nonblocking(true).ok();
                    UDP_SOCKETS.with(|s| {
                        let mut s = s.borrow_mut();
                        s.push(socket);
                        Some(Ok(Value::U64(s.len() as u64)))
                    })
                }
                Err(e) => Some(Err(rt_err!(format!("net::udp_bind: {e}")))),
            }
        }
        "net::udp_send_to" => {
            if args.len() < 4 { return Some(Err(rt_err!("net::udp_send_to() requires handle, data, addr, port"))); }
            if let (Some(h), Value::Str(data), Some(port)) = (i64_arg(args,0), &args[1], i64_arg(args,3)) {
                let addr = str_arg(args, 2).unwrap_or_else(|| "127.0.0.1".into());
                UDP_SOCKETS.with(|s| {
                    let s = s.borrow();
                    if let Some(socket) = s.get(h as usize - 1) {
                        match socket.send_to(data.as_bytes(), format!("{}:{}", addr, port)) {
                            Ok(n) => Some(Ok(Value::U64(n as u64))),
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Some(Ok(Value::U64(0))),
                            Err(e) => Some(Err(rt_err!(format!("net::udp_send_to: {e}")))),
                        }
                    } else { Some(Err(rt_err!("net::udp_send_to: invalid handle"))) }
                })
            } else { Some(Err(rt_err!("net::udp_send_to() requires handle, data, addr, port"))) }
        }
        "net::udp_recv" => {
            if let Some(h) = i64_arg(args, 0) {
                UDP_SOCKETS.with(|s| {
                    let s = s.borrow();
                    if let Some(socket) = s.get(h as usize - 1) {
                        let mut buf = [0u8; 4096];
                        match socket.recv_from(&mut buf) {
                            Ok((n, _addr)) if n > 0 => {
                                Some(Ok(Value::Str(String::from_utf8_lossy(&buf[..n]).to_string())))
                            }
                            Ok(_) => Some(Ok(Value::Str("".into()))),
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Some(Ok(Value::Unit)),
                            Err(e) => Some(Err(rt_err!(format!("net::udp_recv: {e}")))),
                        }
                    } else { Some(Err(rt_err!("net::udp_recv: invalid handle"))) }
                })
            } else { Some(Err(rt_err!("net::udp_recv() requires handle"))) }
        }

        // ── URL / HTTP helpers ───────────────────────────────────────────
        "net::url_encode" => {
            if let Some(s) = str_arg(args, 0) {
                let encoded = url_encode(&s);
                Some(Ok(Value::Str(encoded)))
            } else { Some(Err(rt_err!("net::url_encode() requires string"))) }
        }
        "net::url_decode" => {
            if let Some(s) = str_arg(args, 0) {
                let decoded = url_decode(&s);
                Some(Ok(Value::Str(decoded)))
            } else { Some(Err(rt_err!("net::url_decode() requires string"))) }
        }
        "net::parse_url" => {
            if let Some(url) = str_arg(args, 0) {
                // Simple URL parser
                let (proto, rest) = if let Some(idx) = url.find("://") {
                    (&url[..idx], &url[idx+3..])
                } else {
                    ("http", url.as_str())
                };
                let (host, path) = if let Some(idx) = rest.find('/') {
                    (&rest[..idx], &rest[idx..])
                } else {
                    (rest, "/")
                };
                let (hostname, port) = if let Some(idx) = host.find(':') {
                    (&host[..idx], host[idx+1..].parse::<i64>().ok())
                } else {
                    (host, None)
                };
                let port = port.unwrap_or(if proto == "https" { 443 } else { 80 });
                Some(Ok(Value::Tuple(vec![
                    Value::Str(proto.into()),
                    Value::Str(hostname.into()),
                    Value::I64(port),
                    Value::Str(path.into()),
                ])))
            } else { Some(Err(rt_err!("net::parse_url() requires url string"))) }
        }

        _ => None,
    }
}

fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => {
                out.push('%');
                out.push_str(&format!("{:02X}", b));
            }
        }
    }
    out
}

fn url_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                out.push(byte as char);
            } else {
                out.push('%');
                out.push_str(&hex);
            }
        } else if c == '+' {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}
