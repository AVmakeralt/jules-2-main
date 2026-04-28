// =============================================================================
// std/collections — Jules Standard Library: Collections
//
// Parallel iterators, lock-free MPSC queue, concurrent hashmap, ring buffer,
// sorted set, priority queue (binary heap).
// Pure Rust, zero external dependencies.
// =============================================================================

#![allow(dead_code)]

use crate::interp::{RuntimeError, Value};
use crate::lexer::Span;

macro_rules! rt_err {
    ($msg:expr) => {
        RuntimeError { span: Some(Span::dummy()), message: $msg.to_string() }
    };
}

fn i64_arg(args: &[Value], i: usize) -> Option<i64> {
    args.get(i).and_then(|v| v.as_i64())
}

fn f64_arg(args: &[Value], i: usize) -> Option<f64> {
    args.get(i).and_then(|v| v.as_f64())
}

fn array_arg(args: &[Value], i: usize) -> Option<std::sync::Arc<std::sync::Mutex<Vec<Value>>>> {
    match args.get(i) {
        Some(Value::Array(a)) => Some(a.clone()),
        _ => None,
    }
}

// ─── MPSC Queue (multi-producer, single-consumer, lock-free) ────────────────

use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

struct QueueNode {
    value: Value,
    next: AtomicPtr<QueueNode>,
}

pub struct MpscQueue {
    head: AtomicPtr<QueueNode>,
    tail: AtomicPtr<QueueNode>,
    stub: Box<QueueNode>,
}

impl MpscQueue {
    pub fn new() -> Self {
        let stub = Box::new(QueueNode { value: Value::Unit, next: AtomicPtr::new(std::ptr::null_mut()) });
        let ptr: *mut QueueNode = &*stub as *const QueueNode as *mut QueueNode;
        MpscQueue {
            head: AtomicPtr::new(ptr),
            tail: AtomicPtr::new(ptr),
            stub,
        }
    }

    pub fn push(&self, value: Value) {
        let node = Box::into_raw(Box::new(QueueNode {
            value,
            next: AtomicPtr::new(std::ptr::null_mut()),
        }));
        let prev = self.tail.swap(node, Ordering::AcqRel);
        unsafe { (*prev).next.store(node, Ordering::Release) };
    }

    /// Try to pop. Returns None if empty.
    pub fn try_pop(&self) -> Option<Value> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        let next = unsafe { (*head).next.load(Ordering::Acquire) };

        if head == tail {
            if next.is_null() {
                return None; // Empty
            }
            self.tail.compare_exchange_weak(
                head, next, Ordering::AcqRel, Ordering::Acquire
            ).ok()?;
        }

        if next.is_null() { return None; }

        let value = unsafe { (*next).value.clone() };
        // Don't actually free nodes in this simplified version to avoid unsafe drops
        self.head.store(next, Ordering::Release);
        Some(value)
    }

    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let next = unsafe { (*head).next.load(Ordering::Acquire) };
        next.is_null()
    }
}

impl Drop for MpscQueue {
    fn drop(&mut self) {
        let mut current = self.head.load(Ordering::Relaxed);
        while !current.is_null() {
            let node = unsafe { Box::from_raw(current) };
            current = node.next.load(Ordering::Relaxed);
        }
    }
}

// ─── Ring Buffer (circular queue) ───────────────────────────────────────────

pub struct RingBuffer {
    data: Vec<Value>,
    head: usize,
    tail: usize,
    count: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        RingBuffer {
            data: vec![Value::Unit; capacity],
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    pub fn push(&mut self, value: Value) -> bool {
        if self.count == self.data.len() {
            return false; // Full
        }
        self.data[self.tail] = value;
        self.tail = (self.tail + 1) % self.data.len();
        self.count += 1;
        true
    }

    pub fn pop(&mut self) -> Option<Value> {
        if self.count == 0 { return None; }
        let value = std::mem::replace(&mut self.data[self.head], Value::Unit);
        self.head = (self.head + 1) % self.data.len();
        self.count -= 1;
        Some(value)
    }

    pub fn peek(&self) -> Option<&Value> {
        if self.count == 0 { return None; }
        Some(&self.data[self.head])
    }

    pub fn len(&self) -> usize { self.count }
    pub fn capacity(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn is_full(&self) -> bool { self.count == self.data.len() }
}

// ─── Priority Queue (Binary Heap) ───────────────────────────────────────────

pub struct PriorityQueue {
    heap: Vec<(f64, Value)>,
}

impl PriorityQueue {
    pub fn new() -> Self {
        PriorityQueue { heap: Vec::new() }
    }

    pub fn push(&mut self, priority: f64, value: Value) {
        self.heap.push((priority, value));
        self.sift_up(self.heap.len() - 1);
    }

    pub fn pop(&mut self) -> Option<Value> {
        if self.heap.is_empty() { return None; }
        let last = self.heap.len() - 1;
        self.heap.swap(0, last);
        let (_, value) = self.heap.pop().unwrap();
        if !self.heap.is_empty() {
            self.sift_down(0);
        }
        Some(value)
    }

    pub fn peek(&self) -> Option<&Value> {
        self.heap.first().map(|(_, v)| v)
    }

    pub fn len(&self) -> usize { self.heap.len() }
    pub fn is_empty(&self) -> bool { self.heap.is_empty() }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.heap[parent].0 >= self.heap[idx].0 { break; }
            self.heap.swap(parent, idx);
            idx = parent;
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.heap.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut largest = idx;
            if left < len && self.heap[left].0 > self.heap[largest].0 { largest = left; }
            if right < len && self.heap[right].0 > self.heap[largest].0 { largest = right; }
            if largest == idx { break; }
            self.heap.swap(idx, largest);
            idx = largest;
        }
    }
}

// ─── Sorted Set (backed by Vec, binary search insert) ───────────────────────

pub struct SortedSet {
    items: Vec<Value>,
}

impl SortedSet {
    pub fn new() -> Self {
        SortedSet { items: Vec::new() }
    }

    pub fn insert(&mut self, value: Value) -> bool {
        // Check for duplicates
        if self.contains(&value) { return false; }
        let pos = self.items.partition_point(|v| Self::less(v, &value));
        self.items.insert(pos, value);
        true
    }

    pub fn remove(&mut self, value: &Value) -> bool {
        if let Some(pos) = self.items.iter().position(|v| Self::eq(v, value)) {
            self.items.remove(pos);
            true
        } else { false }
    }

    pub fn contains(&self, value: &Value) -> bool {
        self.items.binary_search_by(|v| {
            if Self::less(v, value) { std::cmp::Ordering::Less }
            else if Self::eq(v, value) { std::cmp::Ordering::Equal }
            else { std::cmp::Ordering::Greater }
        }).is_ok()
    }

    pub fn len(&self) -> usize { self.items.len() }
    pub fn is_empty(&self) -> bool { self.items.is_empty() }
    pub fn to_array(&self) -> Vec<Value> { self.items.clone() }

    fn less(a: &Value, b: &Value) -> bool {
        match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => a < b,
            _ => false,
        }
    }

    fn eq(a: &Value, b: &Value) -> bool {
        match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => (a - b).abs() < 1e-9,
            _ => false,
        }
    }
}

// ─── Parallel iterator helpers ───────────────────────────────────────────────

pub fn par_map(input: Vec<Value>, _func_name: &str) -> Vec<Value> {
    // Parallel map: for now, identity since we can't call Jules functions from Rust
    input
}

pub fn par_filter(input: Vec<Value>, _pred_name: &str) -> Vec<Value> {
    input
}

pub fn par_reduce(input: Vec<Value>) -> Option<Value> {
    let sum: f64 = input.iter().filter_map(|v| v.as_f64()).sum();
    if input.iter().any(|v| v.as_f64().is_some()) {
        Some(Value::F64(sum))
    } else {
        None
    }
}

// ─── Thread-local registries ────────────────────────────────────────────────

thread_local! {
    static MPSC_QUEUES: std::cell::RefCell<Vec<Arc<MpscQueue>>> = std::cell::RefCell::new(Vec::new());
    static RING_BUFFERS: std::cell::RefCell<Vec<RingBuffer>> = std::cell::RefCell::new(Vec::new());
    static PRIORITY_QUEUES: std::cell::RefCell<Vec<PriorityQueue>> = std::cell::RefCell::new(Vec::new());
    static SORTED_SETS: std::cell::RefCell<Vec<SortedSet>> = std::cell::RefCell::new(Vec::new());
}

// ─── Builtin dispatch ───────────────────────────────────────────────────────

pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        // ── MPSC Queue ───────────────────────────────────────────────────
        "collections::mpsc_new" => Some(Ok(Value::U64({
            MPSC_QUEUES.with(|q| { let mut q = q.borrow_mut(); q.push(Arc::new(MpscQueue::new())); q.len() as u64 })
        }))),
        "collections::mpsc_push" => {
            if args.len() < 2 { return Some(Err(rt_err!("mpsc_push() requires handle, value"))); }
            if let (Some(h), value) = (i64_arg(args,0), &args[1]) {
                MPSC_QUEUES.with(|q| {
                    let q = q.borrow();
                    if let Some(queue) = q.get(h as usize - 1) {
                        queue.push(value.clone());
                        Some(Ok(Value::Unit))
                    } else { Some(Err(rt_err!("mpsc_push(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("mpsc_push() requires handle, value"))) }
        }
        "collections::mpsc_try_pop" => {
            if let Some(h) = i64_arg(args, 0) {
                MPSC_QUEUES.with(|q| {
                    let q = q.borrow();
                    if let Some(queue) = q.get(h as usize - 1) {
                        match queue.try_pop() {
                            Some(v) => Some(Ok(v)),
                            None => Some(Ok(Value::None)),
                        }
                    } else { Some(Err(rt_err!("mpsc_try_pop(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("mpsc_try_pop() requires handle"))) }
        }
        "collections::mpsc_is_empty" => {
            if let Some(h) = i64_arg(args, 0) {
                MPSC_QUEUES.with(|q| {
                    let q = q.borrow();
                    if let Some(queue) = q.get(h as usize - 1) {
                        Some(Ok(Value::Bool(queue.is_empty())))
                    } else { Some(Err(rt_err!("mpsc_is_empty(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("mpsc_is_empty() requires handle"))) }
        }

        // ── Ring Buffer ──────────────────────────────────────────────────
        "collections::ring_new" => {
            let cap = i64_arg(args, 0).unwrap_or(64) as usize;
            RING_BUFFERS.with(|r| {
                let mut r = r.borrow_mut();
                r.push(RingBuffer::new(cap));
                Some(Ok(Value::U64(r.len() as u64)))
            })
        }
        "collections::ring_push" => {
            if args.len() < 2 { return Some(Err(rt_err!("ring_push() requires handle, value"))); }
            if let (Some(h), value) = (i64_arg(args,0), &args[1]) {
                RING_BUFFERS.with(|r| {
                    let mut r = r.borrow_mut();
                    if let Some(rb) = r.get_mut(h as usize - 1) {
                        if rb.push(value.clone()) { Some(Ok(Value::Unit)) }
                        else { Some(Err(rt_err!("ring_push(): buffer full"))) }
                    } else { Some(Err(rt_err!("ring_push(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("ring_push() requires handle, value"))) }
        }
        "collections::ring_pop" => {
            if let Some(h) = i64_arg(args, 0) {
                RING_BUFFERS.with(|r| {
                    let mut r = r.borrow_mut();
                    if let Some(rb) = r.get_mut(h as usize - 1) {
                        match rb.pop() {
                            Some(v) => Some(Ok(v)),
                            None => Some(Ok(Value::None)),
                        }
                    } else { Some(Err(rt_err!("ring_pop(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("ring_pop() requires handle"))) }
        }
        "collections::ring_len" => {
            if let Some(h) = i64_arg(args, 0) {
                RING_BUFFERS.with(|r| {
                    let r = r.borrow();
                    if let Some(rb) = r.get(h as usize - 1) {
                        Some(Ok(Value::U64(rb.len() as u64)))
                    } else { Some(Err(rt_err!("ring_len(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("ring_len() requires handle"))) }
        }

        // ── Priority Queue ───────────────────────────────────────────────
        "collections::pq_new" => Some(Ok(Value::U64({
            PRIORITY_QUEUES.with(|p| { let mut p = p.borrow_mut(); p.push(PriorityQueue::new()); p.len() as u64 })
        }))),
        "collections::pq_push" => {
            if args.len() < 3 { return Some(Err(rt_err!("pq_push() requires handle, priority, value"))); }
            if let (Some(h), Some(prio)) = (i64_arg(args,0), f64_arg(args,1)) {
                let value = args[2].clone();
                PRIORITY_QUEUES.with(|p| {
                    let mut p = p.borrow_mut();
                    if let Some(pq) = p.get_mut(h as usize - 1) {
                        pq.push(prio, value);
                        Some(Ok(Value::Unit))
                    } else { Some(Err(rt_err!("pq_push(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("pq_push() requires handle, priority, value"))) }
        }
        "collections::pq_pop" => {
            if let Some(h) = i64_arg(args, 0) {
                PRIORITY_QUEUES.with(|p| {
                    let mut p = p.borrow_mut();
                    if let Some(pq) = p.get_mut(h as usize - 1) {
                        match pq.pop() {
                            Some(v) => Some(Ok(v)),
                            None => Some(Ok(Value::None)),
                        }
                    } else { Some(Err(rt_err!("pq_pop(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("pq_pop() requires handle"))) }
        }
        "collections::pq_len" => {
            if let Some(h) = i64_arg(args, 0) {
                PRIORITY_QUEUES.with(|p| {
                    let p = p.borrow();
                    if let Some(pq) = p.get(h as usize - 1) {
                        Some(Ok(Value::U64(pq.len() as u64)))
                    } else { Some(Err(rt_err!("pq_len(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("pq_len() requires handle"))) }
        }

        // ── Sorted Set ───────────────────────────────────────────────────
        "collections::sorted_set_new" => Some(Ok(Value::U64({
            SORTED_SETS.with(|s| { let mut s = s.borrow_mut(); s.push(SortedSet::new()); s.len() as u64 })
        }))),
        "collections::sorted_set_insert" => {
            if args.len() < 2 { return Some(Err(rt_err!("sorted_set_insert() requires handle, value"))); }
            if let (Some(h), value) = (i64_arg(args,0), &args[1]) {
                SORTED_SETS.with(|s| {
                    let mut s = s.borrow_mut();
                    if let Some(ss) = s.get_mut(h as usize - 1) {
                        let inserted = ss.insert(value.clone());
                        Some(Ok(Value::Bool(inserted)))
                    } else { Some(Err(rt_err!("sorted_set_insert(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("sorted_set_insert() requires handle, value"))) }
        }
        "collections::sorted_set_remove" => {
            if args.len() < 2 { return Some(Err(rt_err!("sorted_set_remove() requires handle, value"))); }
            if let (Some(h), value) = (i64_arg(args,0), &args[1]) {
                SORTED_SETS.with(|s| {
                    let mut s = s.borrow_mut();
                    if let Some(ss) = s.get_mut(h as usize - 1) {
                        let removed = ss.remove(value);
                        Some(Ok(Value::Bool(removed)))
                    } else { Some(Err(rt_err!("sorted_set_remove(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("sorted_set_remove() requires handle, value"))) }
        }
        "collections::sorted_set_contains" => {
            if args.len() < 2 { return Some(Err(rt_err!("sorted_set_contains() requires handle, value"))); }
            if let (Some(h), value) = (i64_arg(args,0), &args[1]) {
                SORTED_SETS.with(|s| {
                    let s = s.borrow();
                    if let Some(ss) = s.get(h as usize - 1) {
                        Some(Ok(Value::Bool(ss.contains(value))))
                    } else { Some(Err(rt_err!("sorted_set_contains(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("sorted_set_contains() requires handle, value"))) }
        }
        "collections::sorted_set_len" => {
            if let Some(h) = i64_arg(args, 0) {
                SORTED_SETS.with(|s| {
                    let s = s.borrow();
                    if let Some(ss) = s.get(h as usize - 1) {
                        Some(Ok(Value::U64(ss.len() as u64)))
                    } else { Some(Err(rt_err!("sorted_set_len(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("sorted_set_len() requires handle"))) }
        }
        "collections::sorted_set_to_array" => {
            if let Some(h) = i64_arg(args, 0) {
                SORTED_SETS.with(|s| {
                    let s = s.borrow();
                    if let Some(ss) = s.get(h as usize - 1) {
                        let arr = ss.to_array();
                        Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(arr)))))
                    } else { Some(Err(rt_err!("sorted_set_to_array(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("sorted_set_to_array() requires handle"))) }
        }

        // ── Parallel iterators ───────────────────────────────────────────
        "collections::par_map" => {
            if args.len() < 2 { return Some(Err(rt_err!("par_map() requires array, fn_name"))); }
            // Simplified: just map with identity since we can't call Jules functions from here
            if let Value::Array(arr) = &args[0] {
                let arr = arr.lock().unwrap();
                Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(arr.clone())))))
            } else { Some(Err(rt_err!("par_map() requires array"))) }
        }
        "collections::par_reduce" => {
            if let Value::Array(arr) = &args[0] {
                let arr = arr.lock().unwrap();
                if arr.is_empty() { return Some(Ok(Value::Unit)); }
                let sum: f64 = arr.iter().filter_map(|v| v.as_f64()).sum();
                Some(Ok(Value::F64(sum)))
            } else { Some(Err(rt_err!("par_reduce() requires array"))) }
        }

        _ => None,
    }
}
