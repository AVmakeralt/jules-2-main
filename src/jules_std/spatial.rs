// =============================================================================
// std/spatial — Jules Standard Library: Spatial Partitioning
//
// SpatialHash, UniformGrid, Quadtree (2D), Octree (3D), BVH.
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

fn f64_arg(args: &[Value], i: usize) -> Option<f64> {
    args.get(i).and_then(|v| v.as_f64())
}

fn i64_arg(args: &[Value], i: usize) -> Option<i64> {
    args.get(i).and_then(|v| v.as_i64())
}

// ─── Spatial Hash ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct SpatialHash {
    cell_size: f32,
    cells: std::collections::HashMap<(i32, i32, i32), Vec<u64>>,
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        SpatialHash {
            cell_size,
            cells: std::collections::HashMap::default(),
        }
    }

    fn key(x: f32, y: f32, z: f32, cell_size: f32) -> (i32, i32, i32) {
        ((x / cell_size).floor() as i32,
         (y / cell_size).floor() as i32,
         (z / cell_size).floor() as i32)
    }

    pub fn insert(&mut self, id: u64, x: f32, y: f32, z: f32) {
        let k = Self::key(x, y, z, self.cell_size);
        self.cells.entry(k).or_default().push(id);
    }

    pub fn query(&self, x: f32, y: f32, z: f32, radius: f32) -> Vec<u64> {
        let cs = self.cell_size;
        let r = radius + cs;
        let min_k = Self::key(x - r, y - r, z - r, cs);
        let max_k = Self::key(x + r, y + r, z + r, cs);
        let mut results = Vec::new();
        for cx in min_k.0..=max_k.0 {
            for cy in min_k.1..=max_k.1 {
                for cz in min_k.2..=max_k.2 {
                    if let Some(ids) = self.cells.get(&(cx, cy, cz)) {
                        results.extend_from_slice(ids);
                    }
                }
            }
        }
        results
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }
}

// ─── Quadtree (2D) ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Quadtree {
    bounds: [f32; 4],  // [x, y, w, h]
    max_depth: u32,
    max_items: usize,
    items: Vec<(u64, f32, f32)>,
    children: Option<Box<[Quadtree; 4]>>,
}

impl Quadtree {
    pub fn new(x: f32, y: f32, w: f32, h: f32, max_depth: u32, max_items: usize) -> Self {
        Quadtree {
            bounds: [x, y, w, h],
            max_depth,
            max_items,
            items: Vec::new(),
            children: None,
        }
    }

    fn split(&mut self) {
        let [x, y, w, h] = self.bounds;
        let hw = w * 0.5;
        let hh = h * 0.5;
        self.children = Some(Box::new([
            Quadtree::new(x, y, hw, hh, self.max_depth - 1, self.max_items),
            Quadtree::new(x + hw, y, hw, hh, self.max_depth - 1, self.max_items),
            Quadtree::new(x, y + hh, hw, hh, self.max_depth - 1, self.max_items),
            Quadtree::new(x + hw, y + hh, hw, hh, self.max_depth - 1, self.max_items),
        ]));
    }

    pub fn insert(&mut self, id: u64, px: f32, py: f32) {
        if self.children.is_none() && self.items.len() >= self.max_items && self.max_depth > 0 {
            self.split();
            let old_items: Vec<_> = self.items.drain(..).collect();
            for (id, x, y) in old_items {
                self.insert(id, x, y);
            }
        }
        if let Some(children) = &mut self.children {
            let hw = self.bounds[2] * 0.5;
            let hh = self.bounds[3] * 0.5;
            let cx = self.bounds[0] + hw;
            let cy = self.bounds[1] + hh;
            let idx = if px >= cx { 1 } else { 0 } | if py >= cy { 2 } else { 0 };
            children[idx].insert(id, px, py);
        } else {
            self.items.push((id, px, py));
        }
    }

    pub fn query(&self, x: f32, y: f32, radius: f32) -> Vec<u64> {
        let mut results = Vec::new();
        let [bx, by, bw, bh] = self.bounds;
        // Check overlap
        if x + radius < bx || x - radius > bx + bw || y + radius < by || y - radius > by + bh {
            return results;
        }
        if let Some(children) = &self.children {
            for child in children.iter() {
                results.extend(child.query(x, y, radius));
            }
        }
        for &(id, px, py) in &self.items {
            if (px - x).hypot(py - y) <= radius {
                results.push(id);
            }
        }
        results
    }
}

// ─── Octree (3D) ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Octree {
    center: [f32; 3],
    half_size: f32,
    max_depth: u32,
    max_items: usize,
    items: Vec<(u64, f32, f32, f32)>,
    children: Option<Box<[Octree; 8]>>,
}

impl Octree {
    pub fn new(cx: f32, cy: f32, cz: f32, half: f32, max_depth: u32, max_items: usize) -> Self {
        Octree {
            center: [cx, cy, cz],
            half_size: half,
            max_depth,
            max_items,
            items: Vec::new(),
            children: None,
        }
    }

    fn split(&mut self) {
        let h = self.half_size * 0.5;
        let c = self.center;
        self.children = Some(Box::new([
            Octree::new(c[0]-h, c[1]-h, c[2]-h, h, self.max_depth-1, self.max_items),
            Octree::new(c[0]+h, c[1]-h, c[2]-h, h, self.max_depth-1, self.max_items),
            Octree::new(c[0]-h, c[1]+h, c[2]-h, h, self.max_depth-1, self.max_items),
            Octree::new(c[0]+h, c[1]+h, c[2]-h, h, self.max_depth-1, self.max_items),
            Octree::new(c[0]-h, c[1]-h, c[2]+h, h, self.max_depth-1, self.max_items),
            Octree::new(c[0]+h, c[1]-h, c[2]+h, h, self.max_depth-1, self.max_items),
            Octree::new(c[0]-h, c[1]+h, c[2]+h, h, self.max_depth-1, self.max_items),
            Octree::new(c[0]+h, c[1]+h, c[2]+h, h, self.max_depth-1, self.max_items),
        ]));
    }

    fn contains(&self, x: f32, y: f32, z: f32) -> bool {
        let h = self.half_size;
        x >= self.center[0]-h && x <= self.center[0]+h &&
        y >= self.center[1]-h && y <= self.center[1]+h &&
        z >= self.center[2]-h && z <= self.center[2]+h
    }

    pub fn insert(&mut self, id: u64, x: f32, y: f32, z: f32) {
        if !self.contains(x, y, z) { return; }
        if self.children.is_none() && self.items.len() >= self.max_items && self.max_depth > 0 {
            self.split();
            let old_items: Vec<_> = self.items.drain(..).collect();
            for (id, px, py, pz) in old_items {
                self.insert(id, px, py, pz);
            }
        }
        if let Some(children) = &mut self.children {
            let idx = if x >= self.center[0] { 1 } else { 0 }
                    | if y >= self.center[1] { 2 } else { 0 }
                    | if z >= self.center[2] { 4 } else { 0 };
            children[idx].insert(id, x, y, z);
        } else {
            self.items.push((id, x, y, z));
        }
    }

    pub fn query(&self, x: f32, y: f32, z: f32, radius: f32) -> Vec<u64> {
        let mut results = Vec::new();
        let h = self.half_size;
        if x + radius < self.center[0]-h || x - radius > self.center[0]+h ||
           y + radius < self.center[1]-h || y - radius > self.center[1]+h ||
           z + radius < self.center[2]-h || z - radius > self.center[2]+h {
            return results;
        }
        if let Some(children) = &self.children {
            for child in children.iter() {
                results.extend(child.query(x, y, z, radius));
            }
        }
        for &(id, px, py, pz) in &self.items {
            if (px-x).hypot(py-y).hypot(pz-z) <= radius {
                results.push(id);
            }
        }
        results
    }
}

// ─── BVH (Bounding Volume Hierarchy) ────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BvhNode {
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub left: Option<usize>,
    pub right: Option<usize>,
    /// If leaf: contains the item id. If internal: None.
    pub item_id: Option<u64>,
}

pub struct Bvh {
    pub nodes: Vec<BvhNode>,
}

impl Bvh {
    pub fn new() -> Self {
        Bvh { nodes: Vec::new() }
    }

    pub fn build(&mut self, items: &[(u64, [f32;3], [f32;3])]) {
        self.nodes.clear();
        if items.is_empty() { return; }
        self.build_recursive(items, 0);
    }

    fn build_recursive(&mut self, items: &[(u64, [f32;3], [f32;3])], depth: usize) -> usize {
        // Compute bounds
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for (_, a_min, a_max) in items {
            for i in 0..3 {
                if a_min[i] < min[i] { min[i] = a_min[i]; }
                if a_max[i] > max[i] { max[i] = a_max[i]; }
            }
        }

        // Leaf: single item
        if items.len() == 1 {
            let idx = self.nodes.len();
            self.nodes.push(BvhNode {
                bounds_min: items[0].1,
                bounds_max: items[0].2,
                left: None, right: None,
                item_id: Some(items[0].0),
            });
            return idx;
        }

        // Find split axis (longest dimension)
        let extent = [max[0]-min[0], max[1]-min[1], max[2]-min[2]];
        let axis = extent.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

        // Sort by centroid
        let mut items: Vec<_> = items.to_vec();
        items.sort_by(|a, b| {
            let ca = (a.1[axis] + a.2[axis]) * 0.5;
            let cb = (b.1[axis] + b.2[axis]) * 0.5;
            ca.partial_cmp(&cb).unwrap()
        });

        let mid = items.len() / 2;
        let (left_items, right_items) = items.split_at(mid);

        let left_idx = self.build_recursive(left_items, depth+1);
        let right_idx = self.build_recursive(right_items, depth+1);

        let idx = self.nodes.len();
        self.nodes.push(BvhNode {
            bounds_min: min, bounds_max: max,
            left: Some(left_idx), right: Some(right_idx),
            item_id: None,
        });
        idx
    }

    pub fn query(&self, q_min: [f32;3], q_max: [f32;3]) -> Vec<u64> {
        let mut results = Vec::new();
        if let Some(root) = self.nodes.first() {
            self.query_recursive(0, q_min, q_max, &mut results);
        }
        results
    }

    fn query_recursive(&self, node_idx: usize, q_min: [f32;3], q_max: [f32;3], results: &mut Vec<u64>) {
        let node = &self.nodes[node_idx];
        // AABB overlap test
        if node.bounds_min[0] > q_max[0] || node.bounds_max[0] < q_min[0] ||
           node.bounds_min[1] > q_max[1] || node.bounds_max[1] < q_min[1] ||
           node.bounds_min[2] > q_max[2] || node.bounds_max[2] < q_min[2] {
            return;
        }
        if let Some(item_id) = node.item_id {
            results.push(item_id);
        } else {
            if let Some(left) = node.left { self.query_recursive(left, q_min, q_max, results); }
            if let Some(right) = node.right { self.query_recursive(right, q_min, q_max, results); }
        }
    }
}

// ─── Builtin dispatch ───────────────────────────────────────────────────────

thread_local! {
    static SPATIAL_HASHES: std::cell::RefCell<Vec<SpatialHash>> = std::cell::RefCell::new(Vec::new());
    static QUADTREES: std::cell::RefCell<Vec<Quadtree>> = std::cell::RefCell::new(Vec::new());
    static OCTREES: std::cell::RefCell<Vec<Octree>> = std::cell::RefCell::new(Vec::new());
    static BVHS: std::cell::RefCell<Vec<Bvh>> = std::cell::RefCell::new(Vec::new());
}

fn new_handle<T>(tl: &impl Fn(&std::cell::RefCell<Vec<T>>) -> std::cell::RefCell<Vec<T>>, value: T) -> u64 {
    // We can't pass closures like this. Use direct access.
    0
}

pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        // ── Spatial Hash ─────────────────────────────────────────────────
        "spatial::hash_new" => {
            let cs = f64_arg(args, 0).unwrap_or(1.0) as f32;
            SPATIAL_HASHES.with(|v| {
                let mut v = v.borrow_mut();
                v.push(SpatialHash::new(cs));
                Value::U64(v.len() as u64)
            });
            Some(Ok(Value::U64(SPATIAL_HASHES.with(|v| v.borrow().len() as u64))))
        }
        "spatial::hash_insert" => {
            if args.len() < 5 { return Some(Err(rt_err!("hash_insert() requires handle, id, x, y, z"))); }
            if let (Some(h), Some(id), Some(x), Some(y), Some(z)) =
                (i64_arg(args,0), i64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4)) {
                SPATIAL_HASHES.with(|v| {
                    let mut v = v.borrow_mut();
                    if let Some(sh) = v.get_mut(h as usize - 1) {
                        sh.insert(id as u64, x as f32, y as f32, z as f32);
                        Some(Ok(Value::Unit))
                    } else { Some(Err(rt_err!("hash_insert(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("hash_insert() requires handle, id, x, y, z"))) }
        }
        "spatial::hash_query" => {
            if args.len() < 5 { return Some(Err(rt_err!("hash_query() requires handle, x, y, z, radius"))); }
            if let (Some(h), Some(x), Some(y), Some(z), Some(r)) =
                (i64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4)) {
                SPATIAL_HASHES.with(|v| {
                    let v = v.borrow();
                    if let Some(sh) = v.get(h as usize - 1) {
                        let ids = sh.query(x as f32, y as f32, z as f32, r as f32);
                        let vals: Vec<Value> = ids.into_iter().map(|id| Value::U64(id)).collect();
                        Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(vals)))))
                    } else { Some(Err(rt_err!("hash_query(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("hash_query() requires handle, x, y, z, radius"))) }
        }
        "spatial::hash_clear" => {
            if let Some(h) = i64_arg(args, 0) {
                SPATIAL_HASHES.with(|v| {
                    let mut v = v.borrow_mut();
                    if let Some(sh) = v.get_mut(h as usize - 1) { sh.clear(); Some(Ok(Value::Unit)) }
                    else { Some(Err(rt_err!("hash_clear(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("hash_clear() requires handle"))) }
        }

        // ── Quadtree ─────────────────────────────────────────────────────
        "spatial::quadtree_new" => {
            if args.len() < 4 { return Some(Err(rt_err!("quadtree_new() requires x, y, w, h"))); }
            if let (Some(x), Some(y), Some(w), Some(h)) =
                (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3)) {
                let depth = i64_arg(args, 4).unwrap_or(6) as u32;
                let max_items = i64_arg(args, 5).unwrap_or(10) as usize;
                QUADTREES.with(|v| {
                    let mut v = v.borrow_mut();
                    v.push(Quadtree::new(x as f32, y as f32, w as f32, h as f32, depth, max_items));
                    Some(Ok(Value::U64(v.len() as u64)))
                })
            } else { Some(Err(rt_err!("quadtree_new() requires x, y, w, h"))) }
        }
        "spatial::quadtree_insert" => {
            if args.len() < 4 { return Some(Err(rt_err!("quadtree_insert() requires handle, id, x, y"))); }
            if let (Some(h), Some(id), Some(x), Some(y)) =
                (i64_arg(args,0), i64_arg(args,1), f64_arg(args,2), f64_arg(args,3)) {
                QUADTREES.with(|v| {
                    let mut v = v.borrow_mut();
                    if let Some(qt) = v.get_mut(h as usize - 1) {
                        qt.insert(id as u64, x as f32, y as f32);
                        Some(Ok(Value::Unit))
                    } else { Some(Err(rt_err!("quadtree_insert(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("quadtree_insert() requires handle, id, x, y"))) }
        }
        "spatial::quadtree_query" => {
            if args.len() < 4 { return Some(Err(rt_err!("quadtree_query() requires handle, x, y, radius"))); }
            if let (Some(h), Some(x), Some(y), Some(r)) =
                (i64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3)) {
                QUADTREES.with(|v| {
                    let v = v.borrow();
                    if let Some(qt) = v.get(h as usize - 1) {
                        let ids = qt.query(x as f32, y as f32, r as f32);
                        let vals: Vec<Value> = ids.into_iter().map(|id| Value::U64(id)).collect();
                        Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(vals)))))
                    } else { Some(Err(rt_err!("quadtree_query(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("quadtree_query() requires handle, x, y, radius"))) }
        }

        // ── Octree ───────────────────────────────────────────────────────
        "spatial::octree_new" => {
            if args.len() < 4 { return Some(Err(rt_err!("octree_new() requires cx, cy, cz, half"))); }
            if let (Some(cx), Some(cy), Some(cz), Some(h)) =
                (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3)) {
                let depth = i64_arg(args, 4).unwrap_or(5) as u32;
                let max_items = i64_arg(args, 5).unwrap_or(8) as usize;
                OCTREES.with(|v| {
                    let mut v = v.borrow_mut();
                    v.push(Octree::new(cx as f32, cy as f32, cz as f32, h as f32, depth, max_items));
                    Some(Ok(Value::U64(v.len() as u64)))
                })
            } else { Some(Err(rt_err!("octree_new() requires cx, cy, cz, half"))) }
        }
        "spatial::octree_insert" => {
            if args.len() < 5 { return Some(Err(rt_err!("octree_insert() requires handle, id, x, y, z"))); }
            if let (Some(h), Some(id), Some(x), Some(y), Some(z)) =
                (i64_arg(args,0), i64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4)) {
                OCTREES.with(|v| {
                    let mut v = v.borrow_mut();
                    if let Some(o) = v.get_mut(h as usize - 1) {
                        o.insert(id as u64, x as f32, y as f32, z as f32);
                        Some(Ok(Value::Unit))
                    } else { Some(Err(rt_err!("octree_insert(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("octree_insert() requires handle, id, x, y, z"))) }
        }
        "spatial::octree_query" => {
            if args.len() < 5 { return Some(Err(rt_err!("octree_query() requires handle, x, y, z, radius"))); }
            if let (Some(h), Some(x), Some(y), Some(z), Some(r)) =
                (i64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4)) {
                OCTREES.with(|v| {
                    let v = v.borrow();
                    if let Some(o) = v.get(h as usize - 1) {
                        let ids = o.query(x as f32, y as f32, z as f32, r as f32);
                        let vals: Vec<Value> = ids.into_iter().map(|id| Value::U64(id)).collect();
                        Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(vals)))))
                    } else { Some(Err(rt_err!("octree_query(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("octree_query() requires handle, x, y, z, radius"))) }
        }

        // ── BVH ──────────────────────────────────────────────────────────
        "spatial::bvh_new" => Some(Ok(Value::U64({
            BVHS.with(|v| { let mut v = v.borrow_mut(); v.push(Bvh::new()); v.len() as u64 })
        }))),
        "spatial::bvh_build" => {
            if args.len() < 2 { return Some(Err(rt_err!("bvh_build() requires handle, items"))); }
            if let (Some(h), Value::Array(arr)) = (i64_arg(args,0), &args[1]) {
                let arr = arr.lock().unwrap();
                // Items: array of tuples (id: u64, min: vec3, max: vec3)
                let mut items = Vec::new();
                for v in arr.iter() {
                    if let Value::Tuple(t) = v {
                        if t.len() >= 3 {
                            if let (Value::U64(id), Value::Vec3(mn), Value::Vec3(mx)) = (&t[0], &t[1], &t[2]) {
                                items.push((*id, *mn, *mx));
                            }
                        }
                    }
                }
                BVHS.with(|v| {
                    let mut v = v.borrow_mut();
                    if let Some(bvh) = v.get_mut(h as usize - 1) {
                        bvh.build(&items);
                        Some(Ok(Value::Unit))
                    } else { Some(Err(rt_err!("bvh_build(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("bvh_build() requires handle, items"))) }
        }
        "spatial::bvh_query" => {
            if args.len() < 7 { return Some(Err(rt_err!("bvh_query() requires handle, min(xyz), max(xyz)"))); }
            if let (Some(h), Some(x0), Some(y0), Some(z0), Some(x1), Some(y1), Some(z1)) =
                (i64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4), f64_arg(args,5), f64_arg(args,6)) {
                BVHS.with(|v| {
                    let v = v.borrow();
                    if let Some(bvh) = v.get(h as usize - 1) {
                        let ids = bvh.query(
                            [x0 as f32, y0 as f32, z0 as f32],
                            [x1 as f32, y1 as f32, z1 as f32],
                        );
                        let vals: Vec<Value> = ids.into_iter().map(|id| Value::U64(id)).collect();
                        Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(vals)))))
                    } else { Some(Err(rt_err!("bvh_query(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("bvh_query() requires handle, min, max"))) }
        }

        _ => None,
    }
}
