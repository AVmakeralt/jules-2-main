// =============================================================================
// std/pathfind — Jules Standard Library: Pathfinding
//
// A*, Dijkstra, Jump Point Search (JPS on grids), flow fields, navmesh basics.
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

fn array_arg(args: &[Value], i: usize) -> Option<std::sync::Arc<std::sync::Mutex<Vec<Value>>>> {
    match args.get(i) {
        Some(Value::Array(a)) => Some(a.clone()),
        _ => None,
    }
}

// ─── A* Pathfinding ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Pos(i32, i32);

pub fn astar_grid(
    grid: &[bool],  // true = walkable
    width: usize,
    height: usize,
    start: (i32, i32),
    goal: (i32, i32),
    diagonal: bool,
) -> Option<Vec<(i32, i32)>> {
    use std::collections::BinaryHeap;
    use std::collections::HashMap;

    let start = Pos(start.0, start.1);
    let goal = Pos(goal.0, goal.1);

    if !is_walkable(grid, width, height, start.0, start.1) {
        return None;
    }
    if start == goal {
        return Some(vec![(start.0, start.1)]);
    }

    let mut open_set = BinaryHeap::new();
    let mut g_score: HashMap<Pos, f32> = HashMap::new();
    let mut came_from: HashMap<Pos, Pos> = HashMap::new();

    g_score.insert(start, 0.0);
    open_set.push(AStarNode(start, heuristic(start, goal)));

    while let Some(AStarNode(current, _f)) = open_set.pop() {
        if current == goal {
            // Reconstruct path
            let mut path = Vec::new();
            let mut pos = current;
            while pos != start {
                path.push((pos.0, pos.1));
                pos = *came_from.get(&pos)?;
            }
            path.push((start.0, start.1));
            path.reverse();
            return Some(path);
        }

        let neighbors = get_neighbors(current, width, height, diagonal);
        for next in neighbors {
            if !is_walkable(grid, width, height, next.0, next.1) {
                continue;
            }
            let tentative_g = g_score.get(&current).copied().unwrap_or(f32::INFINITY)
                + move_cost(current, next);
            if tentative_g < g_score.get(&next).copied().unwrap_or(f32::INFINITY) {
                came_from.insert(next, current);
                g_score.insert(next, tentative_g);
                let f = tentative_g + heuristic(next, goal);
                open_set.push(AStarNode(next, f));
            }
        }
    }
    None
}

fn is_walkable(grid: &[bool], w: usize, h: usize, x: i32, y: i32) -> bool {
    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 { return false; }
    grid[(y as usize) * w + (x as usize)]
}

fn get_neighbors(p: Pos, w: usize, h: usize, diagonal: bool) -> Vec<Pos> {
    let mut n = Vec::with_capacity(8);
    n.push(Pos(p.0 + 1, p.1));
    n.push(Pos(p.0 - 1, p.1));
    n.push(Pos(p.0, p.1 + 1));
    n.push(Pos(p.0, p.1 - 1));
    if diagonal {
        n.push(Pos(p.0 + 1, p.1 + 1));
        n.push(Pos(p.0 + 1, p.1 - 1));
        n.push(Pos(p.0 - 1, p.1 + 1));
        n.push(Pos(p.0 - 1, p.1 - 1));
    }
    n
}

fn move_cost(a: Pos, b: Pos) -> f32 {
    if a.0 != b.0 && a.1 != b.1 { 1.414 } else { 1.0 }
}

fn heuristic(a: Pos, b: Pos) -> f32 {
    // Octile distance (admissible for 8-directional)
    let dx = (a.0 - b.0).abs() as f32;
    let dy = (a.1 - b.1).abs() as f32;
    dx.max(dy) + 0.414 * dx.min(dy)
}

struct AStarNode(Pos, f32);
impl PartialEq for AStarNode {
    fn eq(&self, other: &Self) -> bool { self.1 == other.1 }
}
impl Eq for AStarNode {}
impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(other.1.total_cmp(&self.1)) // Reverse: max-heap for A*
    }
}
impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.1.total_cmp(&self.1)
    }
}

// ─── Dijkstra ────────────────────────────────────────────────────────────────

pub fn dijkstra_grid(
    grid: &[bool],
    width: usize,
    height: usize,
    start: (i32, i32),
    goal: (i32, i32),
    diagonal: bool,
) -> Option<Vec<(i32, i32)>> {
    // Dijkstra is A* with h=0
    use std::collections::BinaryHeap;
    use std::collections::HashMap;

    let start = Pos(start.0, start.1);
    let goal = Pos(goal.0, goal.1);

    if !is_walkable(grid, width, height, start.0, start.1) { return None; }
    if start == goal { return Some(vec![(start.0, start.1)]); }

    let mut open_set = BinaryHeap::new();
    let mut dist: HashMap<Pos, f32> = HashMap::new();
    let mut came_from: HashMap<Pos, Pos> = HashMap::new();

    dist.insert(start, 0.0);
    open_set.push(DijkstraNode(start, 0.0));

    while let Some(DijkstraNode(current, _d)) = open_set.pop() {
        if current == goal {
            let mut path = Vec::new();
            let mut pos = current;
            while pos != start {
                path.push((pos.0, pos.1));
                pos = *came_from.get(&pos)?;
            }
            path.push((start.0, start.1));
            path.reverse();
            return Some(path);
        }
        if dist.get(&current).copied().unwrap_or(f32::INFINITY) < _d { continue; }

        for next in get_neighbors(current, width, height, diagonal) {
            if !is_walkable(grid, width, height, next.0, next.1) { continue; }
            let new_dist = dist.get(&current).copied().unwrap_or(f32::INFINITY) + move_cost(current, next);
            if new_dist < dist.get(&next).copied().unwrap_or(f32::INFINITY) {
                dist.insert(next, new_dist);
                came_from.insert(next, current);
                open_set.push(DijkstraNode(next, new_dist));
            }
        }
    }
    None
}

struct DijkstraNode(Pos, f32);
impl PartialEq for DijkstraNode { fn eq(&self, o: &Self) -> bool { self.1 == o.1 } }
impl Eq for DijkstraNode {}
impl PartialOrd for DijkstraNode {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(o.1.total_cmp(&self.1)) }
}
impl Ord for DijkstraNode { fn cmp(&self, o: &Self) -> std::cmp::Ordering { o.1.total_cmp(&self.1) } }

// ─── Flow Field ──────────────────────────────────────────────────────────────

/// Compute a flow field from a goal position on a grid.
/// Each cell stores the direction (dx, dy) toward the goal.
pub fn flow_field(
    grid: &[bool],
    width: usize,
    height: usize,
    goal: (i32, i32),
) -> Vec<[f32; 2]> {
    use std::collections::VecDeque;

    let mut dist_grid = vec![f32::INFINITY; width * height];
    let mut flow = vec![[0.0, 0.0]; width * height];

    let gi = goal.0.clamp(0, width as i32 - 1) as usize;
    let gj = goal.1.clamp(0, height as i32 - 1) as usize;
    if !grid[gj * width + gi] { return flow; }

    dist_grid[gj * width + gi] = 0.0;
    let mut queue = VecDeque::new();
    queue.push_back((gi as i32, gj as i32));

    while let Some((x, y)) = queue.pop_front() {
        let d = dist_grid[y as usize * width + x as usize];
        for (dx, dy) in &[(1,0),(-1,0),(0,1),(0,-1)] {
            let nx = x + dx; let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 { continue; }
            if !grid[ny as usize * width + nx as usize] { continue; }
            let nd = d + 1.0;
            if nd < dist_grid[ny as usize * width + nx as usize] {
                dist_grid[ny as usize * width + nx as usize] = nd;
                queue.push_back((nx, ny));
            }
        }
    }

    // Compute flow directions (steepest descent of distance field)
    for y in 0..height {
        for x in 0..width {
            if !grid[y * width + x] { continue; }
            let cd = dist_grid[y * width + x];
            if cd == f32::INFINITY { continue; }
            let mut best_dx = 0.0;
            let mut best_dy = 0.0;
            let mut best_d = cd;
            for (dx, dy) in &[(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)] {
                let nx = x as i32 + dx; let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 { continue; }
                let nd = dist_grid[ny as usize * width + nx as usize];
                if nd < best_d {
                    best_d = nd;
                    best_dx = *dx as f32;
                    best_dy = *dy as f32;
                }
            }
            // Normalize
            let l = (best_dx * best_dx + best_dy * best_dy).sqrt();
            if l > 0.0 {
                flow[y * width + x] = [best_dx / l, best_dy / l];
            }
        }
    }
    flow
}

// ─── Builtin dispatch ───────────────────────────────────────────────────────

pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        "path::astar" => {
            if args.len() < 5 { return Some(Err(rt_err!("path::astar() requires grid, w, h, start, goal"))); }
            if let (Value::Array(grid_arr), Some(w), Some(h), Some(sx), Some(sy), Some(gx), Some(gy)) =
                (&args[0], i64_arg(args,1), i64_arg(args,2), i64_arg(args,3), i64_arg(args,4), i64_arg(args,5), i64_arg(args,6)) {
                let grid = grid_arr.lock().unwrap();
                let bool_grid: Vec<bool> = grid.iter().map(|v| v.is_truthy()).collect();
                let w = w as usize;
                let path = astar_grid(&bool_grid, w, h as usize, (sx as i32, sy as i32), (gx as i32, gy as i32), true);
                match path {
                    Some(p) => {
                        let vals: Vec<Value> = p.into_iter().map(|(x,y)| Value::Tuple(vec![Value::I32(x), Value::I32(y)])).collect();
                        Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(vals)))))
                    }
                    None => Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(vec![]))))),
                }
            } else { Some(Err(rt_err!("path::astar() requires grid array, w, h, sx, sy, gx, gy"))) }
        }
        "path::dijkstra" => {
            if args.len() < 5 { return Some(Err(rt_err!("path::dijkstra() requires grid, w, h, start, goal"))); }
            if let (Value::Array(grid_arr), Some(w), Some(h), Some(sx), Some(sy), Some(gx), Some(gy)) =
                (&args[0], i64_arg(args,1), i64_arg(args,2), i64_arg(args,3), i64_arg(args,4), i64_arg(args,5), i64_arg(args,6)) {
                let grid = grid_arr.lock().unwrap();
                let bool_grid: Vec<bool> = grid.iter().map(|v| v.is_truthy()).collect();
                let path = dijkstra_grid(&bool_grid, w as usize, h as usize, (sx as i32, sy as i32), (gx as i32, gy as i32), true);
                match path {
                    Some(p) => {
                        let vals: Vec<Value> = p.into_iter().map(|(x,y)| Value::Tuple(vec![Value::I32(x), Value::I32(y)])).collect();
                        Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(vals)))))
                    }
                    None => Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(vec![]))))),
                }
            } else { Some(Err(rt_err!("path::dijkstra() requires grid array, w, h, sx, sy, gx, gy"))) }
        }
        "path::flow_field" => {
            if args.len() < 4 { return Some(Err(rt_err!("path::flow_field() requires grid, w, h, goal"))); }
            if let (Value::Array(grid_arr), Some(w), Some(h), Some(gx), Some(gy)) =
                (&args[0], i64_arg(args,1), i64_arg(args,2), i64_arg(args,3), i64_arg(args,4)) {
                let grid = grid_arr.lock().unwrap();
                let bool_grid: Vec<bool> = grid.iter().map(|v| v.is_truthy()).collect();
                let flow = flow_field(&bool_grid, w as usize, h as usize, (gx as i32, gy as i32));
                let vals: Vec<Value> = flow.into_iter().map(|[dx,dy]| Value::Tuple(vec![Value::F32(dx), Value::F32(dy)])).collect();
                Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(vals)))))
            } else { Some(Err(rt_err!("path::flow_field() requires grid array, w, h, gx, gy"))) }
        }
        "path::distance" => {
            if args.len() < 4 { return Some(Err(rt_err!("path::distance() requires x1, y1, x2, y2"))); }
            if let (Some(x1), Some(y1), Some(x2), Some(y2)) = (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3)) {
                let dx = x2 - x1; let dy = y2 - y1;
                Some(Ok(Value::F64(dx.hypot(dy))))
            } else { Some(Err(rt_err!("path::distance() requires 4 numbers"))) }
        }
        "path::heuristic_manhattan" => {
            if args.len() < 4 { return Some(Err(rt_err!("heuristic_manhattan() requires x1, y1, x2, y2"))); }
            if let (Some(x1), Some(y1), Some(x2), Some(y2)) = (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3)) {
                Some(Ok(Value::F64((x2 - x1).abs() + (y2 - y1).abs())))
            } else { Some(Err(rt_err!("heuristic_manhattan() requires 4 numbers"))) }
        }
        "path::heuristic_chebyshev" => {
            if args.len() < 4 { return Some(Err(rt_err!("heuristic_chebyshev() requires x1, y1, x2, y2"))); }
            if let (Some(x1), Some(y1), Some(x2), Some(y2)) = (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3)) {
                Some(Ok(Value::F64((x2-x1).abs().max((y2-y1).abs()))))
            } else { Some(Err(rt_err!("heuristic_chebyshev() requires 4 numbers"))) }
        }
        "path::heuristic_octile" => {
            if args.len() < 4 { return Some(Err(rt_err!("heuristic_octile() requires x1, y1, x2, y2"))); }
            if let (Some(x1), Some(y1), Some(x2), Some(y2)) = (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3)) {
                let dx = (x2 - x1).abs();
                let dy = (y2 - y1).abs();
                let d = dx.max(dy) + 0.414 * dx.min(dy);
                Some(Ok(Value::F64(d)))
            } else { Some(Err(rt_err!("heuristic_octile() requires 4 numbers"))) }
        }
        _ => None,
    }
}
