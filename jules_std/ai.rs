// =============================================================================
// std/ai — Jules Standard Library: Agent AI
//
// Behavior trees (sequence, selector, condition, action, parallel),
// Steering behaviors (seek, flee, arrive, pursue, evade, wander, obstacle avoidance),
// Boids / flocking, Utility AI, finite state machines.
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

// ─── Steering Behaviors ─────────────────────────────────────────────────────

/// Seek: steer toward target position
pub fn steer_seek(pos: [f32;2], vel: [f32;2], target: [f32;2], max_speed: f32, max_force: f32) -> [f32;2] {
    let mut desired = [target[0] - pos[0], target[1] - pos[1]];
    let d = (desired[0]*desired[0] + desired[1]*desired[1]).sqrt();
    if d < 0.001 { return [0.0, 0.0]; }
    let scale = max_speed / d;
    desired[0] *= scale;
    desired[1] *= scale;
    let mut steer = [desired[0] - vel[0], desired[1] - vel[1]];
    let s = (steer[0]*steer[0] + steer[1]*steer[1]).sqrt();
    if s > max_force {
        let scale = max_force / s;
        steer[0] *= scale;
        steer[1] *= scale;
    }
    steer
}

/// Flee: steer away from target
pub fn steer_flee(pos: [f32;2], vel: [f32;2], target: [f32;2], max_speed: f32, max_force: f32) -> [f32;2] {
    let steer = steer_seek(pos, vel, target, max_speed, max_force);
    [-steer[0], -steer[1]]
}

/// Arrive: steer toward target, slowing down as it approaches
pub fn steer_arrive(pos: [f32;2], vel: [f32;2], target: [f32;2], slow_radius: f32, max_speed: f32, max_force: f32) -> [f32;2] {
    let mut desired = [target[0] - pos[0], target[1] - pos[1]];
    let d = (desired[0]*desired[0] + desired[1]*desired[1]).sqrt();
    if d < 0.001 { return [0.0, 0.0]; }
    let speed = if d < slow_radius { max_speed * d / slow_radius } else { max_speed };
    let scale = speed / d;
    desired[0] *= scale;
    desired[1] *= scale;
    let mut steer = [desired[0] - vel[0], desired[1] - vel[1]];
    let s = (steer[0]*steer[0] + steer[1]*steer[1]).sqrt();
    if s > max_force {
        let scale = max_force / s;
        steer[0] *= scale;
        steer[1] *= scale;
    }
    steer
}

/// Pursue: predict future position and seek there
pub fn steer_pursue(pos: [f32;2], vel: [f32;2], target_pos: [f32;2], target_vel: [f32;2], max_speed: f32, max_force: f32) -> [f32;2] {
    let dx = target_pos[0] - pos[0]; let dy = target_pos[1] - pos[1];
    let d = (dx*dx + dy*dy).sqrt();
    let speed = (vel[0]*vel[0] + vel[1]*vel[1]).sqrt().max(0.001);
    let prediction = d / speed;
    let future = [target_pos[0] + target_vel[0] * prediction, target_pos[1] + target_vel[1] * prediction];
    steer_seek(pos, vel, future, max_speed, max_force)
}

/// Evade: flee from predicted future position
pub fn steer_evade(pos: [f32;2], vel: [f32;2], target_pos: [f32;2], target_vel: [f32;2], max_speed: f32, max_force: f32) -> [f32;2] {
    let steer = steer_pursue(pos, vel, target_pos, target_vel, max_speed, max_force);
    [-steer[0], -steer[1]]
}

/// Wander: random steering force
pub fn steer_wander(pos: [f32;2], heading: [f32;2], wander_radius: f32, wander_dist: f32, wander_jitter: f32, max_speed: f32, max_force: f32, theta: &mut f32) -> [f32;2] {
    // Add random jitter to wander angle
    *theta += (rand_f32() - 0.5) * wander_jitter;
    let wander_x = wander_radius * theta.cos();
    let wander_y = wander_radius * theta.sin();
    let target_x = pos[0] + heading[0] * wander_dist + wander_x;
    let target_y = pos[1] + heading[1] * wander_dist + wander_y;
    steer_seek(pos, [0.0, 0.0], [target_x, target_y], max_speed, max_force)
}

/// Obstacle avoidance: steer away from nearby obstacles
pub fn steer_avoid(pos: [f32;2], vel: [f32;2], obstacles: &[[f32;3]], avoidance_radius: f32, max_force: f32) -> [f32;2] {
    let mut steer = [0.0, 0.0];
    let speed = (vel[0]*vel[0] + vel[1]*vel[1]).sqrt().max(0.001);
    let look_ahead = avoidance_radius * 2.0;
    let mut count = 0;
    for obs in obstacles {
        let [ox, oy, or] = *obs;
        let mut closest = [ox - pos[0], oy - pos[1]];
        let dx = pos[0] - ox; let dy = pos[1] - oy;
        let dist = (dx*dx + dy*dy).sqrt();
        let combined_r = avoidance_radius + or;
        if dist < combined_r {
            // Project obstacle onto velocity axis
            let ndx = vel[0] / speed; let ndy = vel[1] / speed;
            let proj = closest[0] * ndx + closest[1] * ndy;
            if proj > 0.0 && proj < look_ahead {
                // Lateral distance
                let lat = (dist * dist - proj * proj).sqrt().max(0.0);
                let threat = combined_r - lat;
                if threat > 0.0 {
                    let push = (look_ahead - proj) / look_ahead;
                    steer[0] += ndx * threat * push;
                    steer[1] += ndy * threat * push;
                    count += 1;
                }
            }
        }
    }
    if count > 0 {
        let s = (steer[0]*steer[0] + steer[1]*steer[1]).sqrt().max(0.001);
        if s > max_force {
            let scale = max_force / s;
            steer[0] *= scale;
            steer[1] *= scale;
        }
    }
    steer
}

// ─── Boids / Flocking ───────────────────────────────────────────────────────

pub fn boid_separation(pos: [f32;2], neighbors: &[[f32;2]], min_dist: f32) -> [f32;2] {
    let mut steer = [0.0, 0.0];
    let mut count = 0;
    for &np in neighbors {
        let dx = pos[0] - np[0]; let dy = pos[1] - np[1];
        let d = (dx*dx + dy*dy).sqrt();
        if d < min_dist && d > 0.0 {
            steer[0] += dx / d;
            steer[1] += dy / d;
            count += 1;
        }
    }
    if count > 0 {
        steer[0] /= count as f32;
        steer[1] /= count as f32;
    }
    steer
}

pub fn boid_alignment(vel: [f32;2], neighbor_vel: &[[f32;2]], max_speed: f32) -> [f32;2] {
    if neighbor_vel.is_empty() { return [0.0, 0.0]; }
    let mut avg_vx = 0.0; let mut avg_vy = 0.0;
    for &v in neighbor_vel {
        avg_vx += v[0]; avg_vy += v[1];
    }
    avg_vx /= neighbor_vel.len() as f32;
    avg_vy /= neighbor_vel.len() as f32;
    let s = (avg_vx*avg_vx + avg_vy*avg_vy).sqrt().max(0.001);
    if s > max_speed {
        avg_vx = avg_vx / s * max_speed;
        avg_vy = avg_vy / s * max_speed;
    }
    [avg_vx - vel[0], avg_vy - vel[1]]
}

pub fn boid_cohesion(pos: [f32;2], neighbor_pos: &[[f32;2]], max_speed: f32) -> [f32;2] {
    if neighbor_pos.is_empty() { return [0.0, 0.0]; }
    let mut cx = 0.0; let mut cy = 0.0;
    for &p in neighbor_pos {
        cx += p[0]; cy += p[1];
    }
    cx /= neighbor_pos.len() as f32;
    cy /= neighbor_pos.len() as f32;
    let mut desired = [cx - pos[0], cy - pos[1]];
    let d = (desired[0]*desired[0] + desired[1]*desired[1]).sqrt().max(0.001);
    if d > max_speed {
        desired[0] = desired[0] / d * max_speed;
        desired[1] = desired[1] / d * max_speed;
    }
    [desired[0], desired[1]]
}

/// Full boid update: combines separation, alignment, cohesion
pub fn boid_update(
    pos: [f32;2], vel: [f32;2],
    neighbor_pos: &[[f32;2]], neighbor_vel: &[[f32;2]],
    sep_weight: f32, ali_weight: f32, coh_weight: f32,
    max_speed: f32, max_force: f32, min_sep: f32,
) -> [f32;2] {
    let sep = boid_separation(pos, neighbor_pos, min_sep);
    let ali = boid_alignment(vel, neighbor_vel, max_speed);
    let coh = boid_cohesion(pos, neighbor_pos, max_speed);

    let mut steer = [
        sep[0] * sep_weight + ali[0] * ali_weight + coh[0] * coh_weight,
        sep[1] * sep_weight + ali[1] * ali_weight + coh[1] * coh_weight,
    ];
    let s = (steer[0]*steer[0] + steer[1]*steer[1]).sqrt();
    if s > max_force {
        let scale = max_force / s;
        steer[0] *= scale;
        steer[1] *= scale;
    }
    steer
}

// ─── Utility AI ──────────────────────────────────────────────────────────────

/// Score a set of options and pick the highest.
/// Each option is scored as: score = sum(curve_i(value_i) * weight_i)
/// Curves: "linear", "sigmoid", "exponential", "threshold"
pub fn utility_ai_score(
    values: &[f32],    // normalized [0,1] values for each consideration
    weights: &[f32],   // weights for each consideration
    curves: &[u8],     // 0=linear, 1=sigmoid, 2=exponential, 3=threshold
) -> f32 {
    let mut total = 0.0;
    let mut total_w = 0.0;
    for i in 0..values.len().min(weights.len()).min(curves.len()) {
        let v = values[i].clamp(0.0, 1.0);
        let scored = match curves[i] {
            0 => v,
            1 => { let x = (v - 0.5) * 12.0; 1.0 / (1.0 + (-x).exp()) },
            2 => v * v,
            3 => if v > 0.5 { 1.0 } else { 0.0 },
            _ => v,
        };
        total += scored * weights[i];
        total_w += weights[i];
    }
    if total_w > 0.0 { total / total_w } else { 0.0 }
}

pub fn utility_ai_pick(
    scores: &[f32], // pre-computed utility scores for each action
) -> usize {
    if scores.is_empty() { return 0; }
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    // Softmax selection (temperature = 1.0)
    let exp_scores: Vec<f32> = scores.iter().map(|s| ((s - max_s) * 2.0).exp()).collect();
    let sum: f32 = exp_scores.iter().sum();
    let mut r = rand_f32() * sum;
    for (i, e) in exp_scores.iter().enumerate() {
        r -= e;
        if r <= 0.0 { return i; }
    }
    scores.len() - 1
}

// ─── Behavior Tree (simple) ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum BtNode {
    Sequence(Vec<BtNode>),
    Selector(Vec<BtNode>),
    Parallel { nodes: Vec<BtNode>, required: usize },
    Condition { name: String },
    Action { name: String },
    Inverter(Box<BtNode>),
    Repeater { node: Box<BtNode>, count: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtStatus {
    Success,
    Failure,
    Running,
}

/// Evaluate a behavior tree given a set of condition results.
pub fn bt_evaluate(node: &BtNode, conditions: &std::collections::HashMap<&str, bool>) -> BtStatus {
    match node {
        BtNode::Sequence(children) => {
            for child in children {
                match bt_evaluate(child, conditions) {
                    BtStatus::Success => {},
                    BtStatus::Failure => return BtStatus::Failure,
                    BtStatus::Running => return BtStatus::Running,
                }
            }
            BtStatus::Success
        }
        BtNode::Selector(children) => {
            for child in children {
                match bt_evaluate(child, conditions) {
                    BtStatus::Failure => {},
                    other => return other,
                }
            }
            BtStatus::Failure
        }
        BtNode::Parallel { nodes, required } => {
            let mut success = 0;
            for child in nodes {
                match bt_evaluate(child, conditions) {
                    BtStatus::Success => success += 1,
                    BtStatus::Failure => {},
                    BtStatus::Running => {},
                }
            }
            if success >= *required { BtStatus::Success } else { BtStatus::Running }
        }
        BtNode::Condition { name } => {
            if conditions.get(name.as_str()).copied().unwrap_or(false) {
                BtStatus::Success
            } else {
                BtStatus::Failure
            }
        }
        BtNode::Action { .. } => BtStatus::Success, // Actions always succeed immediately in this model
        BtNode::Inverter(child) => {
            match bt_evaluate(child, conditions) {
                BtStatus::Success => BtStatus::Failure,
                BtStatus::Failure => BtStatus::Success,
                BtStatus::Running => BtStatus::Running,
            }
        }
        BtNode::Repeater { node, count } => {
            for _ in 0..*count {
                match bt_evaluate(node, conditions) {
                    BtStatus::Success => {},
                    other => return other,
                }
            }
            BtStatus::Success
        }
    }
}

// ─── Simple PRNG (for wander steering) ──────────────────────────────────────

thread_local! {
    static AI_RAND: std::cell::Cell<u64> = std::cell::Cell::new(12345);
}

fn rand_f32() -> f32 {
    AI_RAND.with(|r| {
        let mut s = r.get();
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s = s.wrapping_mul(0x2545F4914F6CDD1D);
        r.set(s);
        ((s >> 33) as f32) / (u32::MAX as f32)
    })
}

// ─── Builtin dispatch ───────────────────────────────────────────────────────

pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        // ── Steering ─────────────────────────────────────────────────────
        "ai::seek" => {
            if args.len() < 5 { return Some(Err(rt_err!("ai::seek() requires pos, vel, target, max_speed, max_force"))); }
            if let (Some(px), Some(py), Some(vx), Some(vy), Some(tx), Some(ty), Some(ms), Some(mf)) =
                (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4), f64_arg(args,5), f64_arg(args,6), f64_arg(args,7)) {
                let s = steer_seek([px as f32, py as f32], [vx as f32, vy as f32], [tx as f32, ty as f32], ms as f32, mf as f32);
                Some(Ok(Value::Vec2(s)))
            } else { Some(Err(rt_err!("ai::seek() requires 8 floats"))) }
        }
        "ai::flee" => {
            if args.len() < 5 { return Some(Err(rt_err!("ai::flee() requires pos, vel, target, max_speed, max_force"))); }
            if let (Some(px), Some(py), Some(vx), Some(vy), Some(tx), Some(ty), Some(ms), Some(mf)) =
                (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4), f64_arg(args,5), f64_arg(args,6), f64_arg(args,7)) {
                let s = steer_flee([px as f32, py as f32], [vx as f32, vy as f32], [tx as f32, ty as f32], ms as f32, mf as f32);
                Some(Ok(Value::Vec2(s)))
            } else { Some(Err(rt_err!("ai::flee() requires 8 floats"))) }
        }
        "ai::arrive" => {
            if args.len() < 6 { return Some(Err(rt_err!("ai::arrive() requires pos, vel, target, slow_r, max_speed, max_force"))); }
            if let (Some(px), Some(py), Some(vx), Some(vy), Some(tx), Some(ty), Some(sr), Some(ms), Some(mf)) =
                (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4), f64_arg(args,5), f64_arg(args,6), f64_arg(args,7), f64_arg(args,8)) {
                let s = steer_arrive([px as f32, py as f32], [vx as f32, vy as f32], [tx as f32, ty as f32], sr as f32, ms as f32, mf as f32);
                Some(Ok(Value::Vec2(s)))
            } else { Some(Err(rt_err!("ai::arrive() requires 9 floats"))) }
        }
        "ai::pursue" => {
            if args.len() < 6 { return Some(Err(rt_err!("ai::pursue() requires pos, vel, tpos, tvel, max_speed, max_force"))); }
            if let (Some(px), Some(py), Some(vx), Some(vy), Some(tx), Some(ty), Some(tvx), Some(tvy), Some(ms), Some(mf)) =
                (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4), f64_arg(args,5), f64_arg(args,6), f64_arg(args,7), f64_arg(args,8), f64_arg(args,9)) {
                let s = steer_pursue([px as f32, py as f32], [vx as f32, vy as f32], [tx as f32, ty as f32], [tvx as f32, tvy as f32], ms as f32, mf as f32);
                Some(Ok(Value::Vec2(s)))
            } else { Some(Err(rt_err!("ai::pursue() requires 10 floats"))) }
        }
        "ai::evade" => {
            if args.len() < 6 { return Some(Err(rt_err!("ai::evade() requires pos, vel, tpos, tvel, max_speed, max_force"))); }
            if let (Some(px), Some(py), Some(vx), Some(vy), Some(tx), Some(ty), Some(tvx), Some(tvy), Some(ms), Some(mf)) =
                (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4), f64_arg(args,5), f64_arg(args,6), f64_arg(args,7), f64_arg(args,8), f64_arg(args,9)) {
                let s = steer_evade([px as f32, py as f32], [vx as f32, vy as f32], [tx as f32, ty as f32], [tvx as f32, tvy as f32], ms as f32, mf as f32);
                Some(Ok(Value::Vec2(s)))
            } else { Some(Err(rt_err!("ai::evade() requires 10 floats"))) }
        }

        // ── Boids ────────────────────────────────────────────────────────
        "ai::boid_separation" => {
            if args.len() < 4 { return Some(Err(rt_err!("boid_separation() requires px, py, min_dist, neighbor_positions"))); }
            if let (Some(px), Some(py), Some(md), Value::Array(narr)) = (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), &args[3]) {
                let narr = narr.lock().unwrap();
                let neighbors: Vec<[f32;2]> = narr.iter().filter_map(|v| {
                    if let Value::Vec2(a) = v { Some(*a) } else { None }
                }).collect();
                let s = boid_separation([px as f32, py as f32], &neighbors, md as f32);
                Some(Ok(Value::Vec2(s)))
            } else { Some(Err(rt_err!("boid_separation() requires px, py, min_dist, [vec2, ...]"))) }
        }
        "ai::boid_cohesion" => {
            if args.len() < 3 { return Some(Err(rt_err!("boid_cohesion() requires px, py, max_speed, neighbor_positions"))); }
            if let (Some(px), Some(py), Some(ms), Value::Array(narr)) = (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), &args[3]) {
                let narr = narr.lock().unwrap();
                let neighbors: Vec<[f32;2]> = narr.iter().filter_map(|v| {
                    if let Value::Vec2(a) = v { Some(*a) } else { None }
                }).collect();
                let s = boid_cohesion([px as f32, py as f32], &neighbors, ms as f32);
                Some(Ok(Value::Vec2(s)))
            } else { Some(Err(rt_err!("boid_cohesion() requires px, py, max_speed, [vec2, ...]"))) }
        }
        "ai::boid_alignment" => {
            if args.len() < 3 { return Some(Err(rt_err!("boid_alignment() requires vx, vy, max_speed, neighbor_velocities"))); }
            if let (Some(vx), Some(vy), Some(ms), Value::Array(narr)) = (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), &args[3]) {
                let narr = narr.lock().unwrap();
                let neighbors: Vec<[f32;2]> = narr.iter().filter_map(|v| {
                    if let Value::Vec2(a) = v { Some(*a) } else { None }
                }).collect();
                let s = boid_alignment([vx as f32, vy as f32], &neighbors, ms as f32);
                Some(Ok(Value::Vec2(s)))
            } else { Some(Err(rt_err!("boid_alignment() requires vx, vy, max_speed, [vec2, ...]"))) }
        }
        "ai::boid_update" => {
            if args.len() < 7 { return Some(Err(rt_err!("boid_update() requires pos, vel, neighbor_pos, neighbor_vel, weights, params"))); }
            // Simplified: take individual args
            if let (Some(px), Some(py), Some(vx), Some(vy)) = (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3)) {
                let sep_w = f64_arg(args, 4).unwrap_or(2.5) as f32;
                let ali_w = f64_arg(args, 5).unwrap_or(1.0) as f32;
                let coh_w = f64_arg(args, 6).unwrap_or(1.0) as f32;
                let ms = f64_arg(args, 7).unwrap_or(3.0) as f32;
                let mf = f64_arg(args, 8).unwrap_or(0.3) as f32;
                let min_sep = f64_arg(args, 9).unwrap_or(25.0) as f32;
                let pos = [px as f32, py as f32];
                let vel = [vx as f32, vy as f32];
                // Get neighbors from args[10] and args[11]
                let (npos, nvel) = if args.len() > 11 {
                    let npos: Vec<[f32;2]> = if let Value::Array(a) = &args[10] {
                        a.lock().ok().map(|arr| arr.iter().filter_map(|v| { if let Value::Vec2(x)=v{Some(*x)} else{None} }).collect()).unwrap_or_default()
                    } else { vec![] };
                    let nvel: Vec<[f32;2]> = if let Value::Array(a) = &args[11] {
                        a.lock().ok().map(|arr| arr.iter().filter_map(|v| { if let Value::Vec2(x)=v{Some(*x)} else{None} }).collect()).unwrap_or_default()
                    } else { vec![] };
                    (npos, nvel)
                } else { (vec![], vec![]) };
                let s = boid_update(pos, vel, &npos, &nvel, sep_w, ali_w, coh_w, ms, mf, min_sep);
                Some(Ok(Value::Vec2(s)))
            } else { Some(Err(rt_err!("boid_update() requires at least pos and vel"))) }
        }

        // ── Utility AI ───────────────────────────────────────────────────
        "ai::utility_score" => {
            if args.len() < 3 { return Some(Err(rt_err!("utility_score() requires values[], weights[], curves[]"))); }
            if let (Value::Array(va), Value::Array(wa), Value::Array(ca)) = (&args[0], &args[1], &args[2]) {
                let vals: Vec<f32> = va.lock().ok().map(|a| a.iter().filter_map(|v| v.as_f64().map(|x| x as f32)).collect()).unwrap_or_default();
                let wts: Vec<f32> = wa.lock().ok().map(|a| a.iter().filter_map(|v| v.as_f64().map(|x| x as f32)).collect()).unwrap_or_default();
                let crv: Vec<u8> = ca.lock().ok().map(|a| a.iter().filter_map(|v| v.as_i64().map(|x| x as u8)).collect()).unwrap_or_default();
                Some(Ok(Value::F32(utility_ai_score(&vals, &wts, &crv))))
            } else { Some(Err(rt_err!("utility_score() requires 3 arrays"))) }
        }

        // ── Behavior Trees ───────────────────────────────────────────────
        "ai::bt_sequence" => {
            // Takes array of child statuses and returns overall status
            if let Value::Array(arr) = &args[0] {
                let arr = arr.lock().unwrap();
                // For simplicity: sequence succeeds if all are "success"
                let all_success = arr.iter().all(|v| matches!(v, Value::Str(s) if s == "success"));
                let has_running = arr.iter().any(|v| matches!(v, Value::Str(s) if s == "running"));
                if all_success { Some(Ok(Value::Str("success".into()))) }
                else if has_running { Some(Ok(Value::Str("running".into()))) }
                else { Some(Ok(Value::Str("failure".into()))) }
            } else { Some(Err(rt_err!("bt_sequence() requires array"))) }
        }
        "ai::bt_selector" => {
            if let Value::Array(arr) = &args[0] {
                let arr = arr.lock().unwrap();
                let any_success = arr.iter().any(|v| matches!(v, Value::Str(s) if s == "success"));
                let all_failure = arr.iter().all(|v| matches!(v, Value::Str(s) if s == "failure"));
                if any_success { Some(Ok(Value::Str("success".into()))) }
                else if all_failure { Some(Ok(Value::Str("failure".into()))) }
                else { Some(Ok(Value::Str("running".into()))) }
            } else { Some(Err(rt_err!("bt_selector() requires array"))) }
        }

        _ => None,
    }
}
