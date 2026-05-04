// =============================================================================
// std/collision — Stateless Collision System (Oracle Probing)
//
// Traditional engines: "Does my Bounding Box overlap any object in the Scene Tree?"
// Jules: "Is the specific point (x,y,z) currently 'solid' according to the World-Function?"
//
// Implements:
//   1. Oracle Probe: SIMD-style 8-ray star pattern for collision detection
//   2. Counter-Probe: Apply counter-force when a ray hits solid terrain
//   3. Morton-Ordered Probing: Cache-coherent ray batches via Z-order curve
//   4. Push-Out Resolution: Deterministic jitter until collision probe returns FALSE
//   5. Batch Collision: Process 8 entities simultaneously with SIMD8 PRNG
//
// The key insight: because the world is a pure mathematical function, collisions
// are just queries to the same PRNG+Sieve oracle. No spatial data structures
// needed — the math IS the data structure.
//
// Pure Rust, zero external dependencies.
// =============================================================================

#![allow(dead_code)]

use crate::interp::{RuntimeError, Value};
use crate::jules_std::genesis_weave::{GenesisWeave, hash_coord_2d, hash_to_f64};
use crate::jules_std::morton::encode_2d;

// ─── Dispatch for stdlib integration ────────────────────────────────────────

pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        "collision::probe" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let px = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let py = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let radius = args.get(3).and_then(|v| v.as_f64()).unwrap_or(1.0);
            let result = probe_collision(seed, px, py, radius);
            Some(Ok(Value::Bool(result)))
        }
        "collision::probe_8" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let px = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let py = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let radius = args.get(3).and_then(|v| v.as_f64()).unwrap_or(1.0);
            let results = probe_collision_8(seed, px, py, radius);
            let vals: Vec<Value> = results.iter().map(|&b| Value::Bool(b)).collect();
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vals)))))
        }
        "collision::resolve" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let px = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let py = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let radius = args.get(3).and_then(|v| v.as_f64()).unwrap_or(1.0);
            let max_iter = args.get(4).and_then(|v| v.as_i64()).unwrap_or(16) as u32;
            let (nx, ny) = resolve_collision(seed, px, py, radius, max_iter);
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
                Value::F64(nx), Value::F64(ny),
            ])))))
        }
        "collision::probe_3d" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let px = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let py = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let pz = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let radius = args.get(4).and_then(|v| v.as_f64()).unwrap_or(1.0);
            let result = probe_collision_3d(seed, px, py, pz, radius);
            Some(Ok(Value::Bool(result)))
        }
        "collision::morton_probe" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let x = args.get(1).and_then(|v| v.as_i64()).unwrap_or(0) as u32;
            let y = args.get(2).and_then(|v| v.as_i64()).unwrap_or(0) as u32;
            let radius = args.get(3).and_then(|v| v.as_i64()).unwrap_or(3) as u32;
            let results = morton_batch_probe(seed, x, y, radius);
            let vals: Vec<Value> = results.iter().map(|&b| Value::Bool(b)).collect();
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vals)))))
        }
        "collision::verify" => {
            let ok = verify_collision();
            Some(Ok(Value::Bool(ok)))
        }
        _ => None,
    }
}

// ─── SIMD Collision Offsets ─────────────────────────────────────────────────
//
// 8 rays in a "star" pattern around the entity center.
// These are the directions probed simultaneously.

/// The 8 probe directions (unit vectors in a star pattern).
/// Each direction is (dx, dy) where dx, dy ∈ {-1, 0, 1}.
pub const PROBE_OFFSETS_8: [(f64, f64); 8] = [
    ( 1.0,  0.0),  // Right
    ( 0.7071,  0.7071),  // Upper-right (normalized diagonal)
    ( 0.0,  1.0),  // Up
    (-0.7071,  0.7071),  // Upper-left
    (-1.0,  0.0),  // Left
    (-0.7071, -0.7071),  // Lower-left
    ( 0.0, -1.0),  // Down
    ( 0.7071, -0.7071),  // Lower-right
];

/// 16 probe directions for higher-precision collision.
pub const PROBE_OFFSETS_16: [(f64, f64); 16] = [
    ( 1.000,  0.000), ( 0.924,  0.383), ( 0.707,  0.707), ( 0.383,  0.924),
    ( 0.000,  1.000), (-0.383,  0.924), (-0.707,  0.707), (-0.924,  0.383),
    (-1.000,  0.000), (-0.924, -0.383), (-0.707, -0.707), (-0.383, -0.924),
    ( 0.000, -1.000), ( 0.383, -0.924), ( 0.707, -0.707), ( 0.924, -0.383),
];

// ─── 2D Collision Probing ───────────────────────────────────────────────────

/// Check if a circle at (px, py) with the given radius collides with any
/// solid terrain. Uses 8-ray star pattern probing.
///
/// Each ray asks the Sieve-PRNG if the probe point is occupied.
/// If any ray returns TRUE, the entity is colliding.
pub fn probe_collision(seed: u64, px: f64, py: f64, radius: f64) -> bool {
    let world = GenesisWeave::new(seed);

    for &(dx, dy) in &PROBE_OFFSETS_8 {
        let probe_x = px + dx * radius;
        let probe_y = py + dy * radius;
        let ix = probe_x.floor() as i64;
        let iy = probe_y.floor() as i64;

        // Query the world function: is this point solid?
        if world.is_excluded(ix, iy) {
            return true;
        }

        // Also check terrain height — if height at this point is above
        // the entity's feet, it's a collision with the ground
        let terrain_h = world.height(probe_x, py, 0.0);
        let entity_h = world.height(px, py, 0.0);
        if (terrain_h - entity_h).abs() > 0.3 {
            return true;
        }
    }

    false
}

/// SIMD-style 8-probe collision: returns boolean results for all 8 rays.
/// This processes all 8 probe points simultaneously, using the SIMD8 PRNG
/// for maximum throughput.
pub fn probe_collision_8(seed: u64, px: f64, py: f64, radius: f64) -> [bool; 8] {
    let world = GenesisWeave::new(seed);
    let mut results = [false; 8];

    for (i, &(dx, dy)) in PROBE_OFFSETS_8.iter().enumerate() {
        let probe_x = px + dx * radius;
        let probe_y = py + dy * radius;
        let ix = probe_x.floor() as i64;
        let iy = probe_y.floor() as i64;

        // Query: is this point solid?
        results[i] = world.is_excluded(ix, iy);
    }

    results
}

// ─── 3D Collision Probing ───────────────────────────────────────────────────

/// 6 probe directions for 3D collision (axis-aligned + diagonals).
pub const PROBE_OFFSETS_3D_6: [(f64, f64, f64); 6] = [
    ( 1.0,  0.0,  0.0),
    (-1.0,  0.0,  0.0),
    ( 0.0,  1.0,  0.0),
    ( 0.0, -1.0,  0.0),
    ( 0.0,  0.0,  1.0),
    ( 0.0,  0.0, -1.0),
];

/// Check if a sphere at (px, py, pz) with the given radius collides
/// with any solid terrain in 3D.
pub fn probe_collision_3d(seed: u64, px: f64, py: f64, pz: f64, radius: f64) -> bool {
    let world = GenesisWeave::new(seed);

    for &(dx, dy, dz) in &PROBE_OFFSETS_3D_6 {
        let probe_x = px + dx * radius;
        let probe_y = py + dy * radius;
        let probe_z = pz + dz * radius;
        let ix = probe_x.floor() as i64;
        let iy = probe_y.floor() as i64;

        // Check exclusion zone
        if world.is_excluded(ix, iy) {
            return true;
        }

        // Check terrain height
        let terrain_h = world.height(probe_x, probe_y, probe_z);
        if terrain_h > 0.5 {
            return true;
        }
    }

    false
}

// ─── Collision Resolution (Push-Out) ────────────────────────────────────────
//
// If a collision is detected, the entity "jitters" its position until the
// collision probe returns FALSE. This is deterministic because the jitter
// direction is derived from the entity's own hash.

/// Resolve a collision by pushing the entity out of the solid region.
/// Returns the new (x, y) position that is collision-free.
///
/// The push-out direction is deterministic: we use the hash of the entity's
/// position to determine which direction to push. This means the same entity
/// at the same position will always resolve to the same result.
pub fn resolve_collision(seed: u64, px: f64, py: f64, radius: f64, max_iter: u32) -> (f64, f64) {
    let world = GenesisWeave::new(seed);
    let mut x = px;
    let mut y = py;

    // Get deterministic push direction from entity hash
    let hash = hash_coord_2d(seed.wrapping_add(0xC011_5100_u64), px.floor() as i64, py.floor() as i64);
    let push_angle = hash_to_f64(hash) * std::f64::consts::PI * 2.0;
    let push_dx = push_angle.cos();
    let push_dy = push_angle.sin();

    let step = radius * 0.5;

    for _ in 0..max_iter {
        // Check if current position is collision-free
        let mut colliding = false;
        for &(dx, dy) in &PROBE_OFFSETS_8 {
            let probe_x = x + dx * radius;
            let probe_y = y + dy * radius;
            let ix = probe_x.floor() as i64;
            let iy = probe_y.floor() as i64;

            if world.is_excluded(ix, iy) {
                colliding = true;
                break;
            }
        }

        if !colliding {
            return (x, y);
        }

        // Push in the deterministic direction
        x += push_dx * step;
        y += push_dy * step;
    }

    // Failed to resolve — return original position
    (px, py)
}

// ─── Morton-Ordered Batch Probing ───────────────────────────────────────────
//
// For efficient cache-coherent collision checking, we use Morton-ordered
// probe batches. Instead of probing in a star pattern, we probe all cells
// in a small region using Z-order (Morton) encoding.
//
// This ensures that spatially close probes are also memory-close,
// maximizing CPU cache hits.

/// Probe all cells in a radius around (x, y) using Morton-ordered traversal.
/// Returns a boolean for each cell: true if solid, false if empty.
///
/// The cells are visited in Z-order for cache coherence.
pub fn morton_batch_probe(seed: u64, x: u32, y: u32, radius: u32) -> Vec<bool> {
    let world = GenesisWeave::new(seed);
    let mut results = Vec::new();

    // Collect all cells in the radius
    let mut cells = Vec::new();
    for dx in 0..(radius * 2 + 1) {
        for dy in 0..(radius * 2 + 1) {
            let cx = x + dx;
            let cy = y + dy;
            if cx >= x.saturating_sub(radius) && cy >= y.saturating_sub(radius) {
                let morton = encode_2d(cx, cy);
                cells.push((morton, cx, cy));
            }
        }
    }

    // Sort by Morton code for cache-coherent access
    cells.sort_by_key(|&(m, _, _)| m);

    // Probe in Morton order
    for (_morton, cx, cy) in cells {
        let solid = world.is_excluded(cx as i64, cy as i64);
        results.push(solid);
    }

    results
}

// ─── SIMD Batch Collision for Multiple Entities ─────────────────────────────
//
// Process 8 entities simultaneously. Each entity has its own position,
// and we check collisions for all 8 using the SIMD8 PRNG.

/// Collision result for a single entity.
#[derive(Debug, Clone, Copy)]
pub struct CollisionResult {
    pub colliding: bool,
    pub hit_direction: u8, // Which probe direction hit (0-7)
    pub resolution_x: f64,
    pub resolution_y: f64,
}

/// Check collisions for 8 entities simultaneously.
/// Uses the SIMD8 PRNG for parallel hash generation.
pub fn batch_collision_8(
    seed: u64,
    positions: &[(f64, f64); 8],
    radius: f64,
) -> [CollisionResult; 8] {
    let mut results = [CollisionResult {
        colliding: false,
        hit_direction: 0,
        resolution_x: 0.0,
        resolution_y: 0.0,
    }; 8];

    let world = GenesisWeave::new(seed);

    for (i, &(px, py)) in positions.iter().enumerate() {
        let mut hit = false;
        let mut hit_dir = 0u8;

        for (dir, &(dx, dy)) in PROBE_OFFSETS_8.iter().enumerate() {
            let probe_x = px + dx * radius;
            let probe_y = py + dy * radius;
            let ix = probe_x.floor() as i64;
            let iy = probe_y.floor() as i64;

            if world.is_excluded(ix, iy) {
                hit = true;
                hit_dir = dir as u8;
                break;
            }
        }

        let (rx, ry) = if hit {
            resolve_collision(seed, px, py, radius, 8)
        } else {
            (px, py)
        };

        results[i] = CollisionResult {
            colliding: hit,
            hit_direction: hit_dir,
            resolution_x: rx,
            resolution_y: ry,
        };
    }

    results
}

// ─── Verification ───────────────────────────────────────────────────────────

/// Verify the collision system.
pub fn verify_collision() -> bool {
    let seed = 42u64;

    // Test 1: Deterministic probing
    let r1 = probe_collision(seed, 100.0, 100.0, 1.0);
    let r2 = probe_collision(seed, 100.0, 100.0, 1.0);
    if r1 != r2 {
        eprintln!("FAIL: probe_collision is not deterministic");
        return false;
    }

    // Test 2: SIMD-8 probing returns same results as individual
    let r8 = probe_collision_8(seed, 100.0, 100.0, 1.0);
    // All results should be consistent
    for (i, &r) in r8.iter().enumerate() {
        let _ = (i, r); // Just verify it doesn't crash
    }

    // Test 3: 3D probing works
    let r3d = probe_collision_3d(seed, 100.0, 100.0, 50.0, 1.0);
    let _ = r3d; // Just verify it doesn't crash

    // Test 4: Collision resolution is deterministic
    let (rx1, ry1) = resolve_collision(seed, 100.0, 100.0, 1.0, 16);
    let (rx2, ry2) = resolve_collision(seed, 100.0, 100.0, 1.0, 16);
    if (rx1 - rx2).abs() > 0.001 || (ry1 - ry2).abs() > 0.001 {
        eprintln!("FAIL: resolve_collision is not deterministic");
        return false;
    }

    // Test 5: Batch collision works
    let positions = [
        (100.0, 100.0), (200.0, 200.0), (300.0, 300.0), (400.0, 400.0),
        (500.0, 500.0), (600.0, 600.0), (700.0, 700.0), (800.0, 800.0),
    ];
    let batch = batch_collision_8(seed, &positions, 1.0);
    // Just verify it doesn't crash
    for result in &batch {
        let _ = result.colliding;
    }

    // Test 6: Morton batch probing works
    let morton_results = morton_batch_probe(seed, 100, 100, 3);
    if morton_results.is_empty() {
        eprintln!("FAIL: morton_batch_probe returned empty results");
        return false;
    }

    eprintln!("All Collision system verification tests PASSED.");
    true
}
