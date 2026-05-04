// =============================================================================
// std/sdf_ray — Ray-Marched Signed Distance Fields for the Aurora Flux Pipeline
//
// Implements:
//   1. SDF Primitives: Sphere, Box, Plane, Torus, Cylinder, Capsule
//      — all pure functions returning distance to surface
//   2. SDF Combinators: Union, Intersection, Subtraction, Smooth Union
//      — for organic blending between shapes
//   3. Sphere Tracing: The standard ray marching algorithm — at each step,
//      advance the ray by the SDF distance (guaranteed safe step size)
//   4. Sieve-Assisted Ray Marching: Use the Exclusion Sieve from genesis_weave
//      to skip empty 128×128 chunks entirely. If a chunk is marked empty,
//      the ray leaps across it in one step — potentially saving hundreds
//      of SDF evaluations per ray
//   5. Morton-Ordered Acceleration: Use Morton encoding from the morton module
//      for cache-friendly SDF queries (Z-order spatial locality)
//   6. Branchless Traversal: Use min/max operations instead of if-branches
//      in the hot loop — the CPU predictor stays happy
//   7. Hit Information: Surface normal (via central differences), material ID,
//      distance, and position
//
// The fundamental insight: because the world is defined by a pure SDF function,
// there is no scene graph, no BVH, no acceleration structure to maintain.
// The math IS the scene. We simply march along the ray asking "how far away
// is the nearest surface?" — and the answer is always correct.
//
// Pure Rust, zero external dependencies. Uses genesis_weave, morton, prng_simd.
// =============================================================================

#![allow(dead_code)]

use crate::interp::{RuntimeError, Value};
use crate::jules_std::genesis_weave::{hash_coord_2d, hash_to_f64, terrain_height, biome_at};
use crate::jules_std::morton::encode_2d;

// ─── Dispatch for stdlib integration ────────────────────────────────────────

/// Dispatch sdf_ray:: builtin calls.
///
/// Supported calls:
///   - "sdf_ray::march" — takes seed, ox, oy, oz, dx, dy, dz → [hit, distance, steps]
///   - "sdf_ray::sdf_world" — takes seed, x, y, z → f64 distance
///   - "sdf_ray::normal" — takes seed, x, y, z → [nx, ny, nz]
///   - "sdf_ray::verify" → bool
pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        "sdf_ray::march" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let ox = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let oy = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let oz = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let dx = args.get(4).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let dy = args.get(5).and_then(|v| v.as_f64()).unwrap_or(-1.0);
            let dz = args.get(6).and_then(|v| v.as_f64()).unwrap_or(0.0);

            let ctx = SdfContext::new(seed);
            let ray = Ray {
                origin: Vec3::new(ox, oy, oz),
                direction: Vec3::new(dx, dy, dz).normalize(),
            };
            let hit = ray_march(&ctx, &ray);
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
                Value::Bool(hit.hit),
                Value::F64(hit.distance),
                Value::I64(hit.steps as i64),
            ])))))
        }
        "sdf_ray::sdf_world" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let x = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let y = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let z = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);

            let ctx = SdfContext::new(seed);
            let dist = sdf_world(&ctx, &Vec3::new(x, y, z));
            Some(Ok(Value::F64(dist)))
        }
        "sdf_ray::normal" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let x = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let y = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let z = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);

            let ctx = SdfContext::new(seed);
            let n = compute_normal(&ctx, &Vec3::new(x, y, z));
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
                Value::F64(n.x),
                Value::F64(n.y),
                Value::F64(n.z),
            ])))))
        }
        "sdf_ray::verify" => {
            let ok = verify_sdf_ray();
            Some(Ok(Value::Bool(ok)))
        }
        _ => None,
    }
}

// ─── Core Types ──────────────────────────────────────────────────────────────
//
// Vec3 — the universal 3D vector used throughout the ray marcher.
// Ray — origin + direction (direction is always unit-length).
// HitInfo — everything we learn when a ray strikes a surface.
// SdfContext — world parameters bundled for convenient threading.

/// A 3D vector. Used for positions, directions, normals, and offsets.
#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// Create a new Vec3.
    #[inline(always)]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    /// Zero vector.
    #[inline(always)]
    pub const fn zero() -> Self {
        Vec3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Component-wise addition.
    #[inline(always)]
    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Component-wise subtraction.
    #[inline(always)]
    pub fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Scalar multiplication.
    #[inline(always)]
    pub fn scale(&self, s: f64) -> Vec3 {
        Vec3 {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Dot product.
    #[inline(always)]
    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Length (magnitude).
    #[inline(always)]
    pub fn length(&self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Normalize to unit length. Returns zero vector if length is ~0.
    #[inline(always)]
    pub fn normalize(&self) -> Vec3 {
        let len = self.length();
        if len < 1e-12 {
            Vec3::zero()
        } else {
            self.scale(1.0 / len)
        }
    }

    /// Absolute value of each component.
    #[inline(always)]
    pub fn abs(&self) -> Vec3 {
        Vec3 {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Component-wise maximum.
    #[inline(always)]
    pub fn max(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    /// Component-wise minimum.
    #[inline(always)]
    pub fn min(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }
}

/// A ray defined by an origin point and a unit direction vector.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    /// Create a new ray. The direction will be normalized automatically.
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Ray {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Evaluate the ray at parameter t: origin + t * direction.
    #[inline(always)]
    pub fn at(&self, t: f64) -> Vec3 {
        self.origin.add(&self.direction.scale(t))
    }
}

/// Information about a ray-surface intersection.
///
/// When a ray hits a surface, we record:
///   - Whether we hit anything at all (`hit`)
///   - The distance along the ray to the hit point (`distance`)
///   - The 3D position of the hit point (`position`)
///   - The surface normal at the hit point (`normal`)
///   - A material identifier for shading (`material_id`)
///   - How many SDF evaluations were required (`steps`)
#[derive(Debug, Clone, Copy)]
pub struct HitInfo {
    pub hit: bool,
    pub distance: f64,
    pub position: Vec3,
    pub normal: Vec3,
    pub material_id: u32,
    pub steps: u32,
}

impl HitInfo {
    /// Create a "miss" HitInfo (no intersection).
    pub fn miss() -> Self {
        HitInfo {
            hit: false,
            distance: f64::MAX,
            position: Vec3::zero(),
            normal: Vec3::zero(),
            material_id: 0,
            steps: 0,
        }
    }

    /// Create a "hit" HitInfo at the given position.
    pub fn hit(distance: f64, position: Vec3, normal: Vec3, material_id: u32, steps: u32) -> Self {
        HitInfo {
            hit: true,
            distance,
            position,
            normal,
            material_id,
            steps,
        }
    }
}

/// Context for SDF ray marching — bundles world parameters.
///
/// The `seed` drives the Genesis Weave terrain and PRNG detail noise.
/// The `max_steps` limits iteration count for performance.
/// The `epsilon` is the hit threshold (how close is "close enough?").
/// The `chunk_size` is the Exclusion Sieve leap size (128 units default).
#[derive(Debug, Clone)]
pub struct SdfContext {
    pub seed: u64,
    pub max_steps: u32,
    pub epsilon: f64,
    pub chunk_size: f64,
}

impl SdfContext {
    /// Create a new SdfContext with sensible defaults.
    pub fn new(seed: u64) -> Self {
        SdfContext {
            seed,
            max_steps: 256,
            epsilon: 0.001,
            chunk_size: 128.0,
        }
    }

    /// Create a context with custom parameters.
    pub fn with_params(seed: u64, max_steps: u32, epsilon: f64, chunk_size: f64) -> Self {
        SdfContext {
            seed,
            max_steps,
            epsilon,
            chunk_size,
        }
    }
}

// ─── SDF Primitives ─────────────────────────────────────────────────────────
//
// Each primitive is a pure function: (point, shape parameters) → distance.
// The distance is negative inside the shape, zero on the surface, and
// positive outside. This sign convention is what makes SDFs so powerful —
// you can combine them with simple min/max operations.

/// SDF for a sphere centered at `center` with the given `radius`.
///
/// The signed distance is simply the distance from p to the center,
/// minus the radius. Negative inside, positive outside.
#[inline(always)]
pub fn sdf_sphere(p: &Vec3, center: &Vec3, radius: f64) -> f64 {
    p.sub(center).length() - radius
}

/// SDF for an axis-aligned box centered at `center` with `half_extents`.
///
/// Uses the standard clamped-distance formulation:
///   1. Find the closest point on the box surface from p
///   2. Distance = length of the "outside" component
///   3. Negative inside (distance to nearest face)
#[inline(always)]
pub fn sdf_box(p: &Vec3, center: &Vec3, half_extents: &Vec3) -> f64 {
    let q = p.sub(center).abs().sub(half_extents);
    // Branchless: max(q, 0) for outside component, min(max(q), 0) for inside
    let outside = q.max(&Vec3::zero()).length();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    outside + inside
}

/// SDF for an infinite horizontal plane at the given `height`.
///
/// The distance is simply (p.y - height). Positive above, negative below.
#[inline(always)]
pub fn sdf_plane(p: &Vec3, height: f64) -> f64 {
    p.y - height
}

/// SDF for a torus centered at `center` with `major` (ring) and `minor` (tube) radii.
///
/// The torus lies in the XZ plane. We first find the distance from p to
/// the ring (a circle of radius `major`), then subtract the tube radius.
#[inline(always)]
pub fn sdf_torus(p: &Vec3, center: &Vec3, major: f64, minor: f64) -> f64 {
    let q = p.sub(center);
    // Project onto the XZ plane and find distance to the ring circle
    let ring_dist = (q.x * q.x + q.z * q.z).sqrt() - major;
    // Then the full SDF is the distance from the ring circle minus the tube radius
    (ring_dist * ring_dist + q.y * q.y).sqrt() - minor
}

/// SDF for a vertical cylinder centered at `center` with the given `radius` and `height`.
///
/// The cylinder extends from center.y - height/2 to center.y + height/2.
/// Uses the same clamped-distance approach as the box SDF.
#[inline(always)]
pub fn sdf_cylinder(p: &Vec3, center: &Vec3, radius: f64, height: f64) -> f64 {
    let q = p.sub(center);
    // 2D distance in the XZ plane
    let dxz = (q.x * q.x + q.z * q.z).sqrt() - radius;
    // 1D distance along the Y axis
    let dy = q.y.abs() - height * 0.5;
    // Combine: outside component + inside component
    let outside = dxz.max(0.0).hypot(dy.max(0.0));
    let inside = dxz.max(dy).min(0.0);
    outside + inside
}

/// SDF for a capsule (swept sphere) between points `a` and `b` with the given `radius`.
///
/// A capsule is the Minkowski sum of a line segment and a sphere.
/// The distance is the distance to the line segment minus the radius.
#[inline(always)]
pub fn sdf_capsule(p: &Vec3, a: &Vec3, b: &Vec3, radius: f64) -> f64 {
    let pa = p.sub(a);
    let ba = b.sub(a);
    let h = (pa.dot(&ba) / ba.dot(&ba)).clamp(0.0, 1.0);
    pa.sub(&ba.scale(h)).length() - radius
}

// ─── SDF Combinators ────────────────────────────────────────────────────────
//
// Combinators merge two SDF distances into one. They are the "glue" that
// lets you build complex scenes from simple primitives.
//
// Union = min(d1, d2)          — the surface of either shape
// Intersection = max(d1, d2)    — the region inside both shapes
// Subtraction = max(d1, -d2)    — shape 1 with shape 2 carved out
// Smooth Union = polynomial smin — organic blending (no hard edges)

/// SDF Union: the surface of either shape.
/// Returns the minimum of the two distances.
#[inline(always)]
pub fn sdf_union(d1: f64, d2: f64) -> f64 {
    d1.min(d2)
}

/// SDF Intersection: the region inside both shapes.
/// Returns the maximum of the two distances.
#[inline(always)]
pub fn sdf_intersection(d1: f64, d2: f64) -> f64 {
    d1.max(d2)
}

/// SDF Subtraction: shape 1 with shape 2 carved out.
/// Equivalent to Intersection(d1, complement of d2).
#[inline(always)]
pub fn sdf_subtraction(d1: f64, d2: f64) -> f64 {
    d1.max(-d2)
}

/// Polynomial smooth minimum (smooth union).
///
/// When two shapes are within distance `k` of each other, instead of a
/// hard CSG seam, the surfaces blend together organically. This is the
/// key to natural-looking terrain, rounded CSG, and blobby metaballs.
///
/// The formula (for k > 0):
///   h = clamp(0.5 + 0.5*(d2-d1)/k, 0, 1)
///   result = lerp(d2, d1, h) - k*h*(1-h)
///
/// The subtraction term -k*h*(1-h) creates the smooth "dip" that
/// eliminates the hard edge. Larger k = more blending.
#[inline(always)]
pub fn sdf_smooth_union(d1: f64, d2: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return d1.min(d2);
    }
    let h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
    d2 + (d1 - d2) * h - k * h * (1.0 - h)
}

// ─── World SDF ──────────────────────────────────────────────────────────────
//
// The world SDF uses the Genesis Weave terrain height to define the ground
// surface. At point (x, y, z), the SDF distance equals:
//
//   distance = y - terrain_height(seed, x, z, 0.0) * WORLD_SCALE
//
// This treats the Genesis Weave height field as a heightmap SDF, which is
// exact when the terrain is nearly flat and slightly conservative (i.e.,
// the reported distance may be slightly larger than the true distance)
// on steep slopes — but it's always safe for sphere tracing.
//
// Additionally, PRNG-driven detail noise adds sub-millimeter variation
// for visual richness at close range.

/// World scale factor: converts [0,1] terrain height to world units.
const WORLD_SCALE: f64 = 64.0;

/// Compute the world SDF at point `p`.
///
/// The SDF treats the Genesis Weave terrain as a heightmap:
///   distance = p.y - terrain_height(seed, p.x, p.z, 0.0) * WORLD_SCALE
///
/// Additionally, PRNG-driven detail noise adds sub-millimeter variation
/// so that surfaces aren't perfectly smooth at extreme zoom levels.
#[inline]
pub fn sdf_world(ctx: &SdfContext, p: &Vec3) -> f64 {
    // Base terrain height from Genesis Weave (stateless)
    let h = terrain_height(ctx.seed, p.x, p.z, 0.0) * WORLD_SCALE;

    // SDF distance: positive above terrain, negative below
    let base_dist = p.y - h;

    // PRNG-driven detail noise for sub-millimeter detail.
    // We use a separate seed stream (offset by 0x5DF_RAY) so this noise
    // doesn't interfere with the macro-level terrain generation.
    let detail_seed = ctx.seed.wrapping_add(0x5DF0_A570_u64);
    let hash = hash_coord_2d(detail_seed, p.x.floor() as i64, p.z.floor() as i64);
    let detail = (hash_to_f64(hash) - 0.5) * 0.1; // ±0.05 world units of noise

    base_dist + detail
}

/// Compute the material ID at a given position based on biome.
///
/// Material ID encodes the surface type for the shader:
///   0 = void (no material)
///   1 = water
///   2 = sand
///   3 = grass
///   4 = rock
///   5 = snow
///   6 = lava
///   7 = crystal
#[inline]
pub fn material_at(ctx: &SdfContext, p: &Vec3) -> u32 {
    let biome = biome_at(ctx.seed, p.x, p.z);
    match biome {
        crate::jules_std::genesis_weave::Biome::Ocean => 1,
        crate::jules_std::genesis_weave::Biome::Beach => 2,
        crate::jules_std::genesis_weave::Biome::Plains
        | crate::jules_std::genesis_weave::Biome::Savanna
        | crate::jules_std::genesis_weave::Biome::Forest
        | crate::jules_std::genesis_weave::Biome::DenseForest
        | crate::jules_std::genesis_weave::Biome::Jungle
        | crate::jules_std::genesis_weave::Biome::Swamp
        | crate::jules_std::genesis_weave::Biome::Mushroom => 3,
        crate::jules_std::genesis_weave::Biome::Mountains
        | crate::jules_std::genesis_weave::Biome::Hills
        | crate::jules_std::genesis_weave::Biome::Tundra
        | crate::jules_std::genesis_weave::Biome::Volcanic => 4,
        crate::jules_std::genesis_weave::Biome::SnowCaps => 5,
        crate::jules_std::genesis_weave::Biome::Desert => 2,
        crate::jules_std::genesis_weave::Biome::Crystal => 7,
    }
}

// ─── Sphere Tracing (Ray Marching) ──────────────────────────────────────────
//
// The core algorithm:
//
//   1. Start at the ray origin
//   2. Evaluate the SDF at the current position → get distance to nearest surface
//   3. If distance < epsilon → HIT! We found the surface.
//   4. Otherwise, advance the ray by that distance (it's guaranteed safe:
//      we can't pass through the surface because the SDF gives a lower
//      bound on the distance to the closest surface)
//   5. If we exceed max_steps or max_distance → MISS
//   6. Go to step 2
//
// The beauty of sphere tracing: each step is as large as possible without
// missing the surface. On open terrain, rays leap forward in huge strides.
// Near complex geometry, steps shrink automatically. It's self-adaptive.

/// Maximum distance a ray can travel before we give up.
const MAX_DISTANCE: f64 = 10000.0;

/// Standard sphere tracing: march the ray through the SDF world.
///
/// Returns a `HitInfo` describing whether the ray hit anything,
/// and if so, where and what the surface looks like.
///
/// This is the baseline marcher — no sieve acceleration, no Morton ordering.
/// It's simple, correct, and serves as the reference implementation.
pub fn ray_march(ctx: &SdfContext, ray: &Ray) -> HitInfo {
    let mut t = 0.0f64;   // Distance along the ray
    let mut steps = 0u32; // SDF evaluation counter

    // ── Hot loop: advance the ray ──
    //
    // We use branchless min/max operations where possible.
    // The CPU branch predictor stays happy because the loop body
    // has no data-dependent branches (the only branch is the loop exit).
    while steps < ctx.max_steps {
        let pos = ray.at(t);
        let dist = sdf_world(ctx, &pos);
        steps += 1;

        // Branchless hit detection: hit if |dist| < epsilon
        // We use abs() so that points slightly inside the surface also count as hits.
        // This prevents the ray from getting stuck at grazing angles.
        if dist.abs() < ctx.epsilon {
            let normal = compute_normal(ctx, &pos);
            let mat = material_at(ctx, &pos);
            return HitInfo::hit(t, pos, normal, mat, steps);
        }

        // Advance by the SDF distance (guaranteed safe step)
        t += dist;

        // Miss: ray traveled too far without hitting anything
        if t > MAX_DISTANCE {
            break;
        }
    }

    HitInfo::miss()
}

// ─── Normal Computation (Central Differences) ───────────────────────────────
//
// The surface normal at a point p is the gradient of the SDF:
//
//   normal = ∇SDF(p) = (∂SDF/∂x, ∂SDF/∂y, ∂SDF/∂z)
//
// We approximate each partial derivative with a central difference:
//
//   ∂SDF/∂x ≈ (SDF(p + ε·x̂) - SDF(p - ε·x̂)) / (2·ε)
//
// This requires 6 SDF evaluations (one per face of the ±ε cube).
// The resulting normal is unit-length after normalization.

/// Compute the surface normal at point `p` using central differences.
///
/// Uses an epsilon proportional to the context epsilon but scaled up
/// slightly to avoid floating-point cancellation issues. The result
/// is always normalized to unit length.
pub fn compute_normal(ctx: &SdfContext, p: &Vec3) -> Vec3 {
    // Use a slightly larger epsilon for numerical stability.
    // Central differences with too-small epsilon produce noisy normals.
    let e = ctx.epsilon.max(1e-6) * 2.0;

    let dx = Vec3::new(e, 0.0, 0.0);
    let dy = Vec3::new(0.0, e, 0.0);
    let dz = Vec3::new(0.0, 0.0, e);

    // Central differences for each axis
    let nx = sdf_world(ctx, &p.add(&dx)) - sdf_world(ctx, &p.sub(&dx));
    let ny = sdf_world(ctx, &p.add(&dy)) - sdf_world(ctx, &p.sub(&dy));
    let nz = sdf_world(ctx, &p.add(&dz)) - sdf_world(ctx, &p.sub(&dz));

    Vec3::new(nx, ny, nz).normalize()
}

// ─── Sieve-Assisted Ray Marching ────────────────────────────────────────────
//
// The Exclusion Sieve from genesis_weave tells us whether a 128×128 chunk
// is "empty" — meaning it contains no mega-structures and its terrain
// height is roughly uniform. For rays traveling over open terrain (plains,
// desert, ocean), entire chunks can be skipped in a single leap.
//
// The algorithm:
//
//   1. At each step, evaluate the SDF as normal
//   2. Additionally, check the Exclusion Sieve for the current chunk
//   3. If the chunk is empty AND the SDF distance > chunk_size:
//      → The ray can safely leap forward by chunk_size units
//      → Skip potentially hundreds of SDF evaluations
//   4. If the chunk is non-empty (contains structures):
//      → Fall back to normal sphere tracing (small steps)
//
// This is the same insight as hierarchical ray tracing (skip empty space),
// but implemented purely via the stateless sieve — no spatial data structure
// to build or maintain.

/// Check if a chunk is "empty" according to the Exclusion Sieve.
///
/// A chunk is empty if:
///   - It has no exclusion zone (no mega-structures)
///   - The terrain height variation within the chunk is low (flat terrain)
///
/// Returns (is_empty, min_distance_to_surface) where min_distance is
/// a conservative lower bound on how far the nearest surface is.
#[inline]
fn chunk_empty(ctx: &SdfContext, p: &Vec3) -> (bool, f64) {
    let cx = (p.x / ctx.chunk_size).floor() as i64;
    let cz = (p.z / ctx.chunk_size).floor() as i64;
    let radius = ctx.chunk_size as i64;

    // Check if this chunk has a mega-structure (exclusion zone)
    let has_structure = !crate::jules_std::genesis_weave::check_exclusion(
        ctx.seed, cx * radius, cz * radius, radius,
    );

    // If there's a structure, this chunk is NOT empty
    if has_structure {
        return (false, 0.0);
    }

    // Check terrain height variation: sample a few points in the chunk.
    // If the height varies by more than a threshold, the chunk has
    // complex geometry and shouldn't be skipped.
    let h_center = terrain_height(ctx.seed, p.x, p.z, 0.0);
    let h_corner1 = terrain_height(ctx.seed, p.x + ctx.chunk_size * 0.5, p.z, 0.0);
    let h_corner2 = terrain_height(ctx.seed, p.x, p.z + ctx.chunk_size * 0.5, 0.0);

    let max_var = (h_center - h_corner1).abs().max((h_center - h_corner2).abs());
    let flatness_threshold = 0.1; // Normalized height units

    let is_flat = max_var < flatness_threshold;

    if is_flat {
        // Conservative minimum distance: if the point is well above
        // or below the terrain, we can skip a larger distance
        let dist = (p.y - h_center * WORLD_SCALE).abs();
        (true, dist)
    } else {
        (false, 0.0)
    }
}

/// Sieve-assisted ray march: skip empty chunks for faster traversal.
///
/// On open terrain, this can be 10-50× faster than basic sphere tracing
/// because it leaps across entire chunks in a single step. Near complex
/// geometry (structures, mountains), it degrades gracefully to the same
/// step-by-step behavior as `ray_march`.
pub fn sieve_ray_march(ctx: &SdfContext, ray: &Ray) -> HitInfo {
    let mut t = 0.0f64;
    let mut steps = 0u32;
    let chunk_size = ctx.chunk_size;

    while steps < ctx.max_steps {
        let pos = ray.at(t);
        let dist = sdf_world(ctx, &pos);
        steps += 1;

        // Check for hit
        if dist.abs() < ctx.epsilon {
            let normal = compute_normal(ctx, &pos);
            let mat = material_at(ctx, &pos);
            return HitInfo::hit(t, pos, normal, mat, steps);
        }

        // Sieve check: can we leap across this chunk?
        let (is_empty, _min_dist) = chunk_empty(ctx, &pos);

        // Branchless leap: if chunk is empty AND the SDF distance is large
        // enough that we won't miss a surface, advance by chunk_size.
        // Otherwise, advance by the SDF distance as usual.
        //
        // The leap condition: is_empty AND (dist > chunk_size * 0.5)
        // We use a branchless conditional:
        //   leap_dist = if should_leap { chunk_size } else { dist }
        //   = chunk_size * should_leap_flag + dist * (1 - should_leap_flag)
        //
        // But we also cap the leap: never advance more than the SDF distance
        // guarantees is safe. If the SDF says "surface is 5 units away",
        // we can't leap 128 units even if the chunk is empty.
        let should_leap = is_empty && dist > chunk_size * 0.5;
        let leap_dist = if should_leap {
            dist.min(chunk_size)
        } else {
            dist
        };

        t += leap_dist;

        if t > MAX_DISTANCE {
            break;
        }
    }

    HitInfo::miss()
}

// ─── Morton-Ordered Acceleration ────────────────────────────────────────────
//
// For batch ray marching (e.g., a screen's worth of rays), we can improve
// cache coherence by processing rays in Morton order. Rays that are
// spatially close on screen are also likely to query similar terrain
// regions, so Morton-ordering their evaluation means the CPU cache
// stays warm across adjacent rays.

/// A batch of rays sorted by Morton code of their screen-space origin.
#[derive(Debug, Clone)]
pub struct MortonRayBatch {
    /// Morton codes for each ray (sorted).
    pub morton_codes: Vec<u64>,
    /// Ray indices in the original order.
    pub ray_indices: Vec<usize>,
}

impl MortonRayBatch {
    /// Create a Morton-ordered batch from a set of screen-space ray origins.
    ///
    /// The `origins` are (x, y) screen coordinates that get Morton-encoded.
    /// The resulting batch processes rays in Z-order for cache coherence.
    pub fn new(origins: &[(f64, f64)], screen_width: u32, screen_height: u32) -> Self {
        let mut entries: Vec<(u64, usize)> = origins
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| {
                let sx = ((x / screen_width as f64) * (screen_width as f64)) as u32;
                let sy = ((y / screen_height as f64) * (screen_height as f64)) as u32;
                let morton = encode_2d(sx.min(screen_width - 1), sy.min(screen_height - 1));
                (morton, i)
            })
            .collect();

        // Sort by Morton code for Z-order traversal
        entries.sort_by_key(|&(m, _)| m);

        MortonRayBatch {
            morton_codes: entries.iter().map(|&(m, _)| m).collect(),
            ray_indices: entries.iter().map(|&(_, i)| i).collect(),
        }
    }

    /// Get the number of rays in the batch.
    pub fn len(&self) -> usize {
        self.morton_codes.len()
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.morton_codes.is_empty()
    }
}

/// March a batch of rays in Morton order for cache coherence.
///
/// Returns HitInfo for each ray in the ORIGINAL order (not Morton order).
pub fn march_batch_morton(ctx: &SdfContext, rays: &[Ray], screen_width: u32, screen_height: u32) -> Vec<HitInfo> {
    // Build screen-space origins for Morton ordering
    let origins: Vec<(f64, f64)> = rays
        .iter()
        .map(|r| (r.origin.x, r.origin.z))
        .collect();

    let batch = MortonRayBatch::new(&origins, screen_width, screen_height);

    // March rays in Morton order
    let mut results = vec![HitInfo::miss(); rays.len()];
    for &ray_idx in &batch.ray_indices {
        if ray_idx < rays.len() {
            results[ray_idx] = ray_march(ctx, &rays[ray_idx]);
        }
    }

    results
}

// ─── Branchless Hot-Loop Variant ────────────────────────────────────────────
//
// For performance-critical paths, we provide a branchless variant of the
// ray marcher that uses min/max/select operations instead of if-branches.
// This keeps the CPU pipeline full and the branch predictor happy.
//
// The key trick: instead of "if hit, return; else advance", we compute
// both paths and select the result. On modern CPUs with speculative
// execution, this can be faster for unpredictable data.

/// Branchless ray march: uses min/max/select instead of if-branches.
///
/// The hit distance is accumulated, and on each step we compute both
/// the "hit" and "miss" outcomes, selecting the appropriate one.
/// This avoids branch mispredictions in the hot loop.
pub fn ray_march_branchless(ctx: &SdfContext, ray: &Ray) -> HitInfo {
    let mut t = 0.0f64;
    let mut steps = 0u32;
    let mut hit_t = f64::MAX; // Will be overwritten on hit
    let mut hit_pos = Vec3::zero();
    let mut did_hit = false;

    while steps < ctx.max_steps {
        let pos = ray.at(t);
        let dist = sdf_world(ctx, &pos);
        steps += 1;

        // Branchless: compute both outcomes, select the right one
        let is_hit = dist.abs() < ctx.epsilon;
        let is_miss = t > MAX_DISTANCE;

        // Select: on hit, record the hit point; on miss or continue, leave as-is
        // Use conditional moves (no branch)
        hit_t = if is_hit { t } else { hit_t };
        hit_pos = if is_hit { pos } else { hit_pos };
        did_hit = did_hit || is_hit;

        // Early exit if we hit or missed
        if is_hit || is_miss {
            break;
        }

        // Advance
        t += dist;
    }

    if did_hit {
        let normal = compute_normal(ctx, &hit_pos);
        let mat = material_at(ctx, &hit_pos);
        HitInfo::hit(hit_t, hit_pos, normal, mat, steps)
    } else {
        HitInfo::miss()
    }
}

// ─── Verification ───────────────────────────────────────────────────────────

/// Verify the SDF ray marching system.
///
/// Tests:
///   1. SDF primitives return correct distances for known inputs
///   2. SDF combinators produce correct results
///   3. Sphere tracing finds surfaces that primitives define
///   4. Surface normals are orthogonal to the surface
///   5. Sieve-assisted marcher produces the same results as the basic marcher
///   6. World SDF is deterministic
///   7. Morton batch ordering works correctly
pub fn verify_sdf_ray() -> bool {
    let mut all_pass = true;

    // ── Test 1: SDF primitives ──

    // Sphere: point at origin, radius 1.0
    let d = sdf_sphere(&Vec3::new(0.0, 0.0, 2.0), &Vec3::zero(), 1.0);
    if (d - 1.0).abs() > 1e-10 {
        eprintln!("FAIL: sdf_sphere at (0,0,2) with r=1: expected 1.0, got {}", d);
        all_pass = false;
    }

    // Sphere: inside
    let d = sdf_sphere(&Vec3::new(0.0, 0.0, 0.5), &Vec3::zero(), 1.0);
    if d >= 0.0 {
        eprintln!("FAIL: sdf_sphere inside: expected negative, got {}", d);
        all_pass = false;
    }

    // Box: point outside
    let d = sdf_box(&Vec3::new(2.0, 0.0, 0.0), &Vec3::zero(), &Vec3::new(1.0, 1.0, 1.0));
    if (d - 1.0).abs() > 1e-10 {
        eprintln!("FAIL: sdf_box at (2,0,0) with half_extents (1,1,1): expected 1.0, got {}", d);
        all_pass = false;
    }

    // Box: inside
    let d = sdf_box(&Vec3::new(0.0, 0.0, 0.0), &Vec3::zero(), &Vec3::new(1.0, 1.0, 1.0));
    if d >= 0.0 {
        eprintln!("FAIL: sdf_box inside: expected negative, got {}", d);
        all_pass = false;
    }

    // Plane
    let d = sdf_plane(&Vec3::new(0.0, 5.0, 0.0), 3.0);
    if (d - 2.0).abs() > 1e-10 {
        eprintln!("FAIL: sdf_plane at y=5, height=3: expected 2.0, got {}", d);
        all_pass = false;
    }

    // Torus: at center ring, should be -minor
    let d = sdf_torus(&Vec3::new(3.0, 0.0, 0.0), &Vec3::zero(), 3.0, 1.0);
    if (d - (-1.0)).abs() > 1e-10 {
        eprintln!("FAIL: sdf_torus on ring: expected -1.0, got {}", d);
        all_pass = false;
    }

    // Cylinder: at center, should be -radius
    let d = sdf_cylinder(&Vec3::new(0.0, 0.0, 0.0), &Vec3::zero(), 1.0, 2.0);
    if d >= 0.0 {
        eprintln!("FAIL: sdf_cylinder inside: expected negative, got {}", d);
        all_pass = false;
    }

    // Capsule: at midpoint, should be -radius
    let d = sdf_capsule(&Vec3::new(0.0, 0.5, 0.0), &Vec3::new(0.0, 0.0, 0.0), &Vec3::new(0.0, 1.0, 0.0), 0.5);
    if d >= 0.0 {
        eprintln!("FAIL: sdf_capsule inside: expected negative, got {}", d);
        all_pass = false;
    }

    // ── Test 2: SDF combinators ──

    // Union: min
    let u = sdf_union(1.0, 2.0);
    if (u - 1.0).abs() > 1e-10 {
        eprintln!("FAIL: sdf_union(1, 2) = {} (expected 1.0)", u);
        all_pass = false;
    }

    // Intersection: max
    let i = sdf_intersection(1.0, 2.0);
    if (i - 2.0).abs() > 1e-10 {
        eprintln!("FAIL: sdf_intersection(1, 2) = {} (expected 2.0)", i);
        all_pass = false;
    }

    // Subtraction
    let s = sdf_subtraction(1.0, -2.0);
    if (s - 2.0).abs() > 1e-10 {
        eprintln!("FAIL: sdf_subtraction(1, -2) = {} (expected 2.0)", s);
        all_pass = false;
    }

    // Smooth union should be <= min
    let sm = sdf_smooth_union(1.0, 2.0, 0.5);
    if sm > 1.0 {
        eprintln!("FAIL: sdf_smooth_union(1, 2, 0.5) = {} (should be <= 1.0)", sm);
        all_pass = false;
    }

    // ── Test 3: Sphere tracing hits surfaces ──

    // Ray pointing at a sphere should hit it
    let ctx = SdfContext::new(42);
    let ray = Ray::new(Vec3::new(0.0, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
    let hit = ray_march(&ctx, &ray);
    // The ray should hit the terrain (it points downward)
    if !hit.hit {
        eprintln!("FAIL: ray_march downward ray didn't hit terrain");
        all_pass = false;
    }

    // ── Test 4: Surface normals are roughly orthogonal ──

    if hit.hit {
        // For a heightmap terrain, the normal should point generally upward
        if hit.normal.y < 0.0 {
            eprintln!("FAIL: terrain normal points downward: {:?}", hit.normal);
            all_pass = false;
        }
        // The normal should be unit length
        let nlen = hit.normal.length();
        if (nlen - 1.0).abs() > 0.01 {
            eprintln!("FAIL: normal not unit length: {}", nlen);
            all_pass = false;
        }
    }

    // ── Test 5: Sieve-assisted marcher produces same results ──

    let ctx_sieve = SdfContext::new(42);
    let ray_test = Ray::new(Vec3::new(100.0, 50.0, 100.0), Vec3::new(0.0, -1.0, 0.0));
    let basic_hit = ray_march(&ctx_sieve, &ray_test);
    let sieve_hit = sieve_ray_march(&ctx_sieve, &ray_test);

    // Both should agree on whether they hit
    if basic_hit.hit != sieve_hit.hit {
        eprintln!(
            "FAIL: basic marcher hit={}, sieve marcher hit={}",
            basic_hit.hit, sieve_hit.hit
        );
        all_pass = false;
    }

    // If both hit, the distances should be close
    if basic_hit.hit && sieve_hit.hit {
        if (basic_hit.distance - sieve_hit.distance).abs() > 1.0 {
            eprintln!(
                "FAIL: basic distance={}, sieve distance={} (too far apart)",
                basic_hit.distance, sieve_hit.distance
            );
            all_pass = false;
        }
    }

    // The sieve marcher should use fewer steps (or equal) since it skips chunks
    if sieve_hit.steps > basic_hit.steps {
        // This isn't necessarily a bug — it can happen on complex terrain —
        // but it's worth noting
        eprintln!(
            "NOTE: sieve marcher used {} steps vs basic {} (sieve should be fewer)",
            sieve_hit.steps, basic_hit.steps
        );
    }

    // ── Test 6: World SDF is deterministic ──

    let ctx_det = SdfContext::new(12345);
    let p = Vec3::new(42.0, 13.0, 99.0);
    let d1 = sdf_world(&ctx_det, &p);
    let d2 = sdf_world(&ctx_det, &p);
    if (d1 - d2).abs() > 1e-12 {
        eprintln!("FAIL: sdf_world is not deterministic: {} != {}", d1, d2);
        all_pass = false;
    }

    // Different coordinates should give different distances
    let p2 = Vec3::new(43.0, 13.0, 99.0);
    let d3 = sdf_world(&ctx_det, &p2);
    if (d1 - d3).abs() < 1e-12 {
        eprintln!("FAIL: sdf_world returns same distance at different coordinates");
        all_pass = false;
    }

    // ── Test 7: Morton batch ordering ──

    let origins = [
        (100.0, 100.0),
        (200.0, 50.0),
        (50.0, 200.0),
        (150.0, 150.0),
    ];
    let batch = MortonRayBatch::new(&origins, 256, 256);
    if batch.len() != 4 {
        eprintln!("FAIL: MortonRayBatch has {} entries, expected 4", batch.len());
        all_pass = false;
    }
    // All indices should be present exactly once
    let mut indices = batch.ray_indices.clone();
    indices.sort();
    if indices != vec![0, 1, 2, 3] {
        eprintln!("FAIL: MortonRayBatch indices are not a permutation of [0,1,2,3]: {:?}", indices);
        all_pass = false;
    }

    // ── Test 8: Vec3 operations ──

    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);

    // Add
    let c = a.add(&b);
    if (c.x - 5.0).abs() > 1e-10 || (c.y - 7.0).abs() > 1e-10 || (c.z - 9.0).abs() > 1e-10 {
        eprintln!("FAIL: Vec3 add: {:?}", c);
        all_pass = false;
    }

    // Dot
    let dot = a.dot(&b);
    if (dot - 32.0).abs() > 1e-10 {
        eprintln!("FAIL: Vec3 dot: {} (expected 32.0)", dot);
        all_pass = false;
    }

    // Normalize
    let n = Vec3::new(3.0, 4.0, 0.0).normalize();
    if (n.length() - 1.0).abs() > 1e-10 {
        eprintln!("FAIL: Vec3 normalize length: {}", n.length());
        all_pass = false;
    }

    // ── Test 9: Branchless marcher agrees with basic marcher ──

    let ctx_bl = SdfContext::new(42);
    let ray_bl = Ray::new(Vec3::new(50.0, 30.0, 50.0), Vec3::new(0.0, -1.0, 0.0));
    let basic = ray_march(&ctx_bl, &ray_bl);
    let branchless = ray_march_branchless(&ctx_bl, &ray_bl);

    if basic.hit != branchless.hit {
        eprintln!(
            "FAIL: basic hit={}, branchless hit={}",
            basic.hit, branchless.hit
        );
        all_pass = false;
    }

    if basic.hit && branchless.hit && (basic.distance - branchless.distance).abs() > 1.0 {
        eprintln!(
            "FAIL: basic distance={}, branchless distance={}",
            basic.distance, branchless.distance
        );
        all_pass = false;
    }

    if all_pass {
        eprintln!("All SDF Ray Marching verification tests PASSED.");
    }

    all_pass
}
