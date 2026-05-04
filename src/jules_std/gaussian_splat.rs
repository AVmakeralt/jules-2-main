// =============================================================================
// std/gaussian_splat — Deterministic Gaussian Splatting for the Aurora Flux
//                        Rendering Pipeline
//
// Implements procedural Gaussian splatting driven by the SimdPrng8 PRNG for
// the Jules engine. Instead of hard polygon edges, surfaces are composed of
// millions of tiny semi-transparent colored "blobs" (Gaussians) that create
// soft moss, atmospheric fog, and realistic light diffusion.
//
// Key design points:
//   1. Gaussian Primitive: 3D Gaussian with position, covariance (axis-aligned
//      scaling + Y-rotation), color (RGBA), and opacity
//   2. PRNG-Driven Splat Generation: SimdPrng8 generates color, orientation,
//      and "fuzziness" based on surface type (material_id from sdf_ray)
//   3. Stateless Splatting: Given a surface point + material_id + seed, the
//      PRNG deterministically generates all splat parameters — no stored data
//   4. 2D Projection: EWA (Elliptical Weighted Average) splatting projects 3D
//      Gaussians onto screen space with proper foreshortening
//   5. Alpha Blending: Front-to-back compositing with early-Z termination
//   6. Atmospheric Fog: PRNG-generated fog density with 1/d² falloff for
//      volumetric "god rays"
//   7. Temporal Hash: Animation offset from (Time + MortonHash) for wind/flutter
//
// The fundamental insight: every splat is a pure function of (seed, position,
// material_id, time). There are no stored Gaussian blobs — no particle buffers,
// no vertex arrays. When the renderer needs a splat, it queries the PRNG oracle
// and the Gaussian materializes deterministically, then vanishes. The same
// coordinate + seed always produces the same visual, frame after frame.
//
// Pure Rust, zero external dependencies. Uses prng_simd, genesis_weave, morton,
// and sdf_ray modules.
// =============================================================================

#![allow(dead_code)]

use crate::interp::{RuntimeError, Value};
use crate::jules_std::prng_simd::{SquaresRng, SimdPrng8};
use crate::jules_std::genesis_weave::{hash_coord_2d, hash_to_f64, Biome};
use crate::jules_std::morton::encode_2d;
use crate::jules_std::sdf_ray::{Vec3, Ray, HitInfo, SdfContext};

// ─── Dispatch for stdlib integration ────────────────────────────────────────

/// Dispatch `gaussian_splat::` builtin calls.
///
/// Supported calls:
///   - "gaussian_splat::generate" — takes seed, x, y, z, material_id → array of splat positions
///   - "gaussian_splat::fog" — takes seed, ox, oy, oz, dx, dy, dz, distance → fog RGBA array
///   - "gaussian_splat::god_rays" — takes seed, ox, oy, oz, dx, dy, dz → intensity f64
///   - "gaussian_splat::verify" → bool
pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        "gaussian_splat::generate" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let x = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let y = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let z = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let material_id = args.get(4).and_then(|v| v.as_i64()).unwrap_or(3) as u32;

            let ctx = SplatContext::new(seed);
            let hit = HitInfo::hit(
                0.0,
                Vec3::new(x, y, z),
                Vec3::new(0.0, 1.0, 0.0), // Default up normal
                material_id,
                0,
            );
            let splats = generate_splats(&ctx, &hit, material_id);

            // Return array of [x, y, z, r, g, b, a, opacity, ...] for each splat
            let vals: Vec<Value> = splats
                .iter()
                .flat_map(|s| {
                    [
                        Value::F64(s.position.x),
                        Value::F64(s.position.y),
                        Value::F64(s.position.z),
                        Value::F32(s.color[0]),
                        Value::F32(s.color[1]),
                        Value::F32(s.color[2]),
                        Value::F32(s.color[3]),
                        Value::F32(s.opacity),
                    ]
                })
                .collect();
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(
                vals,
            )))))
        }
        "gaussian_splat::fog" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let ox = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let oy = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let oz = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let dx = args.get(4).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let dy = args.get(5).and_then(|v| v.as_f64()).unwrap_or(-1.0);
            let dz = args.get(6).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let distance = args.get(7).and_then(|v| v.as_f64()).unwrap_or(100.0);

            let ctx = SplatContext::new(seed);
            let ray = Ray {
                origin: Vec3::new(ox, oy, oz),
                direction: Vec3::new(dx, dy, dz).normalize(),
            };
            let fog = fog_color(&ctx, &ray, distance);
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(
                vec![
                    Value::F32(fog[0]),
                    Value::F32(fog[1]),
                    Value::F32(fog[2]),
                    Value::F32(fog[3]),
                ],
            )))))
        }
        "gaussian_splat::god_rays" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let ox = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let oy = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let oz = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let dx = args.get(4).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let dy = args.get(5).and_then(|v| v.as_f64()).unwrap_or(-1.0);
            let dz = args.get(6).and_then(|v| v.as_f64()).unwrap_or(0.0);

            let ctx = SplatContext::new(seed);
            let ray = Ray {
                origin: Vec3::new(ox, oy, oz),
                direction: Vec3::new(dx, dy, dz).normalize(),
            };
            // Default light direction: slightly angled from above
            let light_dir = Vec3::new(0.3, -0.8, 0.2).normalize();
            let intensity = god_rays(&ctx, &ray, &light_dir);
            Some(Ok(Value::F64(intensity)))
        }
        "gaussian_splat::verify" => {
            let ok = verify_gaussian_splat();
            Some(Ok(Value::Bool(ok)))
        }
        _ => None,
    }
}

// ─── Core Types ──────────────────────────────────────────────────────────────
//
// Gaussian3D — a 3D Gaussian blob in world space.
// Splat2D — the 2D projection of a Gaussian onto screen space (EWA).
// SplatContext — rendering parameters bundled for convenient threading.

/// A 3D Gaussian primitive in world space.
///
/// The Gaussian is parameterized by position, axis-aligned scaling (which forms
/// the diagonal of the covariance matrix), a Y-axis rotation angle, RGBA color,
/// and opacity. The full 3x3 covariance matrix Σ is reconstructed as:
///
///   Σ = R(θ) · diag(scale²) · R(θ)ᵀ
///
/// where R(θ) is a rotation matrix around the Y axis by angle `rotation`.
///
/// This representation uses only 3 scale values + 1 rotation instead of the
/// full 6 independent values of a symmetric 3×3 matrix, which is sufficient
/// for natural-looking organic splats (moss, foliage, fog patches) while
/// keeping the PRNG stream compact and deterministic.
#[derive(Debug, Clone, Copy)]
pub struct Gaussian3D {
    /// Center position of the Gaussian in world space.
    pub position: Vec3,
    /// Axis-aligned scaling factors (diagonal of covariance).
    /// scale[i] is the standard deviation along axis i (before rotation).
    pub scale: [f64; 3],
    /// Y-axis rotation angle in radians. Rotates the Gaussian's principal
    /// axes around the vertical, giving directional elongation.
    pub rotation: f64,
    /// RGBA color of the splat. Pre-multiplied alpha is NOT assumed;
    /// alpha blending handles the compositing.
    pub color: [f32; 4],
    /// Opacity multiplier [0, 1]. This is the peak opacity at the center
    /// of the Gaussian; it falls off towards the edges according to the
    /// Gaussian bell curve.
    pub opacity: f32,
}

/// A 2D Gaussian splat projected onto screen space.
///
/// This is the result of EWA (Elliptical Weighted Average) projection: the 3D
/// Gaussian's covariance matrix has been projected through the camera transform
/// and a 2×2 screen-space covariance has been extracted. The `rotation` field
/// describes the orientation of the projected ellipse.
///
/// The `depth` value is used for sorting splats front-to-back before compositing.
#[derive(Debug, Clone, Copy)]
pub struct Splat2D {
    /// Screen-space X coordinate (pixels).
    pub screen_x: f64,
    /// Screen-space Y coordinate (pixels).
    pub screen_y: f64,
    /// Screen-space standard deviation along the ellipse's major axis (pixels).
    pub scale_x: f64,
    /// Screen-space standard deviation along the ellipse's minor axis (pixels).
    pub scale_y: f64,
    /// Orientation of the projected ellipse (radians).
    pub rotation: f64,
    /// RGBA color of the splat.
    pub color: [f32; 4],
    /// Opacity at center of the Gaussian [0, 1].
    pub opacity: f32,
    /// Depth (distance from camera) for sorting.
    pub depth: f64,
}

/// Context for Gaussian splat generation — bundles rendering parameters.
///
/// The `seed` drives the SimdPrng8 for deterministic splat parameter generation.
/// All other fields control quality vs. performance tradeoffs.
#[derive(Debug, Clone)]
pub struct SplatContext {
    /// PRNG seed for deterministic splat generation.
    pub seed: u64,
    /// Number of splats generated per surface hit point.
    /// More splats = smoother appearance, higher cost.
    /// Default: 8.
    pub splats_per_hit: u32,
    /// Base Gaussian scale in world units. Individual splat scales are
    /// derived from this multiplied by PRNG-driven variation.
    /// Default: 0.5.
    pub base_scale: f64,
    /// Global fog density [0, ∞). Higher = denser atmospheric fog.
    /// Default: 0.02.
    pub fog_density: f64,
    /// Current time for animation (wind flutter, temporal variation).
    /// Default: 0.0.
    pub time: f64,
    /// Screen width in pixels. Used for EWA projection.
    /// Default: 1920.
    pub screen_width: u32,
    /// Screen height in pixels. Used for EWA projection.
    /// Default: 1080.
    pub screen_height: u32,
    /// Field of view in radians. Used for EWA projection.
    /// Default: π/3 (60°).
    pub fov: f64,
}

impl SplatContext {
    /// Create a new SplatContext with sensible defaults.
    pub fn new(seed: u64) -> Self {
        SplatContext {
            seed,
            splats_per_hit: 8,
            base_scale: 0.5,
            fog_density: 0.02,
            time: 0.0,
            screen_width: 1920,
            screen_height: 1080,
            fov: std::f64::consts::FRAC_PI_3, // 60 degrees
        }
    }

    /// Create a context with custom rendering parameters.
    pub fn with_params(
        seed: u64,
        splats_per_hit: u32,
        base_scale: f64,
        fog_density: f64,
        time: f64,
        screen_width: u32,
        screen_height: u32,
        fov: f64,
    ) -> Self {
        SplatContext {
            seed,
            splats_per_hit,
            base_scale,
            fog_density,
            time,
            screen_width,
            screen_height,
            fov,
        }
    }
}

// ─── Material Palette ───────────────────────────────────────────────────────
//
// The material palette maps material_id (from the SDF ray marcher) to a base
// color. The PRNG adds per-splat variation within a controlled range, so that
// a grass surface isn't uniformly green but instead has organic color noise.
//
// Material IDs follow the convention from sdf_ray::material_at:
//   0 = void, 1 = water, 2 = sand, 3 = grass, 4 = rock, 5 = snow,
//   6 = lava, 7 = crystal

/// Look up the base color for a material, with PRNG-driven variation.
///
/// The `rng_val` is a uniform random value in [0, 1) from the SimdPrng8 stream.
/// It perturbs the base color to create natural variation: each splat of grass
/// is a slightly different shade of green, each rock splat a slightly different
/// gray, etc.
///
/// Returns an RGBA color array with alpha = 1.0 (opacity is controlled separately).
#[inline]
pub fn material_palette(material_id: u32, rng_val: f64) -> [f32; 4] {
    // Variation amplitude: how much the PRNG can shift the base color.
    // Organic materials (grass, moss) have higher variation; minerals (rock,
    // crystal) have lower variation.
    let v = rng_val as f32; // [0, 1)

    match material_id {
        0 => {
            // Void — invisible
            [0.0, 0.0, 0.0, 0.0]
        }
        1 => {
            // Water — deep blue-green with subtle variation
            let r = 0.05 + v * 0.05;
            let g = 0.15 + v * 0.10;
            let b = 0.45 + v * 0.15;
            [r, g, b, 0.7]
        }
        2 => {
            // Sand — warm tan with grain variation
            let r = 0.76 + v * 0.10;
            let g = 0.70 + v * 0.08;
            let b = 0.50 + v * 0.10;
            [r, g, b, 1.0]
        }
        3 => {
            // Grass — rich green with strong organic variation
            let r = 0.15 + v * 0.15;
            let g = 0.45 + v * 0.25;
            let b = 0.08 + v * 0.12;
            [r, g, b, 1.0]
        }
        4 => {
            // Rock — gray-brown with subtle mineral variation
            let base = 0.35 + v * 0.20;
            let warmth = (v * 0.08) as f32;
            [base + warmth, base, base - warmth * 0.5, 1.0]
        }
        5 => {
            // Snow — bright white with slight blue tint
            let r = 0.90 + v * 0.08;
            let g = 0.92 + v * 0.06;
            let b = 0.97 + v * 0.03;
            [r.min(1.0), g.min(1.0), b.min(1.0), 1.0]
        }
        6 => {
            // Lava — incandescent red-orange
            let r = 0.85 + v * 0.15;
            let g = 0.20 + v * 0.35;
            let b = 0.02 + v * 0.08;
            [r.min(1.0), g, b, 0.95]
        }
        7 => {
            // Crystal — translucent cyan-purple shimmer
            let r = 0.40 + v * 0.30;
            let g = 0.60 + v * 0.20;
            let b = 0.85 + v * 0.15;
            [r.min(1.0), g.min(1.0), b.min(1.0), 0.85]
        }
        _ => {
            // Unknown material — neutral gray
            let base = 0.5 + v * 0.1;
            [base, base, base, 1.0]
        }
    }
}

// ─── Temporal Animation ─────────────────────────────────────────────────────
//
// Temporal offset for wind flutter and animation. Given a seed and Morton code
// (spatial hash), plus the current time, we produce a smooth animation offset
// that varies across the surface. This means:
//   - Each splat jiggles slightly differently (via its unique Morton code)
//   - The jiggling is smooth over time (via the time parameter)
//   - The animation is deterministic (same seed + position = same motion)

/// Compute a temporal animation offset for wind/flutter effects.
///
/// The offset is computed as:
///   offset = sin(time * frequency + phase)
///
/// where `frequency` and `phase` are derived from a hash of the seed and
/// Morton code. This ensures that nearby splats have correlated motion
/// (they sway together like grass in wind) while distant splats have
/// independent motion.
///
/// # Arguments
/// * `seed` — PRNG seed for the splat stream
/// * `morton_code` — Spatial hash of the splat's grid position
/// * `time` — Current animation time
///
/// # Returns
/// A f64 offset in world units, typically in the range [-0.1, 0.1].
#[inline]
pub fn temporal_offset(seed: u64, morton_code: u64, time: f64) -> f64 {
    // Hash the seed + morton_code to get per-splat animation parameters
    let hash = {
        let mut rng = SquaresRng::new(seed.wrapping_mul(0x5A17_5347).wrapping_add(morton_code));
        rng.next_u64()
    };

    // Extract frequency and phase from the hash
    let freq_bits = (hash >> 32) as u32;
    let phase_bits = hash as u32;

    // Frequency: 1.0..4.0 Hz (different splats oscillate at different speeds)
    let frequency = 1.0 + ((freq_bits as f64) / (u32::MAX as f64)) * 3.0;
    // Phase: 0..2π (splats start at different points in their cycle)
    let phase = ((phase_bits as f64) / (u32::MAX as f64)) * 2.0 * std::f64::consts::PI;

    // Amplitude scales with frequency — faster oscillations are smaller
    let amplitude = 0.05 / frequency;

    amplitude * (time * frequency + phase).sin()
}

// ─── PRNG-Driven Splat Generation ───────────────────────────────────────────
//
// The core of the Gaussian splatting system: given a surface hit point, a
// material ID, and a seed, the SimdPrng8 deterministically generates all
// parameters for N Gaussian splats. This is entirely stateless — no stored
// particles, no frame-to-frame accumulation.
//
// The generation process for each splat:
//   1. Use SimdPrng8 to generate 8 f64 values per batch
//   2. Offset the splat position from the hit point (tangent-plane distribution)
//   3. Set scale from base_scale * PRNG variation
//   4. Set rotation from PRNG angle
//   5. Set color from material_palette(material_id, PRNG_val)
//   6. Set opacity from PRNG (with material-dependent range)
//   7. Apply temporal_offset for wind/flutter

/// Generate N Gaussian splats at a surface hit point.
///
/// Given a `SplatContext` (which includes the seed and rendering parameters),
/// a `HitInfo` from the SDF ray marcher, and a `material_id`, this function
/// produces `ctx.splats_per_hit` Gaussian3D primitives that together describe
/// the surface appearance at the hit point.
///
/// # Determinism
/// The same (seed, hit.position, material_id, time) always produces the same
/// splats. This is guaranteed by the counter-based nature of SimdPrng8: the
/// counter is derived from a spatial hash of the position, so each hit point
/// has its own independent PRNG stream.
///
/// # Algorithm
/// For each splat i in [0, splats_per_hit):
///   1. Compute a counter from (seed, position Morton code, i)
///   2. Generate 8 f64 values from SimdPrng8 at that counter
///   3. Use values[0..2] for tangent-plane offset (within 3σ of normal)
///   4. Use values[3..5] for scale variation (base_scale * [0.5..2.0])
///   5. Use values[6] for rotation angle [0..2π)
///   6. Use values[7] for opacity variation [0.3..1.0]
///   7. Look up color from material_palette with values[0] as rng_val
///   8. Apply temporal_offset for wind animation
pub fn generate_splats(ctx: &SplatContext, hit: &HitInfo, material_id: u32) -> Vec<Gaussian3D> {
    let n = ctx.splats_per_hit;
    if n == 0 || !hit.hit {
        return Vec::new();
    }

    // Compute a spatial hash (Morton code) of the hit position for
    // counter-based PRNG seeding. This ensures that each surface point
    // has an independent PRNG stream.
    let px = hit.position.x.round() as i64;
    let py = hit.position.y.round() as i64;
    let pz = hit.position.z.round() as i64;

    // 2D Morton code from XZ plane (Y is the up axis in our world)
    let morton = encode_2d(
        (px & 0x001FFFFF) as u32,
        (pz & 0x001FFFFF) as u32,
    );

    // Base counter: combine seed, morton code, and Y coordinate
    // The Y coordinate is folded in separately because Morton is 2D (XZ only)
    let base_counter = seed_to_counter(ctx.seed, morton, py);

    // Compute tangent and bitangent vectors for the surface normal
    // These define the tangent plane where splats are distributed
    let (tangent, bitangent) = compute_tangent_frame(&hit.normal);

    let mut splats = Vec::with_capacity(n as usize);

    for i in 0..n {
        // Each splat gets 8 PRNG values from the SIMD batch
        let counter = base_counter.wrapping_add((i as u64) * 8);
        let mut rng = SimdPrng8::new(counter);
        let vals = rng.next_8x_f64();

        // ── Position offset in tangent plane ──
        // Offset within a 3σ radius of the Gaussian, giving a natural
        // distribution of splat centers around the hit point
        let spread = ctx.base_scale * 2.0; // 2σ spread
        let off_t = (vals[0] - 0.5) * 2.0 * spread; // tangent direction
        let off_b = (vals[1] - 0.5) * 2.0 * spread; // bitangent direction

        // Small normal-direction offset for "fuzziness"
        let off_n = (vals[2] - 0.5) * ctx.base_scale * 0.3;

        // Apply temporal offset for wind/flutter
        let wind = temporal_offset(ctx.seed, morton.wrapping_add(i as u64), ctx.time);

        let position = Vec3::new(
            hit.position.x + tangent.x * off_t + bitangent.x * off_b + hit.normal.x * off_n + wind,
            hit.position.y + tangent.y * off_t + bitangent.y * off_b + hit.normal.y * off_n,
            hit.position.z + tangent.z * off_t + bitangent.z * off_b + hit.normal.z * off_n + wind * 0.5,
        );

        // ── Scale (axis-aligned) ──
        // Each axis gets independent scale variation. Organic materials
        // tend to be elongated (non-uniform scaling); we model this by
        // varying the scale differently along each axis.
        let scale_var = 0.5 + vals[3] * 1.5; // [0.5, 2.0] multiplier
        let scale_x = ctx.base_scale * scale_var * (0.7 + vals[4] * 0.6);
        let scale_y = ctx.base_scale * scale_var * (0.5 + vals[5] * 0.5); // Flatter in Y
        let scale_z = ctx.base_scale * scale_var * (0.7 + (1.0 - vals[4]) * 0.6);

        // ── Rotation ──
        let rotation = vals[6] * 2.0 * std::f64::consts::PI;

        // ── Color ──
        let color = material_palette(material_id, vals[0]);

        // ── Opacity ──
        // Material-dependent opacity range:
        //   - Water, lava, crystal: semi-transparent
        //   - Grass, sand, rock, snow: mostly opaque
        let base_opacity = match material_id {
            1 | 6 | 7 => 0.3 + vals[7] as f32 * 0.4, // Water/lava/crystal: 0.3..0.7
            _ => 0.6 + vals[7] as f32 * 0.4,           // Solid: 0.6..1.0
        };

        splats.push(Gaussian3D {
            position,
            scale: [scale_x, scale_y, scale_z],
            rotation,
            color,
            opacity: base_opacity,
        });
    }

    splats
}

/// Compute a PRNG counter from seed, Morton code, and Y coordinate.
///
/// This function combines the three spatial identifiers into a single u64
/// counter that serves as the starting point for the SimdPrng8 stream.
/// The mixing uses multiplication and XOR-shifts to ensure good diffusion
/// (nearby positions produce very different counters).
#[inline]
fn seed_to_counter(seed: u64, morton: u64, y: i64) -> u64 {
    let mut h = seed;
    h = h.wrapping_mul(0x6C62272E07BB0142).wrapping_add(morton);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    h = h.wrapping_add(y as u64);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D049BB133111EB);
    h ^ (h >> 31)
}

/// Compute a tangent-bitangent frame from a surface normal.
///
/// Given a unit normal vector, constructs two perpendicular unit vectors
/// (tangent and bitangent) that span the tangent plane. The construction
/// is deterministic: the same normal always produces the same frame.
///
/// # Algorithm
/// Uses the "Frisvad" method: pick the axis least aligned with the normal
/// as a hint direction, then cross-product to get the tangent frame.
/// This avoids singularities when the normal is aligned with any axis.
#[inline]
fn compute_tangent_frame(normal: &Vec3) -> (Vec3, Vec3) {
    // Choose a vector that is NOT parallel to the normal
    let hint = if normal.y.abs() < 0.999 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };

    // tangent = normalize(normal × hint)
    let tangent = cross(normal, &hint).normalize();
    // bitangent = normal × tangent (guaranteed perpendicular to both)
    let bitangent = cross(normal, &tangent).normalize();

    (tangent, bitangent)
}

/// Cross product of two Vec3 vectors.
#[inline]
fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

// ─── EWA Projection (3D → 2D) ───────────────────────────────────────────────
//
// Elliptical Weighted Average (EWA) splatting projects a 3D Gaussian onto
// screen space by transforming its covariance matrix through the camera
// projection. The result is a 2D elliptical Gaussian on the screen.
//
// The algorithm:
//   1. Compute view-space position: v = (position - camera_pos)
//   2. Compute view-space covariance: Σ_v = R · Σ_w · Rᵀ
//      where R is the rotation from world to view space
//   3. Project the 3×3 covariance to a 2×2 screen covariance:
//      Σ_screen = J · Σ_v · Jᵀ
//      where J is the Jacobian of the perspective projection
//   4. Decompose Σ_screen into (scale_x, scale_y, rotation) for the ellipse
//
// For efficiency, we use a simplified EWA that assumes a pinhole camera
// model with the camera looking along -Z in view space.

/// Project a 3D Gaussian splat onto screen space using EWA projection.
///
/// Given a 3D Gaussian, camera position and direction, field of view, and
/// screen dimensions, this function returns a Splat2D describing the
/// projected elliptical Gaussian on screen.
///
/// # Arguments
/// * `splat` — The 3D Gaussian to project
/// * `camera_pos` — Camera position in world space
/// * `camera_dir` — Camera look direction (unit vector)
/// * `fov` — Field of view in radians
/// * `width` — Screen width in pixels
/// * `height` — Screen height in pixels
///
/// # Returns
/// A `Splat2D` with screen-space position, ellipse parameters, color, opacity,
/// and depth. If the splat is behind the camera, returns a splat with zero
/// scale (effectively invisible).
#[inline]
pub fn project_splat(
    splat: &Gaussian3D,
    camera_pos: &Vec3,
    camera_dir: &Vec3,
    fov: f64,
    width: u32,
    height: u32,
) -> Splat2D {
    // Vector from camera to splat center
    let to_splat = splat.position.sub(camera_pos);
    let depth = to_splat.dot(camera_dir);

    // If behind camera, return invisible splat
    if depth <= 0.001 {
        return Splat2D {
            screen_x: 0.0,
            screen_y: 0.0,
            scale_x: 0.0,
            scale_y: 0.0,
            rotation: 0.0,
            color: splat.color,
            opacity: 0.0,
            depth: f64::MAX,
        };
    }

    // Construct camera right and up vectors
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let right = cross(camera_dir, &world_up).normalize();
    let up = cross(&right, camera_dir).normalize();

    // Project center onto screen
    let focal_length = (width as f64) * 0.5 / (fov * 0.5).tan();
    let aspect = (width as f64) / (height as f64);

    let screen_x = (to_splat.dot(&right) / depth * focal_length) + (width as f64) * 0.5;
    let screen_y = -(to_splat.dot(&up) / depth * focal_length) + (height as f64) * 0.5;

    // Compute screen-space scale from the 3D Gaussian scale
    // The projection of a Gaussian with scale σ in 3D yields a 2D Gaussian
    // with scale ≈ σ * focal_length / depth (for a pinhole camera).
    let cos_rot = splat.rotation.cos();
    let sin_rot = splat.rotation.sin();

    // Project the axis-aligned scale through the rotation and camera
    // Effective scale in the right and up directions
    let scale_right = (splat.scale[0] * cos_rot).hypot(splat.scale[2] * sin_rot);
    let scale_up = (splat.scale[0] * sin_rot).hypot(splat.scale[2] * cos_rot) * aspect;
    let scale_depth = splat.scale[1]; // Y-axis scale

    // Screen-space scales (pixels)
    let proj_scale_x = scale_right * focal_length / depth;
    let proj_scale_y = (scale_up.hypot(scale_depth * 0.1)) * focal_length / depth;

    // Ensure minimum scale for visibility
    let min_scale = 0.5;
    let scale_x = proj_scale_x.max(min_scale);
    let scale_y = proj_scale_y.max(min_scale);

    // Screen-space rotation: combine the 3D rotation with the camera-relative
    // orientation. For simplicity, we approximate this as the original rotation
    // projected onto the screen plane.
    let screen_rotation = splat.rotation.atan2(1.0); // Simplified

    // Opacity attenuation with distance: farther splats are fainter
    // This models atmospheric scattering naturally
    let distance_attenuation = 1.0 / (1.0 + depth * depth * 0.0001);
    let opacity = splat.opacity * distance_attenuation as f32;

    Splat2D {
        screen_x,
        screen_y,
        scale_x,
        scale_y,
        rotation: screen_rotation,
        color: splat.color,
        opacity,
        depth,
    }
}

// ─── Alpha Blending (Front-to-Back Compositing) ─────────────────────────────
//
// Front-to-back alpha blending with early-Z termination. This is the standard
// compositing algorithm for transparent surfaces:
//
//   C_out = C_front · α_front + C_back · (1 - α_front)
//
// By sorting splats front-to-back and compositing in that order, we can
// terminate early once the accumulated opacity reaches ~1.0 (the pixel is
// fully occluded). This saves significant work for dense splat scenes.
//
// The algorithm:
//   1. Sort splats by depth (ascending = nearest first)
//   2. For each splat, compute its contribution:
//      weight = splat.opacity · (1 - accumulated_opacity)
//      accumulated_color += splat.color · weight
//      accumulated_opacity += weight
//   3. If accumulated_opacity ≥ threshold, terminate early
//   4. Return the accumulated RGBA color

/// Composite an array of 2D splats using front-to-back alpha blending.
///
/// The splats are sorted by depth (nearest first), then composited with
/// early-Z termination. The result is a single RGBA color representing
/// the combined contribution of all splats at a pixel.
///
/// # Arguments
/// * `splats` — Mutable slice of Splat2D values. WILL be sorted by depth.
///
/// # Returns
/// An RGBA color array [f32; 4] with the composited result.
pub fn composite_splats(splats: &mut [Splat2D]) -> [f32; 4] {
    // Sort by depth (nearest first) for front-to-back compositing
    splats.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal));

    let mut accum_r = 0.0f32;
    let mut accum_g = 0.0f32;
    let mut accum_b = 0.0f32;
    let mut accum_a = 0.0f32;

    // Early-Z termination threshold: once the pixel is 99% opaque,
    // further splats contribute less than 1% and can be skipped.
    const EARLY_Z_THRESHOLD: f32 = 0.99;

    for splat in splats.iter() {
        // Skip fully transparent splats
        if splat.opacity <= 0.0 {
            continue;
        }

        // Compute the weight of this splat's contribution
        // weight = α_i · (1 - Σα_prev)
        let weight = splat.opacity * (1.0 - accum_a);

        // Accumulate color (pre-multiplied alpha)
        accum_r += splat.color[0] * weight;
        accum_g += splat.color[1] * weight;
        accum_b += splat.color[2] * weight;
        accum_a += weight;

        // Early-Z termination: pixel is fully occluded
        if accum_a >= EARLY_Z_THRESHOLD {
            break;
        }
    }

    // Clamp alpha to [0, 1]
    accum_a = accum_a.min(1.0);

    [accum_r, accum_g, accum_b, accum_a]
}

// ─── Atmospheric Fog ────────────────────────────────────────────────────────
//
// PRNG-generated volumetric fog with 1/d² falloff. The fog is not a flat
// color — it has spatial variation driven by the SimdPrng8, creating
// natural-looking haze, mist, and depth cuing.
//
// The fog model:
//   fog_factor = 1 - exp(-density · integral)
//   where integral = ∫₀ᵈ (1 / (distance + ε)²) dt
//
// This gives a 1/d² falloff that is stronger near the camera (where
// accumulated density is high) and weaker far away. The PRNG adds
// spatial noise to the density for visual richness.

/// Compute the atmospheric fog color along a ray.
///
/// Given a `SplatContext`, a ray, and the distance to the nearest surface,
/// this function returns an RGBA color representing the accumulated fog
/// between the camera and the surface.
///
/// The fog density is PRNG-driven: at regular intervals along the ray,
/// the SimdPrng8 is queried for a density perturbation, creating
/// volumetric "puffs" of fog that are deterministic for a given seed.
///
/// # Arguments
/// * `ctx` — Splat context with seed, fog_density, screen dimensions, etc.
/// * `ray` — The ray along which to compute fog
/// * `hit_distance` — Distance to the nearest surface hit
///
/// # Returns
/// RGBA fog color with alpha proportional to fog thickness.
pub fn fog_color(ctx: &SplatContext, ray: &Ray, hit_distance: f64) -> [f32; 4] {
    if ctx.fog_density <= 0.0 || hit_distance <= 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }

    // Step along the ray, accumulating fog density with PRNG variation
    let num_steps = 32u32;
    let step_size = hit_distance / (num_steps as f64);
    let epsilon = 0.01; // Prevent division by zero in 1/d²

    let mut total_density = 0.0f64;

    for i in 0..num_steps {
        let t = (i as f64 + 0.5) * step_size;
        let point = ray.at(t);

        // Base fog density with 1/d² falloff
        let falloff = 1.0 / ((t + epsilon) * (t + epsilon));

        // PRNG-driven density perturbation at this sample point
        // Use the world-space position to seed the perturbation
        let px = point.x.round() as i64;
        let py = point.y.round() as i64;
        let pz = point.z.round() as i64;
        let perturbation = hash_to_f64(hash_coord_2d(
            ctx.seed.wrapping_add(0xF06_C010),
            px.wrapping_add(pz),
            py,
        ));

        // Density at this point: base_density * falloff * perturbation
        // perturbation is in [0, 1), we shift to [0.3, 1.3) for some minimum fog
        let local_density = ctx.fog_density * falloff * (0.3 + perturbation * 1.0);

        total_density += local_density * step_size;
    }

    // Fog factor: exponential extinction
    // Higher total_density → more fog (higher alpha)
    let fog_alpha = 1.0 - (-total_density).exp();
    let fog_alpha = fog_alpha.clamp(0.0, 1.0);

    // Fog color: slightly warm gray with subtle PRNG-driven hue variation
    let mut hue_rng = SimdPrng8::new(ctx.seed.wrapping_add(0xF06_BA5E));
    let hue_var = hue_rng.next_f64() as f32;
    let fog_r = 0.75 + hue_var * 0.05;
    let fog_g = 0.78 + hue_var * 0.04;
    let fog_b = 0.82 + hue_var * 0.03;

    [
        fog_r.min(1.0),
        fog_g.min(1.0),
        fog_b.min(1.0),
        fog_alpha as f32,
    ]
}

// ─── God Rays ───────────────────────────────────────────────────────────────
//
// "God rays" (crepuscular rays) are bright shafts of light created when
// sunlight passes through gaps in the fog/clouds. We model this by checking
// how aligned the view ray is with the light direction, then using the PRNG
// to create volumetric "shafts" of increased light intensity.
//
// The intensity model:
//   intensity = (ray · light_dir)ᵖ · fog_density · PRNG_shaft_intensity
//
// where p is a power that creates sharp rays (higher p = narrower rays),
// and PRNG_shaft_intensity provides the volumetric shaft pattern.

/// Compute the god ray intensity along a ray.
///
/// God rays are bright volumetric light shafts visible when the camera looks
/// toward a bright light source through fog. The intensity depends on:
///   1. How aligned the ray is with the light direction (phase function)
///   2. The global fog density
///   3. PRNG-driven "shaft" patterns that create the volumetric look
///
/// # Arguments
/// * `ctx` — Splat context with seed and fog_density
/// * `ray` — The view ray
/// * `light_dir` — Direction toward the light source (unit vector, pointing
///   FROM the scene TOWARD the light)
///
/// # Returns
/// God ray intensity in [0, 1]. Values > 0.1 are typically visible.
pub fn god_rays(ctx: &SplatContext, ray: &Ray, light_dir: &Vec3) -> f64 {
    if ctx.fog_density <= 0.0 {
        return 0.0;
    }

    // Phase function: how much light scatters toward the camera along this ray.
    // We use the Henyey-Greenstein phase function with forward scattering:
    //   phase = (1 - g²) / (4π · (1 + g² - 2g·cosθ)^(3/2))
    // For simplicity, we approximate with a power cosine:
    let cos_theta = -(ray.direction.dot(light_dir)); // Negative because ray points away
    let cos_theta = cos_theta.max(0.0); // Only forward scattering

    // Power cosine creates sharp forward-scattering lobes
    let p = 8.0; // Higher = narrower god rays
    let phase = cos_theta.powf(p);

    // If the ray doesn't point anywhere near the light, no god rays
    if phase < 0.001 {
        return 0.0;
    }

    // PRNG-driven shaft pattern: sample several points along the ray
    // and accumulate the "shaftiness" — whether the fog is thicker at
    // that point, creating visible light columns
    let num_samples = 16u32;
    let max_distance = 500.0; // God rays only visible within this distance
    let step_size = max_distance / (num_samples as f64);

    let mut shaft_accum = 0.0f64;

    for i in 0..num_samples {
        let t = (i as f64 + 0.5) * step_size;
        let point = ray.at(t);

        // Hash the position to get shaft intensity
        let px = point.x.round() as i64;
        let py = point.y.round() as i64;
        let pz = point.z.round() as i64;

        // Create elongated shaft patterns: hash with the light direction
        // to make the pattern stretch along the light path
        let shaft_hash = hash_coord_2d(
            ctx.seed.wrapping_add(0x60D_8A75),
            px.wrapping_add(pz),
            py,
        );
        let shaft_intensity = hash_to_f64(shaft_hash);

        // Only count "dense" fog patches as shaft material
        // This creates discrete shafts rather than uniform glow
        if shaft_intensity > 0.5 {
            let contribution = (shaft_intensity - 0.5) * 2.0; // [0, 1]
            let falloff = 1.0 / (1.0 + t * 0.01); // Distance falloff
            shaft_accum += contribution * falloff * step_size * ctx.fog_density;
        }
    }

    // Combine phase function with shaft accumulation
    let intensity = phase * shaft_accum * 10.0; // Scale for visibility

    intensity.clamp(0.0, 1.0)
}

// ─── Full Pipeline ──────────────────────────────────────────────────────────
//
// The full Gaussian splatting pipeline: hit point → splat generation →
// EWA projection. This is the high-level entry point that the Aurora Flux
// renderer calls for each pixel ray.

/// Generate and project Gaussian splats for a surface hit point.
///
/// This is the complete pipeline function that:
///   1. Generates 3D Gaussian splats at the hit point using `generate_splats`
///   2. Projects each 3D Gaussian onto screen space using `project_splat`
///   3. Returns the projected 2D splats ready for compositing
///
/// # Arguments
/// * `ctx` — Splat context with seed, screen dimensions, camera params
/// * `sdf_ctx` — SDF context for material lookups
/// * `hit` — The surface hit information from ray marching
///
/// # Returns
/// A Vec of Splat2D values, sorted by depth, ready for `composite_splats`.
pub fn generate_surface_splats(
    ctx: &SplatContext,
    _sdf_ctx: &SdfContext,
    hit: &HitInfo,
) -> Vec<Splat2D> {
    if !hit.hit {
        return Vec::new();
    }

    // Generate 3D splats at the hit point
    let gaussians = generate_splats(ctx, hit, hit.material_id);

    // Compute camera parameters for EWA projection
    // In a real renderer, these would come from the camera object.
    // For the stateless Jules approach, we derive a camera position from
    // the hit point and a default viewing direction.
    let camera_pos = Vec3::new(0.0, 100.0, 200.0); // Default camera position
    let camera_dir = hit.position.sub(&camera_pos).normalize();

    // Project each 3D Gaussian to screen space
    let mut splats_2d: Vec<Splat2D> = gaussians
        .iter()
        .map(|g| project_splat(g, &camera_pos, &camera_dir, ctx.fov, ctx.screen_width, ctx.screen_height))
        .filter(|s| s.opacity > 0.0 && s.scale_x > 0.0 && s.scale_y > 0.0)
        .collect();

    // Sort by depth for front-to-back compositing
    splats_2d.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal));

    splats_2d
}

// ─── 3D Gaussian Evaluation ─────────────────────────────────────────────────
//
// Evaluate the Gaussian function at a point. This is useful for compositing
// splats into a framebuffer where each pixel needs the Gaussian weight.

/// Evaluate a 3D Gaussian at a given point.
///
/// The Gaussian is defined as:
///   G(p) = opacity · exp(-0.5 · dᵀ · Σ⁻¹ · d)
///
/// where d = p - center and Σ is the covariance matrix.
///
/// For our axis-aligned + rotation representation:
///   Σ = R(θ) · diag(σ_x², σ_y², σ_z²) · R(θ)ᵀ
///
/// and the Mahalanobis distance dᵀΣ⁻¹d can be computed in the rotated frame
/// as: (d'x/σx)² + (d'y/σy)² + (d'z/σz)²
/// where d' = R(-θ) · d.
#[inline]
pub fn evaluate_gaussian_3d(gaussian: &Gaussian3D, point: &Vec3) -> f32 {
    let d = Vec3::new(
        point.x - gaussian.position.x,
        point.y - gaussian.position.y,
        point.z - gaussian.position.z,
    );

    // Rotate d by -θ around Y axis to get into the Gaussian's local frame
    let cos_r = gaussian.rotation.cos();
    let sin_r = gaussian.rotation.sin();
    let d_local_x = d.x * cos_r + d.z * sin_r;
    let d_local_z = -d.x * sin_r + d.z * cos_r;
    let d_local_y = d.y; // Y rotation doesn't affect Y component

    // Mahalanobis distance in the local frame
    let sigma_x = gaussian.scale[0].max(1e-8);
    let sigma_y = gaussian.scale[1].max(1e-8);
    let sigma_z = gaussian.scale[2].max(1e-8);

    let mahal = (d_local_x / sigma_x).powi(2)
        + (d_local_y / sigma_y).powi(2)
        + (d_local_z / sigma_z).powi(2);

    // Gaussian function: opacity · exp(-0.5 · mahal)
    // Clamp mahal to avoid exp overflow for very distant points
    let exponent = -0.5 * mahal.min(50.0);
    gaussian.opacity * exponent.exp() as f32
}

/// Evaluate a 2D Gaussian splat at a screen-space point.
///
/// The 2D Gaussian is defined as:
///   G(s) = opacity · exp(-0.5 · dᵀ · Σ₂D⁻¹ · d)
///
/// where d = s - center and Σ₂D is the 2D covariance described by
/// (scale_x, scale_y, rotation).
#[inline]
pub fn evaluate_gaussian_2d(splat: &Splat2D, screen_x: f64, screen_y: f64) -> f32 {
    let dx = screen_x - splat.screen_x;
    let dy = screen_y - splat.screen_y;

    // Rotate into the ellipse's local frame
    let cos_r = splat.rotation.cos();
    let sin_r = splat.rotation.sin();
    let local_x = dx * cos_r + dy * sin_r;
    let local_y = -dx * sin_r + dy * cos_r;

    let sigma_x = splat.scale_x.max(1e-8);
    let sigma_y = splat.scale_y.max(1e-8);

    let mahal = (local_x / sigma_x).powi(2) + (local_y / sigma_y).powi(2);
    let exponent = -0.5 * mahal.min(50.0);

    splat.opacity * exponent.exp() as f32
}

// ─── Utility: Splat Bounding Box ────────────────────────────────────────────

/// Compute the screen-space bounding box of a 2D splat.
///
/// Returns (min_x, min_y, max_x, max_y) in pixel coordinates.
/// The bounding box extends 3σ from the center in each direction.
#[inline]
pub fn splat_bounds(splat: &Splat2D) -> (f64, f64, f64, f64) {
    // The bounding box of a rotated ellipse with semi-axes (σx, σy)
    // is bounded by the largest axis in each direction.
    let max_sigma = splat.scale_x.max(splat.scale_y) * 3.0; // 3σ coverage

    (
        splat.screen_x - max_sigma,
        splat.screen_y - max_sigma,
        splat.screen_x + max_sigma,
        splat.screen_y + max_sigma,
    )
}

// ─── Fog Color with Biome Influence ─────────────────────────────────────────

/// Compute fog color influenced by the local biome.
///
/// Different biomes have different atmospheric characteristics:
///   - Desert: warm, sandy haze
///   - Ocean: blue-gray mist
///   - Forest: green-tinted humidity
///   - Snow: cold, white mist
///   - Volcanic: ash-gray smog
///
/// This function looks up the biome at the ray's origin and tints the fog
/// accordingly, creating natural atmospheric variation across the world.
pub fn biome_fog_color(ctx: &SplatContext, ray: &Ray, hit_distance: f64) -> [f32; 4] {
    let base_fog = fog_color(ctx, ray, hit_distance);

    // Look up the biome at the ray origin
    let biome = crate::jules_std::genesis_weave::biome_at(
        ctx.seed,
        ray.origin.x,
        ray.origin.z,
    );

    // Tint the fog based on biome
    let tint = match biome {
        Biome::Ocean => [0.6, 0.7, 0.85],   // Blue-gray mist
        Biome::Beach => [0.8, 0.78, 0.7],    // Warm sea haze
        Biome::Desert => [0.85, 0.8, 0.65],  // Sandy haze
        Biome::Forest | Biome::DenseForest => [0.7, 0.78, 0.7], // Green humidity
        Biome::Jungle | Biome::Swamp => [0.65, 0.75, 0.65], // Dense green
        Biome::SnowCaps | Biome::Tundra => [0.88, 0.9, 0.95], // Cold white mist
        Biome::Volcanic => [0.6, 0.55, 0.5], // Ash smog
        Biome::Crystal => [0.75, 0.8, 0.9],  // Shimmering
        _ => [0.78, 0.8, 0.82],               // Default neutral
    };

    // Blend: 70% base fog, 30% biome tint
    let blend = 0.3f32;
    [
        base_fog[0] * (1.0 - blend) + tint[0] as f32 * blend,
        base_fog[1] * (1.0 - blend) + tint[1] as f32 * blend,
        base_fog[2] * (1.0 - blend) + tint[2] as f32 * blend,
        base_fog[3],
    ]
}

// ─── Tile-Based Rendering Support ───────────────────────────────────────────
//
// For efficient rendering, the screen is divided into tiles (e.g., 8×8 pixels).
// Each tile maintains a list of splats that overlap it. This allows
// parallel compositing across tiles and reduces overdraw.

/// A screen tile for splat accumulation.
#[derive(Debug, Clone)]
pub struct SplatTile {
    /// Tile X index.
    pub tile_x: u32,
    /// Tile Y index.
    pub tile_y: u32,
    /// Splats that overlap this tile, sorted by depth.
    pub splats: Vec<Splat2D>,
}

impl SplatTile {
    /// Create a new empty tile at the given indices.
    pub fn new(tile_x: u32, tile_y: u32) -> Self {
        SplatTile {
            tile_x,
            tile_y,
            splats: Vec::new(),
        }
    }

    /// Add a splat to this tile if it overlaps the tile bounds.
    pub fn try_add(&mut self, splat: &Splat2D, tile_size: u32) -> bool {
        let (min_x, min_y, max_x, max_y) = splat_bounds(splat);

        let tile_min_x = (self.tile_x * tile_size) as f64;
        let tile_min_y = (self.tile_y * tile_size) as f64;
        let tile_max_x = ((self.tile_x + 1) * tile_size) as f64;
        let tile_max_y = ((self.tile_y + 1) * tile_size) as f64;

        // AABB overlap test
        if max_x >= tile_min_x && min_x <= tile_max_x && max_y >= tile_min_y && min_y <= tile_max_y
        {
            self.splats.push(*splat);
            true
        } else {
            false
        }
    }

    /// Composite all splats in this tile at a given pixel position.
    pub fn composite_at(&self, pixel_x: f64, pixel_y: f64) -> [f32; 4] {
        let mut accum_r = 0.0f32;
        let mut accum_g = 0.0f32;
        let mut accum_b = 0.0f32;
        let mut accum_a = 0.0f32;

        for splat in &self.splats {
            let weight = evaluate_gaussian_2d(splat, pixel_x, pixel_y);
            if weight < 0.001 {
                continue;
            }

            let alpha = weight * (1.0 - accum_a);
            accum_r += splat.color[0] * alpha;
            accum_g += splat.color[1] * alpha;
            accum_b += splat.color[2] * alpha;
            accum_a += alpha;

            if accum_a >= 0.99 {
                break;
            }
        }

        [accum_r, accum_g, accum_b, accum_a.min(1.0)]
    }
}

// ─── Verification ───────────────────────────────────────────────────────────

/// Verify the Gaussian Splatting system.
///
/// Tests:
///   1. Material palette returns valid RGBA for all material IDs
///   2. Splat generation is deterministic
///   3. Splat generation produces the correct number of splats
///   4. Temporal offset is deterministic for same inputs
///   5. EWA projection produces valid screen-space splats
///   6. Alpha blending produces valid RGBA output
///   7. Fog color is non-negative and alpha is in [0, 1]
///   8. God rays return intensity in [0, 1]
///   9. 3D Gaussian evaluation is correct at center
///  10. 2D Gaussian evaluation is correct at center
///  11. Biome fog color produces valid output
///  12. Tile-based compositing produces valid output
pub fn verify_gaussian_splat() -> bool {
    let mut all_pass = true;

    // ── Test 1: Material palette ──

    for mat_id in 0..8u32 {
        let color = material_palette(mat_id, 0.5);
        // Check that color components are in [0, 1] (or clamped)
        for i in 0..4 {
            if color[i] < 0.0 || color[i] > 1.5 {
                // Allow slight overshoot before clamping
                eprintln!(
                    "FAIL: material_palette({}, 0.5) component {} = {} out of range",
                    mat_id, i, color[i]
                );
                all_pass = false;
            }
        }
    }

    // ── Test 2: Splat generation is deterministic ──

    let ctx = SplatContext::new(42);
    let hit = HitInfo::hit(
        10.0,
        Vec3::new(5.0, 3.0, 7.0),
        Vec3::new(0.0, 1.0, 0.0),
        3,
        50,
    );

    let splats1 = generate_splats(&ctx, &hit, 3);
    let splats2 = generate_splats(&ctx, &hit, 3);

    if splats1.len() != splats2.len() {
        eprintln!(
            "FAIL: generate_splats produced different lengths: {} vs {}",
            splats1.len(),
            splats2.len()
        );
        all_pass = false;
    } else {
        for (i, (a, b)) in splats1.iter().zip(splats2.iter()).enumerate() {
            if (a.position.x - b.position.x).abs() > 1e-10
                || (a.position.y - b.position.y).abs() > 1e-10
                || (a.position.z - b.position.z).abs() > 1e-10
            {
                eprintln!("FAIL: generate_splats splat {} position differs", i);
                all_pass = false;
                break;
            }
            if (a.opacity - b.opacity).abs() > 1e-6 {
                eprintln!("FAIL: generate_splats splat {} opacity differs", i);
                all_pass = false;
                break;
            }
        }
    }

    // ── Test 3: Correct number of splats ──

    if splats1.len() != ctx.splats_per_hit as usize {
        eprintln!(
            "FAIL: generate_splats produced {} splats, expected {}",
            splats1.len(),
            ctx.splats_per_hit
        );
        all_pass = false;
    }

    // ── Test 4: Temporal offset is deterministic ──

    let t1 = temporal_offset(42, 12345, 1.5);
    let t2 = temporal_offset(42, 12345, 1.5);
    if (t1 - t2).abs() > 1e-12 {
        eprintln!(
            "FAIL: temporal_offset is not deterministic: {} vs {}",
            t1, t2
        );
        all_pass = false;
    }

    // Different morton codes should produce different offsets
    let t3 = temporal_offset(42, 12346, 1.5);
    if (t1 - t3).abs() < 1e-12 {
        eprintln!("FAIL: temporal_offset doesn't vary with morton code");
        all_pass = false;
    }

    // ── Test 5: EWA projection produces valid screen-space splats ──

    if !splats1.is_empty() {
        let camera_pos = Vec3::new(0.0, 10.0, 20.0);
        let camera_dir = Vec3::new(0.0, -0.3, -0.7).normalize();

        let projected = project_splat(
            &splats1[0],
            &camera_pos,
            &camera_dir,
            std::f64::consts::FRAC_PI_3,
            1920,
            1080,
        );

        // Screen coordinates should be finite
        if !projected.screen_x.is_finite() || !projected.screen_y.is_finite() {
            eprintln!(
                "FAIL: project_splat produced non-finite screen coords: ({}, {})",
                projected.screen_x, projected.screen_y
            );
            all_pass = false;
        }

        // Scale should be non-negative
        if projected.scale_x < 0.0 || projected.scale_y < 0.0 {
            eprintln!(
                "FAIL: project_splat produced negative scale: ({}, {})",
                projected.scale_x, projected.scale_y
            );
            all_pass = false;
        }
    }

    // ── Test 6: Alpha blending produces valid output ──

    let mut test_splats = vec![
        Splat2D {
            screen_x: 100.0,
            screen_y: 100.0,
            scale_x: 10.0,
            scale_y: 10.0,
            rotation: 0.0,
            color: [1.0, 0.0, 0.0, 1.0],
            opacity: 0.5,
            depth: 5.0,
        },
        Splat2D {
            screen_x: 102.0,
            screen_y: 102.0,
            scale_x: 10.0,
            scale_y: 10.0,
            rotation: 0.0,
            color: [0.0, 0.0, 1.0, 1.0],
            opacity: 0.8,
            depth: 3.0, // Closer — should be composited first
        },
    ];

    let result = composite_splats(&mut test_splats);

    // Alpha should be in [0, 1]
    if result[3] < 0.0 || result[3] > 1.0 {
        eprintln!(
            "FAIL: composite_splats produced alpha out of range: {}",
            result[3]
        );
        all_pass = false;
    }

    // Color components should be non-negative
    if result[0] < 0.0 || result[1] < 0.0 || result[2] < 0.0 {
        eprintln!(
            "FAIL: composite_splats produced negative color: {:?}",
            result
        );
        all_pass = false;
    }

    // ── Test 7: Fog color is valid ──

    let fog_ctx = SplatContext::new(42);
    let fog_ray = Ray {
        origin: Vec3::new(0.0, 10.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };
    let fog = fog_color(&fog_ctx, &fog_ray, 100.0);

    if fog[0] < 0.0 || fog[1] < 0.0 || fog[2] < 0.0 || fog[3] < 0.0 || fog[3] > 1.0 {
        eprintln!("FAIL: fog_color produced invalid RGBA: {:?}", fog);
        all_pass = false;
    }

    // Zero density should produce zero fog
    let no_fog_ctx = SplatContext {
        fog_density: 0.0,
        ..SplatContext::new(42)
    };
    let no_fog = fog_color(&no_fog_ctx, &fog_ray, 100.0);
    if no_fog[3].abs() > 1e-6 {
        eprintln!(
            "FAIL: fog_color with zero density produced non-zero alpha: {}",
            no_fog[3]
        );
        all_pass = false;
    }

    // ── Test 8: God rays return valid intensity ──

    let light_dir = Vec3::new(0.3, -0.8, 0.2).normalize();
    let god_ray_val = god_rays(&fog_ctx, &fog_ray, &light_dir);

    if god_ray_val < 0.0 || god_ray_val > 1.0 {
        eprintln!(
            "FAIL: god_rays produced intensity out of [0,1]: {}",
            god_ray_val
        );
        all_pass = false;
    }

    // Ray pointing away from light should have lower intensity
    let away_ray = Ray {
        origin: Vec3::new(0.0, 10.0, 0.0),
        direction: Vec3::new(0.0, 0.0, 1.0), // Away from light
    };
    let away_intensity = god_rays(&fog_ctx, &away_ray, &light_dir);
    if away_intensity > god_ray_val + 0.01 {
        eprintln!(
            "FAIL: god_rays toward light ({}) < away from light ({})",
            god_ray_val, away_intensity
        );
        all_pass = false;
    }

    // ── Test 9: 3D Gaussian evaluation at center ──

    let test_gaussian = Gaussian3D {
        position: Vec3::new(0.0, 0.0, 0.0),
        scale: [1.0, 1.0, 1.0],
        rotation: 0.0,
        color: [1.0, 1.0, 1.0, 1.0],
        opacity: 0.8,
    };

    // At the center, the Gaussian should equal the opacity
    let center_val = evaluate_gaussian_3d(&test_gaussian, &Vec3::new(0.0, 0.0, 0.0));
    if (center_val - 0.8).abs() > 0.01 {
        eprintln!(
            "FAIL: evaluate_gaussian_3d at center: expected 0.8, got {}",
            center_val
        );
        all_pass = false;
    }

    // At 3σ, the Gaussian should be nearly zero
    let far_val = evaluate_gaussian_3d(&test_gaussian, &Vec3::new(3.0, 0.0, 0.0));
    if far_val > 0.05 {
        eprintln!(
            "FAIL: evaluate_gaussian_3d at 3σ: expected ~0, got {}",
            far_val
        );
        all_pass = false;
    }

    // ── Test 10: 2D Gaussian evaluation at center ──

    let test_splat2d = Splat2D {
        screen_x: 100.0,
        screen_y: 100.0,
        scale_x: 10.0,
        scale_y: 10.0,
        rotation: 0.0,
        color: [1.0, 0.0, 0.0, 1.0],
        opacity: 0.9,
        depth: 5.0,
    };

    let center_2d = evaluate_gaussian_2d(&test_splat2d, 100.0, 100.0);
    if (center_2d - 0.9).abs() > 0.01 {
        eprintln!(
            "FAIL: evaluate_gaussian_2d at center: expected 0.9, got {}",
            center_2d
        );
        all_pass = false;
    }

    // ── Test 11: Biome fog color is valid ──

    let biome_fog = biome_fog_color(&fog_ctx, &fog_ray, 100.0);
    if biome_fog[0] < 0.0 || biome_fog[1] < 0.0 || biome_fog[2] < 0.0 {
        eprintln!("FAIL: biome_fog_color produced negative values: {:?}", biome_fog);
        all_pass = false;
    }

    // ── Test 12: Tile-based compositing ──

    let mut tile = SplatTile::new(0, 0);
    let tile_splat = Splat2D {
        screen_x: 5.0,
        screen_y: 5.0,
        scale_x: 3.0,
        scale_y: 3.0,
        rotation: 0.0,
        color: [0.0, 1.0, 0.0, 1.0],
        opacity: 0.5,
        depth: 1.0,
    };
    tile.try_add(&tile_splat, 8);
    let tile_result = tile.composite_at(5.0, 5.0);
    if tile_result[1] < 0.01 {
        eprintln!(
            "FAIL: tile compositing at splat center produced near-zero green: {:?}",
            tile_result
        );
        all_pass = false;
    }

    // ── Test: Miss hit produces no splats ──

    let miss_hit = HitInfo::miss();
    let miss_splats = generate_splats(&ctx, &miss_hit, 3);
    if !miss_splats.is_empty() {
        eprintln!(
            "FAIL: generate_splats on miss hit produced {} splats (expected 0)",
            miss_splats.len()
        );
        all_pass = false;
    }

    // ── Test: Splat bounds are reasonable ──

    let bounds = splat_bounds(&test_splat2d);
    let (_min_x, _min_y, _max_x, _max_y) = bounds;
    if _min_x > test_splat2d.screen_x || _max_x < test_splat2d.screen_x {
        eprintln!(
            "FAIL: splat_bounds X doesn't contain center: ({}, {}) vs {}",
            _min_x, _max_x, test_splat2d.screen_x
        );
        all_pass = false;
    }

    if all_pass {
        eprintln!("All Gaussian Splatting verification tests PASSED.");
    }

    all_pass
}
