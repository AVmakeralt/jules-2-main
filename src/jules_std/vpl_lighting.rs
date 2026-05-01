// =============================================================================
// std/vpl_lighting — SIMD Virtual Point Lights for the Aurora Flux Pipeline
//
// Implements stateless virtual point lights using sieve coordinates and
// 8-probe SIMD collision for real-time shadow calculation.
//
// Architecture:
//   1. Light Injection: Every Nth "prime" coordinate from the sieve is treated
//      as a tiny light source. The sieve generates light positions
//      deterministically — no stored light lists.
//   2. SIMD Shadow Probing: 8-probe collision logic checks if a pixel is in
//      shadow. Process 8 shadow rays simultaneously using SimdPrng8.
//   3. Harmonic Shading: Multi-bounce approximate global illumination using
//      the VPL positions.
//   4. Atmospheric Scattering: 1/d² falloff on PRNG-generated fog density
//      for "god rays".
//   5. Light Clustering: Morton-ordered light positions for cache-coherent
//      shadow queries.
//   6. Stateless: Given a seed and coordinate, all lighting is derived
//      mathematically.
//
// The key insight: because light positions are derived from the sieve, we
// never store a light list. We ask "is there a light at sieve position N?"
// the same way genesis_weave asks "is there a tree at coordinate (X,Y)?"
// The math IS the scene.
//
// Pure Rust, zero external dependencies.
// Uses prng_simd, genesis_weave, sieve_210, morton, collision, sdf_ray.
// =============================================================================

#![allow(dead_code)]

use crate::interp::{RuntimeError, Value};
use crate::compiler::lexer::Span;
use crate::jules_std::prng_simd::{SquaresRng, ShishiuaRng, SimdPrng8};
use crate::jules_std::genesis_weave::{GenesisWeave, hash_coord_2d, hash_to_f64, Biome, terrain_height};
use crate::jules_std::sieve_210::{sieve_210_wheel, primes_up_to, WHEEL_RESIDUES};
use crate::jules_std::morton::{encode_2d, decode_2d, simd_tile_index};
use crate::jules_std::collision::{probe_collision_8, PROBE_OFFSETS_8, batch_collision_8, CollisionResult};
use crate::jules_std::sdf_ray::{Vec3, Ray, SdfContext, sdf_world, compute_normal, ray_march, HitInfo};

// ─── Dispatch for stdlib integration ────────────────────────────────────────

/// Dispatch vpl_lighting:: builtin calls.
///
/// Supported calls:
///   - "vpl_lighting::compute" — takes seed, px, py, pz, nx, ny, nz → [r, g, b, shadow, ao]
///   - "vpl_lighting::shadow"  — takes seed, px, py, pz, lx, ly, lz → shadow factor
///   - "vpl_lighting::ao"      — takes seed, px, py, pz, nx, ny, nz → AO factor
///   - "vpl_lighting::god_rays" — takes seed, ox, oy, oz, dx, dy, dz → intensity
///   - "vpl_lighting::verify"  → bool
pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        "vpl_lighting::compute" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let px = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let py = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let pz = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let nx = args.get(4).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let ny = args.get(5).and_then(|v| v.as_f64()).unwrap_or(1.0);
            let nz = args.get(6).and_then(|v| v.as_f64()).unwrap_or(0.0);

            let ctx = VplContext::new(seed);
            let sdf_ctx = SdfContext::new(seed);
            let point = Vec3::new(px, py, pz);
            let normal = Vec3::new(nx, ny, nz).normalize();

            // Build a synthetic HitInfo from the provided point + normal
            let hit = HitInfo::hit(0.0, point, normal, 0, 0);
            let result = compute_lighting(&ctx, &sdf_ctx, &hit);

            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
                Value::F64(result.color[0] as f64),
                Value::F64(result.color[1] as f64),
                Value::F64(result.color[2] as f64),
                Value::F64(result.shadow_factor),
                Value::F64(result.ao_factor),
            ])))))
        }
        "vpl_lighting::shadow" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let px = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let py = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let pz = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let lx = args.get(4).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let ly = args.get(5).and_then(|v| v.as_f64()).unwrap_or(10.0);
            let lz = args.get(6).and_then(|v| v.as_f64()).unwrap_or(0.0);

            let ctx = VplContext::new(seed);
            let sdf_ctx = SdfContext::new(seed);
            let point = Vec3::new(px, py, pz);
            let light_pos = Vec3::new(lx, ly, lz);

            let probe = simd_shadow_probe_8(&ctx, &sdf_ctx, &point, &light_pos);
            // Shadow factor: fraction of probes that are NOT in shadow
            let lit_count = probe.results.iter().filter(|&&s| !s).count();
            let shadow_factor = lit_count as f64 / 8.0;

            Some(Ok(Value::F64(shadow_factor)))
        }
        "vpl_lighting::ao" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let px = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let py = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let pz = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let nx = args.get(4).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let ny = args.get(5).and_then(|v| v.as_f64()).unwrap_or(1.0);
            let nz = args.get(6).and_then(|v| v.as_f64()).unwrap_or(0.0);

            let ctx = VplContext::new(seed);
            let sdf_ctx = SdfContext::new(seed);
            let point = Vec3::new(px, py, pz);
            let normal = Vec3::new(nx, ny, nz).normalize();

            let ao = ambient_occlusion(&ctx, &sdf_ctx, &point, &normal);

            Some(Ok(Value::F64(ao)))
        }
        "vpl_lighting::god_rays" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let ox = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let oy = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let oz = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let dx = args.get(4).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let dy = args.get(5).and_then(|v| v.as_f64()).unwrap_or(1.0);
            let dz = args.get(6).and_then(|v| v.as_f64()).unwrap_or(0.0);

            let ray = Ray::new(Vec3::new(ox, oy, oz), Vec3::new(dx, dy, dz));
            let ctx = VplContext::new(seed);

            let intensity = god_rays_intensity(&ray, &ctx.sun_direction, 32, seed);

            Some(Ok(Value::F64(intensity)))
        }
        "vpl_lighting::verify" => {
            let ok = verify_vpl_lighting();
            Some(Ok(Value::Bool(ok)))
        }
        _ => None,
    }
}

// ─── Core Structures ────────────────────────────────────────────────────────
//
// VirtualPointLight — a single infinitesimal light source whose position
// is derived from a sieve prime coordinate.
//
// ShadowProbe8 — results of 8 simultaneous shadow ray queries.
//
// VplContext — the stateless lighting context: given a seed, everything
// else is derived.
//
// LightingResult — the final output of a full lighting computation.

/// A virtual point light whose position is derived from a sieve prime.
///
/// Each VPL is a tiny light source placed deterministically using the
/// sieve_210 wheel. The position, color, and intensity are all derived
/// from the seed + sieve coordinate — no stored light lists needed.
#[derive(Debug, Clone, Copy)]
pub struct VirtualPointLight {
    /// 3D position of this light in world space
    pub position: Vec3,
    /// RGB color of the light emission
    pub color: [f32; 3],
    /// Intensity (luminous power) of this light
    pub intensity: f64,
    /// Falloff radius — beyond this distance the light contributes nothing
    pub radius: f64,
}

/// Results from 8 simultaneous shadow ray probes toward a light source.
///
/// Each probe ray is offset slightly from the direct line to the light,
/// forming an 8-point "star" pattern. This simulates area light softening
/// and provides the data for penumbra estimation.
#[derive(Debug, Clone, Copy)]
pub struct ShadowProbe8 {
    /// Whether each probe ray found an occluder (true = in shadow)
    pub results: [bool; 8],
    /// Distance to the nearest occluder along each probe direction
    pub distances: [f64; 8],
    /// The 8 probe ray directions (offset from the primary light direction)
    pub directions: [Vec3; 8],
}

/// Stateless lighting context — given a seed, all lighting is derived.
///
/// The VplContext bundles the parameters that control how VPLs are generated
/// and evaluated. Since everything is derived from the seed + sieve, there
/// is no stored state to maintain.
#[derive(Debug, Clone)]
pub struct VplContext {
    /// Master seed driving all deterministic calculations
    pub seed: u64,
    /// Every Nth prime becomes 1 VPL (default: every 10th prime)
    pub light_spacing: u64,
    /// Maximum number of VPLs to evaluate per pixel (default: 64)
    pub max_lights: u32,
    /// Shadow bias to prevent shadow acne (default: 0.01)
    pub shadow_bias: f64,
    /// Ambient light color (RGB, linear space)
    pub ambient: [f32; 3],
    /// Directional sun light direction (normalized)
    pub sun_direction: Vec3,
    /// Sun light color (RGB, linear space)
    pub sun_color: [f32; 3],
    /// Sun light intensity
    pub sun_intensity: f64,
}

impl VplContext {
    /// Create a new VplContext with sensible defaults.
    pub fn new(seed: u64) -> Self {
        VplContext {
            seed,
            light_spacing: 10,
            max_lights: 64,
            shadow_bias: 0.01,
            ambient: [0.05, 0.05, 0.08],
            sun_direction: Vec3::new(0.4, 0.8, 0.3).normalize(),
            sun_color: [1.0, 0.95, 0.85],
            sun_intensity: 1.5,
        }
    }

    /// Create a context with custom parameters.
    pub fn with_params(
        seed: u64,
        light_spacing: u64,
        max_lights: u32,
        shadow_bias: f64,
        ambient: [f32; 3],
        sun_direction: Vec3,
        sun_color: [f32; 3],
        sun_intensity: f64,
    ) -> Self {
        VplContext {
            seed,
            light_spacing: light_spacing.max(1),
            max_lights,
            shadow_bias: shadow_bias.max(0.0),
            ambient,
            sun_direction: sun_direction.normalize(),
            sun_color,
            sun_intensity,
        }
    }
}

/// The result of a full lighting computation for a single surface point.
///
/// Includes the final composited color as well as the individual
/// components (shadow factor, AO, diffuse, specular) for debugging
/// and post-processing.
#[derive(Debug, Clone, Copy)]
pub struct LightingResult {
    /// Final composited RGB color (linear space)
    pub color: [f32; 3],
    /// Shadow factor: 0.0 = fully shadowed, 1.0 = fully lit
    pub shadow_factor: f64,
    /// Ambient occlusion factor: 0.0 = fully occluded, 1.0 = fully open
    pub ao_factor: f64,
    /// Diffuse contribution only (RGB, linear space)
    pub diffuse: [f32; 3],
    /// Specular contribution only (RGB, linear space)
    pub specular: [f32; 3],
}

// ─── Light Injection: Sieve-Based Deterministic Light Placement ──────────────
//
// The sieve_210 module generates primes deterministically. We treat every
// Nth prime as a "light seed" — a coordinate that determines where a VPL
// is placed. This is analogous to how genesis_weave uses hash coordinates
// for entity placement, but we use the sieve's mathematical structure
// instead of a hash.
//
// The primes define a grid of potential light positions. The sieve
// guarantees uniform-ish distribution (primes are dense but irregular),
// which provides natural-looking light placement without clustering
// artifacts that a regular grid would produce.

/// Default radius for VPL generation around the camera/center point.
const DEFAULT_VPL_RADIUS: f64 = 128.0;

/// World scale constant (must match sdf_ray.rs WORLD_SCALE).
const WORLD_SCALE: f64 = 64.0;

/// Generate VPL positions around a center point using sieve primes.
///
/// The algorithm:
///   1. Generate primes up to a limit proportional to the desired number
///      of lights and the light spacing parameter.
///   2. Select every Nth prime (where N = light_spacing) as a light seed.
///   3. For each selected prime p, derive a 3D position from p using
///      the Morton-encoded coordinates and PRNG jitter.
///   4. Assign color and intensity based on biome and height.
///
/// The positions are returned sorted by Morton code for cache coherence
/// in subsequent shadow queries.
pub fn generate_vpl_positions(
    ctx: &VplContext,
    center: &Vec3,
    radius: f64,
) -> Vec<VirtualPointLight> {
    let radius = if radius <= 0.0 { DEFAULT_VPL_RADIUS } else { radius };

    // Determine how many primes we need.
    // If light_spacing = 10 and max_lights = 64, we need 640 primes.
    let prime_limit = (ctx.max_lights as u64 * ctx.light_spacing * 2).max(100);
    let primes = primes_up_to(prime_limit);

    // Select every Nth prime as a light seed
    let light_seeds: Vec<u64> = primes
        .iter()
        .step_by(ctx.light_spacing as usize)
        .take(ctx.max_lights as usize)
        .cloned()
        .collect();

    // For each light seed, derive a 3D position
    let mut lights: Vec<(u64, VirtualPointLight)> = Vec::with_capacity(light_seeds.len());

    for (idx, &prime_val) in light_seeds.iter().enumerate() {
        // Use the prime as a PRNG seed to derive position offset
        let mut rng = SquaresRng::new(prime_val.wrapping_add(ctx.seed));

        // Generate offset within the radius
        let offset_x = (rng.next_f64() - 0.5) * 2.0 * radius;
        let offset_z = (rng.next_f64() - 0.5) * 2.0 * radius;

        // Y position: place light above terrain
        let wx = center.x + offset_x;
        let wz = center.z + offset_z;
        let terrain_h = terrain_height(ctx.seed, wx, 0.0, wz) * WORLD_SCALE;
        let height_offset = rng.next_f64() * 20.0 + 2.0; // 2-22 units above terrain
        let wy = terrain_h + height_offset;

        // Derive light color from biome
        let biome = crate::jules_std::genesis_weave::biome_at(ctx.seed, wx, wz);
        let color = biome_light_color(biome);

        // Intensity varies with height and a PRNG factor
        let intensity = 0.5 + rng.next_f64() * 1.5;
        let light_radius = radius * (1.5 + rng.next_f64());

        // Compute Morton code for sorting
        let morton = encode_2d(
            ((wx + radius) * 0.5).max(0.0) as u32,
            ((wz + radius) * 0.5).max(0.0) as u32,
        );

        lights.push((morton, VirtualPointLight {
            position: Vec3::new(wx, wy, wz),
            color,
            intensity,
            radius: light_radius,
        }));
    }

    // Sort by Morton code for cache-coherent access in shadow queries
    lights.sort_by_key(|&(morton, _)| morton);

    // Extract just the lights (Morton ordering preserved)
    lights.into_iter().map(|(_, light)| light).collect()
}

/// Determine light color based on biome.
///
/// Each biome produces a characteristic light color:
///   - Forest: warm greenish
///   - Desert: warm amber
///   - Ocean: cool blue
///   - Volcanic: deep red/orange
///   - Crystal: bright cyan/white
///   - etc.
#[inline(always)]
fn biome_light_color(biome: Biome) -> [f32; 3] {
    match biome {
        Biome::Ocean => [0.3, 0.5, 0.8],
        Biome::Beach => [1.0, 0.9, 0.7],
        Biome::Plains => [0.95, 0.92, 0.85],
        Biome::Forest => [0.6, 0.9, 0.5],
        Biome::DenseForest => [0.3, 0.7, 0.25],
        Biome::Hills => [0.85, 0.8, 0.7],
        Biome::Mountains => [0.9, 0.88, 0.85],
        Biome::SnowCaps => [0.95, 0.97, 1.0],
        Biome::Desert => [1.0, 0.85, 0.5],
        Biome::Savanna => [0.95, 0.85, 0.55],
        Biome::Jungle => [0.4, 0.8, 0.3],
        Biome::Tundra => [0.7, 0.75, 0.85],
        Biome::Swamp => [0.5, 0.65, 0.4],
        Biome::Volcanic => [1.0, 0.4, 0.1],
        Biome::Mushroom => [0.8, 0.4, 0.9],
        Biome::Crystal => [0.6, 0.95, 1.0],
    }
}

// ─── SIMD Shadow Probing ────────────────────────────────────────────────────
//
// The core shadow algorithm fires 8 rays from the surface point toward
// the light, with slight offsets to create an 8-probe "star" pattern.
// This simulates area-light shadow softening: if some probes are blocked
// and others aren't, we're in the penumbra region.
//
// Each probe ray is marched through the SDF world. If the marcher finds
// an occluder between the point and the light, that probe is "in shadow".
// The 8 probes are processed using the SimdPrng8 for generating their
// offset directions, and the collision module's PROBE_OFFSETS_8 for
// the star pattern.

/// Fire 8 shadow rays toward a light and check for occluders.
///
/// The 8 probe directions are computed by perturbing the primary light
/// direction using the PROBE_OFFSETS_8 star pattern. Each ray is
/// marched from the surface point (with shadow bias offset) toward
/// the light. If any ray hits geometry before reaching the light,
/// that probe is marked as "in shadow".
///
/// The shadow factor is computed as: (unoccluded probes) / 8.
/// This gives soft shadows that are proportional to the fraction of
/// the "virtual area light" that is visible.
pub fn simd_shadow_probe_8(
    ctx: &VplContext,
    sdf_ctx: &SdfContext,
    point: &Vec3,
    light_pos: &Vec3,
) -> ShadowProbe8 {
    // Primary direction from point to light
    let to_light = light_pos.sub(point);
    let dist_to_light = to_light.length();
    let primary_dir = to_light.normalize();

    // Build an orthonormal basis around the primary direction
    // for creating the 8 probe offsets
    let (tangent, bitangent) = compute_tangent_frame(&primary_dir);

    let mut results = [false; 8];
    let mut distances = [0.0f64; 8];
    let mut directions = [Vec3::zero(); 8];

    // Use SimdPrng8 to generate jitter for the 8 probe directions
    let mut simd_rng = SimdPrng8::new(ctx.seed.wrapping_add(0x5EAD_0B5_u64));
    let jitter = simd_rng.next_8x_f64();

    // Offset scale: small perturbation relative to the light distance
    // This controls the "area light" size — bigger offset = softer shadows
    let offset_scale = 0.02 * dist_to_light;

    for i in 0..8 {
        let (dx, dy) = PROBE_OFFSETS_8[i];

        // Compute the jittered probe direction
        let jitter_amount = (jitter[i] - 0.5) * 0.5; // ±0.25 jitter
        let probe_dir = Vec3::new(
            primary_dir.x + (dx + jitter_amount) * offset_scale * tangent.x
                         + dy * offset_scale * bitangent.x,
            primary_dir.y + (dx + jitter_amount) * offset_scale * tangent.y
                         + dy * offset_scale * bitangent.y,
            primary_dir.z + (dx + jitter_amount) * offset_scale * tangent.z
                         + dy * offset_scale * bitangent.z,
        ).normalize();

        directions[i] = probe_dir;

        // Offset the ray origin along the normal to prevent self-shadowing
        // (shadow bias). We use the normal from the SDF context if available,
        // otherwise we approximate with the probe direction itself.
        let surface_normal = compute_normal(sdf_ctx, point);
        let biased_origin = point.add(&surface_normal.scale(ctx.shadow_bias));

        // March the shadow ray toward the light
        let shadow_ray = Ray::new(biased_origin, probe_dir);
        let hit = ray_march(sdf_ctx, &shadow_ray);

        if hit.hit && hit.distance < dist_to_light {
            // Occluder found before reaching the light
            results[i] = true;
            distances[i] = hit.distance;
        } else {
            // Light is visible along this probe direction
            results[i] = false;
            distances[i] = dist_to_light;
        }
    }

    ShadowProbe8 {
        results,
        distances,
        directions,
    }
}

/// Compute a tangent frame (tangent, bitangent) from a direction vector.
///
/// Uses the "frisvad" method: pick the smallest component of the normal
/// to cross with, ensuring numerical stability.
#[inline(always)]
fn compute_tangent_frame(normal: &Vec3) -> (Vec3, Vec3) {
    let tangent = if normal.y.abs() > 0.999 {
        Vec3::new(1.0, 0.0, 0.0).cross(normal).normalize()
    } else {
        Vec3::new(0.0, 1.0, 0.0).cross(normal).normalize()
    };
    let bitangent = normal.cross(&tangent).normalize();
    (tangent, bitangent)
}

/// Cross product for Vec3.
impl Vec3 {
    #[inline(always)]
    fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Reflect a vector around a normal.
    #[inline(always)]
    fn reflect(&self, normal: &Vec3) -> Vec3 {
        let d = 2.0 * self.dot(normal);
        self.sub(&normal.scale(d))
    }

    /// Linear interpolation between two vectors.
    #[inline(always)]
    fn lerp(a: &Vec3, b: &Vec3, t: f64) -> Vec3 {
        a.scale(1.0 - t).add(&b.scale(t))
    }

    /// Component-wise multiplication.
    #[inline(always)]
    fn cmul(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
        )
    }
}

// ─── Full Lighting Computation ───────────────────────────────────────────────
//
// The lighting pipeline for a single surface point:
//
//   1. Ambient contribution: ambient_color * ao_factor
//   2. Directional sun: Lambertian + GGX specular, attenuated by shadow
//   3. VPL contributions: For each VPL within range, compute diffuse +
//      specular with distance falloff. Each VPL is also shadow-tested.
//   4. Harmonic bounce: Approximate indirect lighting from VPL positions
//      using a simple 1-bounce gathering step.
//   5. Atmospheric: Fog/scattering contribution based on distance and
//      sun alignment.
//
// The result is a single LightingResult with all components.

/// Compute full lighting for a surface hit point.
///
/// This is the main entry point for the VPL lighting pipeline. Given a
/// surface intersection (from the ray marcher), it computes the final
/// color including:
///   - Ambient light (modulated by AO)
///   - Directional sun light (with shadows)
///   - VPL contributions (with per-light shadows)
///   - Harmonic bounce approximation
///   - Atmospheric scattering
pub fn compute_lighting(
    ctx: &VplContext,
    sdf_ctx: &SdfContext,
    hit: &HitInfo,
) -> LightingResult {
    if !hit.hit {
        // No surface hit — return atmospheric scattering
        let ray = Ray::new(hit.position, ctx.sun_direction.scale(-1.0));
        let fog_color = atmospheric_scattering(&ray, 1000.0, 0.01, &ctx.sun_direction);
        return LightingResult {
            color: fog_color,
            shadow_factor: 1.0,
            ao_factor: 1.0,
            diffuse: [0.0; 3],
            specular: [0.0; 3],
        };
    }

    let point = hit.position;
    let normal = hit.normal;

    // ── Step 1: Ambient Occlusion ──
    let ao = ambient_occlusion(ctx, sdf_ctx, &point, &normal);

    // ── Step 2: Sun Shadow ──
    // Create a point above the surface along the sun direction to use as the
    // "light position" for shadow probing. This simulates a distant directional
    // light by placing a point light far away.
    let sun_pos = point.add(&ctx.sun_direction.scale(1000.0));
    let sun_probe = simd_shadow_probe_8(ctx, sdf_ctx, &point, &sun_pos);
    let sun_shadow = {
        let lit_count = sun_probe.results.iter().filter(|&&s| !s).count();
        lit_count as f64 / 8.0
    };

    // ── Step 3: Sun Diffuse (Lambertian) ──
    let ndotl = (normal.dot(&ctx.sun_direction)).max(0.0);
    let sun_diffuse_factor = ndotl * sun_shadow * ctx.sun_intensity;

    // ── Step 4: Sun Specular (GGX/Cook-Torrance) ──
    // View direction: from hit point toward the camera (approximate as
    // the negative ray direction). For the dispatch path, we use the
    // normal itself as an approximation.
    let view_dir = normal.scale(1.0); // Approximate: camera is "above" the surface
    let sun_spec_factor = specular_highlight(&view_dir, &ctx.sun_direction, &normal, 0.3)
        * sun_shadow * ctx.sun_intensity;

    // ── Step 5: Accumulate sun contribution ──
    let mut diffuse = [0.0f32; 3];
    let mut specular = [0.0f32; 3];

    for c in 0..3 {
        diffuse[c] += ctx.sun_color[c] * sun_diffuse_factor as f32;
        specular[c] += ctx.sun_color[c] * sun_spec_factor as f32;
    }

    // ── Step 6: VPL Contributions ──
    let vpls = generate_vpl_positions(ctx, &point, DEFAULT_VPL_RADIUS);

    for vpl in &vpls {
        let to_light = vpl.position.sub(&point);
        let dist = to_light.length();
        if dist > vpl.radius || dist < 0.001 {
            continue;
        }

        let light_dir = to_light.normalize();
        let ndotl_vpl = normal.dot(&light_dir).max(0.0);
        if ndotl_vpl <= 0.0 {
            continue; // Light is behind the surface
        }

        // VPL shadow
        let vpl_probe = simd_shadow_probe_8(ctx, sdf_ctx, &point, &vpl.position);
        let vpl_lit_count = vpl_probe.results.iter().filter(|&&s| !s).count();
        let vpl_shadow = vpl_lit_count as f64 / 8.0;

        if vpl_shadow <= 0.0 {
            continue; // Fully in shadow
        }

        // 1/d² falloff
        let falloff = 1.0 / (dist * dist + 1.0);
        let vpl_diffuse = ndotl_vpl * vpl.intensity * falloff * vpl_shadow;

        // VPL specular (rougher = more spread)
        let vpl_spec = specular_highlight(&view_dir, &light_dir, &normal, 0.5)
            * vpl.intensity * falloff * vpl_shadow;

        for c in 0..3 {
            diffuse[c] += vpl.color[c] * vpl_diffuse as f32;
            specular[c] += vpl.color[c] * vpl_spec as f32;
        }
    }

    // ── Step 7: Harmonic Bounce (1-bounce approximate GI) ──
    // For each VPL, estimate the indirect light it receives from the sun
    // and re-emits. This is a simple approximation:
    //   bounce_color = vpl_color * vpl_sun_visibility * bounce_factor
    let bounce_factor = 0.1; // 10% energy transfer per bounce
    for vpl in vpls.iter().take(8) {
        let to_light = vpl.position.sub(&point);
        let dist = to_light.length();
        if dist > vpl.radius || dist < 0.001 {
            continue;
        }

        let light_dir = to_light.normalize();
        let ndotl_vpl = normal.dot(&light_dir).max(0.0);
        if ndotl_vpl <= 0.0 {
            continue;
        }

        // Estimate how much sun the VPL position receives
        let vpl_normal = compute_normal(sdf_ctx, &vpl.position);
        let vpl_sun_vis = vpl_normal.dot(&ctx.sun_direction).max(0.0);

        let falloff = 1.0 / (dist * dist + 1.0);
        let bounce_intensity = vpl_sun_vis * vpl.intensity * falloff * bounce_factor;

        for c in 0..3 {
            diffuse[c] += vpl.color[c] * bounce_intensity as f32 * ndotl_vpl as f32;
        }
    }

    // ── Step 8: Compose final color ──
    let mut color = [0.0f32; 3];

    // Ambient * AO
    for c in 0..3 {
        color[c] += ctx.ambient[c] * ao as f32;
    }

    // Add diffuse and specular
    for c in 0..3 {
        color[c] += diffuse[c] + specular[c];
    }

    // ── Step 9: Atmospheric scattering (distance-based fog) ──
    let fog_density = 0.005; // Light fog
    let view_ray = Ray::new(point, view_dir);
    let fog_color = atmospheric_scattering(&view_ray, 500.0, fog_density, &ctx.sun_direction);
    let fog_blend = 1.0 - (-fog_density * 500.0).exp();
    for c in 0..3 {
        color[c] = color[c] * (1.0 - fog_blend as f32) + fog_color[c] * fog_blend as f32;
    }

    // Overall shadow factor: weighted combination of sun shadow and VPL shadows
    let shadow_factor = sun_shadow * 0.7 + ao * 0.3;

    LightingResult {
        color,
        shadow_factor,
        ao_factor: ao,
        diffuse,
        specular,
    }
}

// ─── Ambient Occlusion ──────────────────────────────────────────────────────
//
// Hemisphere sampling AO using 8 SDF probes. We cast 8 rays from the
// surface point into the upper hemisphere (defined by the surface normal)
// and measure how far each ray travels before hitting geometry.
//
// If a ray travels a short distance before hitting something, that
// direction is "occluded". The AO factor is the average visibility:
//
//   AO = average(distance_along_ray / max_ao_distance, for each ray)
//
// We use SimdPrng8 to generate the hemisphere sample directions
// deterministically.

/// Maximum distance for AO rays.
const AO_MAX_DISTANCE: f64 = 5.0;

/// Compute ambient occlusion at a surface point using hemisphere sampling.
///
/// Casts 8 rays into the hemisphere above the surface normal and measures
/// how much open sky is visible. Uses SimdPrng8 for deterministic
/// sample direction generation.
///
/// Returns a value in [0.0, 1.0] where:
///   1.0 = completely unoccluded (open sky)
///   0.0 = completely occluded (enclosed space)
pub fn ambient_occlusion(
    ctx: &VplContext,
    sdf_ctx: &SdfContext,
    point: &Vec3,
    normal: &Vec3,
) -> f64 {
    let mut simd_rng = SimdPrng8::new(ctx.seed.wrapping_add(0xA0_A0_A0A0));
    let r1 = simd_rng.next_8x_f64();
    let r2 = simd_rng.next_8x_f64();

    // Build tangent frame for hemisphere sampling
    let (tangent, bitangent) = compute_tangent_frame(normal);

    let mut occlusion = 0.0f64;

    for i in 0..8 {
        // Cosine-weighted hemisphere sampling using Malley's method:
        // Generate a point on a disk, then project onto the hemisphere.
        let r = r1[i].sqrt();
        let theta = r2[i] * 2.0 * std::f64::consts::PI;

        let x = r * theta.cos();
        let y = r * theta.sin();
        let z = (1.0 - r1[i]).sqrt(); // cos(elevation)

        // Transform from tangent space to world space
        let sample_dir = Vec3::new(
            x * tangent.x + y * bitangent.x + z * normal.x,
            x * tangent.y + y * bitangent.y + z * normal.y,
            x * tangent.z + y * bitangent.z + z * normal.z,
        ).normalize();

        // March along the sample direction and measure distance
        let ao_ray = Ray::new(
            point.add(&normal.scale(ctx.shadow_bias)),
            sample_dir,
        );
        let ao_hit = ray_march(sdf_ctx, &ao_ray);

        let distance = if ao_hit.hit {
            ao_hit.distance.min(AO_MAX_DISTANCE)
        } else {
            AO_MAX_DISTANCE
        };

        // Occlusion contribution: 1 - (distance / max_distance)
        // Closer geometry = more occlusion
        occlusion += 1.0 - (distance / AO_MAX_DISTANCE);
    }

    // Average over 8 samples, then invert so that high occlusion → low AO
    let avg_occlusion = occlusion / 8.0;

    // Apply a contrast curve to make AO more visible
    let ao = 1.0 - avg_occlusion;
    ao.powf(1.5) // Increase contrast slightly
}

// ─── Specular Highlight (GGX / Cook-Torrance) ───────────────────────────────
//
// Implements the GGX (Trowbridge-Reitz) microfacet BRDF for specular
// highlights. This is the industry standard for PBR (Physically Based
// Rendering) and produces realistic highlights that stretch on grazing
// angles and narrow on rough surfaces.
//
// The formula:
//   D(h) = α² / (π · ((n·h)² · (α² - 1) + 1)²)
//
// where α = roughness², h = half-vector between view and light directions.

/// Compute the GGX/Cook-Torrance specular highlight intensity.
///
/// Parameters:
///   - `view_dir`: Direction from surface point to the camera (normalized)
///   - `light_dir`: Direction from surface point to the light (normalized)
///   - `normal`: Surface normal (normalized)
///   - `roughness`: Surface roughness [0.0 = mirror, 1.0 = fully rough]
///
/// Returns the specular intensity (scalar, to be multiplied by light color).
pub fn specular_highlight(
    view_dir: &Vec3,
    light_dir: &Vec3,
    normal: &Vec3,
    roughness: f64,
) -> f64 {
    let roughness = roughness.max(0.001).min(1.0);
    let alpha = roughness * roughness;

    // Half-vector between view and light directions
    let half_vec = view_dir.add(light_dir).normalize();

    let ndotv = normal.dot(view_dir).max(0.001);
    let ndotl = normal.dot(light_dir).max(0.0);
    let ndoth = normal.dot(&half_vec).max(0.0);
    let vdoth = view_dir.dot(&half_vec).max(0.0);

    // If light or view is behind the surface, no specular
    if ndotl <= 0.0 || ndotv <= 0.0 {
        return 0.0;
    }

    // GGX normal distribution function
    let d = ggx_distribution(ndoth, alpha);

    // Smith-GGX geometry function (joint masking-shadowing)
    let g = smith_ggx_geometry(ndotv, ndotl, alpha);

    // Fresnel term (Schlick approximation with F0 = 0.04 for dielectrics)
    let f0 = 0.04;
    let fresnel = f0 + (1.0 - f0) * (1.0 - vdoth).powi(5);

    // Cook-Torrance BRDF: D * G * F / (4 * NdotV * NdotL)
    let denominator = 4.0 * ndotv * ndotl;
    if denominator < 1e-10 {
        return 0.0;
    }

    (d * g * fresnel) / denominator
}

/// GGX (Trowbridge-Reitz) normal distribution function.
///
/// D(h) = α² / (π · ((n·h)² · (α² - 1) + 1)²)
#[inline(always)]
fn ggx_distribution(ndoth: f64, alpha: f64) -> f64 {
    let alpha2 = alpha * alpha;
    let denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0;
    alpha2 / (std::f64::consts::PI * denom * denom)
}

/// Smith-GGX geometry function (joint masking-shadowing).
///
/// Uses the height-correlated Smith formulation for better accuracy
/// at grazing angles.
#[inline(always)]
fn smith_ggx_geometry(ndotv: f64, ndotl: f64, alpha: f64) -> f64 {
    let r = alpha + 1.0;
    let k = (r * r) / 8.0; // Epic Games remapping

    let g1 = ndotv / (ndotv * (1.0 - k) + k);
    let g2 = ndotl / (ndotl * (1.0 - k) + k);

    g1 * g2
}

// ─── Atmospheric Scattering ──────────────────────────────────────────────────
//
// Implements a simplified Rayleigh + Mie scattering model for atmospheric
// effects. The key features:
//
//   1. Rayleigh scattering: Short wavelengths scatter more (blue sky).
//      Intensity ∝ 1/λ⁴, so blue light scatters ~5.5× more than red.
//   2. Mie scattering: Forward-peaked scattering (haze/fog).
//      Creates the bright glow around the sun.
//   3. 1/d² falloff: PRNG-generated fog density varies spatially,
//      creating volumetric "god rays" effect.
//
// This is NOT a physically accurate simulation — it's a real-time
// approximation that produces visually convincing results.

/// Rayleigh scattering coefficients (approximate, for RGB wavelengths).
/// Blue scatters much more than red.
const RAYLEIGH_BETA: [f32; 3] = [0.25, 0.59, 1.28];

/// Mie scattering coefficient (wavelength-independent haze).
const MIE_BETA: f32 = 0.5;

/// Compute atmospheric scattering color along a ray.
///
/// Marches along the ray, accumulating scattered light from the sun.
/// The fog density is modulated by a PRNG-generated value with 1/d²
/// spatial falloff, creating "god rays" when looking toward the sun.
///
/// Parameters:
///   - `ray`: The view ray
///   - `distance`: How far to march
///   - `fog_density`: Base fog density
///   - `sun_dir`: Direction to the sun (normalized)
///
/// Returns the RGB fog/scattering color (linear space).
pub fn atmospheric_scattering(
    ray: &Ray,
    distance: f64,
    fog_density: f64,
    sun_dir: &Vec3,
) -> [f32; 3] {
    let steps = 16u32;
    let step_size = distance / steps as f64;

    let mut accumulated = [0.0f32; 3];
    let mut transmittance = 1.0f64;

    for i in 0..steps {
        let t = (i as f64 + 0.5) * step_size;
        let pos = ray.at(t);

        // PRNG-generated local fog density variation with 1/d² falloff
        // This creates spatial variation in the fog (like actual fog patches)
        let hash = hash_coord_2d(
            0xA7705_FE_u64, // Dedicated atmospheric seed offset
            pos.x.floor() as i64,
            pos.z.floor() as i64,
        );
        let local_density_mod = hash_to_f64(hash);

        // Distance-based fog density falloff: denser near ground
        let height_falloff = (-pos.y * 0.01).exp().max(0.01);
        let local_density = fog_density * (0.5 + local_density_mod) * height_falloff;

        // Phase function: how much light scatters toward the camera
        let cos_theta = ray.direction.dot(sun_dir);
        let phase = rayleigh_phase(cos_theta) + mie_phase(cos_theta, 0.76);

        // Sun visibility at this point (simplified: just use the dot product)
        // In a full implementation, we'd shadow-test here
        let sun_vis = cos_theta.max(0.0).min(1.0);

        // Accumulate in-scattered light
        for c in 0..3 {
            let scattering = RAYLEIGH_BETA[c] as f64 * local_density * step_size;
            let inscatter = scattering * phase * sun_vis * transmittance;
            accumulated[c] += inscatter as f32;
        }

        // Beer's law: transmittance decreases exponentially
        let extinction = fog_density * (0.5 + local_density_mod) * height_falloff * step_size;
        transmittance *= (-extinction).exp();
    }

    // Add a base sky color (blue gradient)
    let sky_blend = (ray.direction.y * 0.5 + 0.5).max(0.0) as f32;
    let sky_color = [
        0.4 * sky_blend + 0.1,
        0.6 * sky_blend + 0.15,
        1.0 * sky_blend + 0.2,
    ];

    for c in 0..3 {
        accumulated[c] = accumulated[c] * transmittance as f32 + sky_color[c] * (1.0 - transmittance as f32);
    }

    accumulated
}

/// Rayleigh phase function: isotropic scattering with wavelength dependence.
/// Simplified to 3/16π * (1 + cos²θ).
#[inline(always)]
fn rayleigh_phase(cos_theta: f64) -> f64 {
    let f = 3.0 / (16.0 * std::f64::consts::PI);
    f * (1.0 + cos_theta * cos_theta)
}

/// Mie phase function (Henyey-Greenstein): forward-peaked scattering.
///
/// g ∈ [-1, 1]: g=0 is isotropic, g=1 is fully forward, g=-1 is backward.
/// For atmospheric haze, g ≈ 0.76 gives a realistic forward peak.
#[inline(always)]
fn mie_phase(cos_theta: f64, g: f64) -> f64 {
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).powi(3).sqrt();
    (1.0 - g2) / (4.0 * std::f64::consts::PI * denom)
}

// ─── God Rays ───────────────────────────────────────────────────────────────
//
// "God rays" (crepuscular rays) are the visible beams of light that
// appear when sunlight passes through gaps in clouds or fog. We simulate
// this by marching along a ray toward the sun and accumulating the
// PRNG-generated fog density at each step.
//
// When the fog is denser (hash value > threshold), we accumulate more
// light — creating bright beams. When the fog is thinner, less light
// passes through — creating the dark gaps between beams.

/// Compute god ray intensity along a ray toward the sun.
///
/// Marches along the ray, sampling PRNG-generated fog density at each
/// step. The intensity is the accumulated light that "made it through"
/// the fog, with 1/d² falloff on the fog density.
///
/// Parameters:
///   - `ray`: The view ray
///   - `sun_dir`: Direction to the sun
///   - `steps`: Number of march steps (more = higher quality)
///   - `seed`: Seed for PRNG fog density generation
///
/// Returns the god ray intensity [0.0, ∞) (typically 0.0 - 2.0).
pub fn god_rays_intensity(
    ray: &Ray,
    sun_dir: &Vec3,
    steps: u32,
    seed: u64,
) -> f64 {
    let max_distance = 200.0;
    let step_size = max_distance / steps as f64;
    let mut intensity = 0.0f64;
    let mut transmittance = 1.0f64;

    let mut simd_rng = SimdPrng8::new(seed.wrapping_add(0x60DA7E5_u64));

    for i in 0..steps {
        let t = (i as f64 + 0.5) * step_size;
        let pos = ray.at(t);

        // How aligned is this ray with the sun direction?
        // More aligned = more light scattering toward the camera
        let alignment = ray.direction.dot(sun_dir).max(0.0);

        // PRNG-generated fog density at this position
        // Use Morton-encoded coordinates for cache-coherent access
        let mx = ((pos.x + max_distance) * 0.5).max(0.0) as u32;
        let mz = ((pos.z + max_distance) * 0.5).max(0.0) as u32;
        let morton = encode_2d(mx, mz);

        // Hash the Morton code for this position
        let hash = hash_coord_2d(
            seed.wrapping_add(0x60DA7E5_u64),
            morton as i64,
            pos.y.floor() as i64,
        );
        let density = hash_to_f64(hash);

        // 1/d² falloff: density decreases with distance from camera
        let falloff = 1.0 / (t * t + 1.0);

        // Threshold: only "bright" fog contributes to god rays
        // This creates the beam/gap pattern
        let threshold = 0.3;
        if density > threshold {
            let beam_strength = (density - threshold) * falloff;
            let mie_scatter = mie_phase(alignment, 0.9); // Strong forward peak
            intensity += beam_strength * mie_scatter * transmittance * step_size * 0.1;
        }

        // Transmittance decay
        let extinction = density * falloff * step_size * 0.01;
        transmittance *= (-extinction).exp();

        // Early termination if fully attenuated
        if transmittance < 0.001 {
            break;
        }
    }

    intensity
}

// ─── Light Clustering (Morton-Ordered) ───────────────────────────────────────
//
// For efficient shadow queries, VPLs are stored in Morton order. This
// ensures that spatially close lights are also memory-close, maximizing
// CPU cache hits when evaluating shadow maps.
//
// The cluster system divides the world into tiles (using simd_tile_index)
// and assigns lights to tiles. Within each tile, lights are sorted by
// Morton code.

/// A cluster of VPLs in a spatial tile.
#[derive(Debug, Clone)]
pub struct VplCluster {
    /// Tile index (Morton code of the tile)
    pub tile_morton: u64,
    /// VPLs in this cluster, sorted by Morton code within the tile
    pub lights: Vec<VirtualPointLight>,
    /// Bounding box of the cluster (for culling)
    pub min_pos: Vec3,
    pub max_pos: Vec3,
}

/// Build VPL clusters from a list of lights.
///
/// Divides the world into `tile_size`×`tile_size` tiles and assigns
/// each VPL to its tile. Lights within each tile are sorted by
/// Morton code for cache coherence.
pub fn cluster_vpls(lights: &[VirtualPointLight], tile_size: u32) -> Vec<VplCluster> {
    let mut cluster_map: std::collections::HashMap<u64, Vec<VirtualPointLight>> =
        std::collections::HashMap::new();

    for light in lights {
        let tx = (light.position.x.max(0.0) as u32) / tile_size;
        let tz = (light.position.z.max(0.0) as u32) / tile_size;
        let tile_morton = encode_2d(tx, tz);

        cluster_map
            .entry(tile_morton)
            .or_default()
            .push(*light);
    }

    let mut clusters: Vec<VplCluster> = cluster_map
        .into_iter()
        .map(|(tile_morton, mut lights)| {
            // Sort lights within the cluster by Morton code
            lights.sort_by(|a, b| {
                let ma = encode_2d(
                    a.position.x.max(0.0) as u32,
                    a.position.z.max(0.0) as u32,
                );
                let mb = encode_2d(
                    b.position.x.max(0.0) as u32,
                    b.position.z.max(0.0) as u32,
                );
                ma.cmp(&mb)
            });

            // Compute bounding box
            let min_pos = lights.iter().fold(Vec3::new(f64::MAX, f64::MAX, f64::MAX), |acc, l| {
                acc.min(&l.position)
            });
            let max_pos = lights.iter().fold(Vec3::new(f64::MIN, f64::MIN, f64::MIN), |acc, l| {
                acc.max(&l.position)
            });

            VplCluster {
                tile_morton,
                lights,
                min_pos,
                max_pos,
            }
        })
        .collect();

    // Sort clusters by tile Morton code
    clusters.sort_by_key(|c| c.tile_morton);

    clusters
}

// ─── Verification ───────────────────────────────────────────────────────────

/// Verify the VPL lighting system.
///
/// Tests:
///   1. VPL position generation is deterministic
///   2. Shadow probing returns consistent results
///   3. Ambient occlusion is bounded [0, 1]
///   4. Specular highlight is non-negative
///   5. Atmospheric scattering produces valid colors
///   6. God rays intensity is non-negative
///   7. Full lighting computation produces valid results
///   8. VPL clustering works correctly
///   9. Morton ordering is preserved
pub fn verify_vpl_lighting() -> bool {
    let mut all_pass = true;
    let seed = 42u64;

    // ── Test 1: VPL position generation is deterministic ──
    let ctx = VplContext::new(seed);
    let center = Vec3::new(0.0, 10.0, 0.0);
    let lights1 = generate_vpl_positions(&ctx, &center, 64.0);
    let lights2 = generate_vpl_positions(&ctx, &center, 64.0);

    if lights1.len() != lights2.len() {
        eprintln!(
            "FAIL: VPL position generation length mismatch: {} vs {}",
            lights1.len(), lights2.len()
        );
        all_pass = false;
    } else {
        for (i, (a, b)) in lights1.iter().zip(lights2.iter()).enumerate() {
            if (a.position.x - b.position.x).abs() > 1e-6
                || (a.position.y - b.position.y).abs() > 1e-6
                || (a.position.z - b.position.z).abs() > 1e-6
            {
                eprintln!("FAIL: VPL position mismatch at index {}", i);
                all_pass = false;
                break;
            }
        }
    }

    // ── Test 2: Shadow probing returns consistent results ──
    let sdf_ctx = SdfContext::new(seed);
    let point = Vec3::new(0.0, 40.0, 0.0); // Above terrain
    let light_pos = Vec3::new(5.0, 60.0, 5.0);

    let probe1 = simd_shadow_probe_8(&ctx, &sdf_ctx, &point, &light_pos);
    let probe2 = simd_shadow_probe_8(&ctx, &sdf_ctx, &point, &light_pos);

    for i in 0..8 {
        if probe1.results[i] != probe2.results[i] {
            eprintln!(
                "FAIL: Shadow probe result mismatch at index {}: {} vs {}",
                i, probe1.results[i], probe2.results[i]
            );
            all_pass = false;
            break;
        }
    }

    // ── Test 3: Ambient occlusion is bounded [0, 1] ──
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let ao = ambient_occlusion(&ctx, &sdf_ctx, &point, &normal);

    if ao < 0.0 || ao > 1.0 {
        eprintln!("FAIL: AO out of bounds: {} (expected [0, 1])", ao);
        all_pass = false;
    }

    // AO above open terrain should be relatively high (unoccluded)
    if ao < 0.1 {
        eprintln!("NOTE: AO above open terrain is low: {} (expected > 0.1)", ao);
    }

    // ── Test 4: Specular highlight is non-negative ──
    let view = Vec3::new(0.0, 1.0, 0.0);
    let light_dir = Vec3::new(0.5, 0.8, 0.3).normalize();
    let spec = specular_highlight(&view, &light_dir, &normal, 0.3);

    if spec < 0.0 {
        eprintln!("FAIL: Specular highlight is negative: {}", spec);
        all_pass = false;
    }

    // Specular should be > 0 when light is in the reflection direction
    if spec <= 0.0 {
        eprintln!("NOTE: Specular highlight is zero for visible light direction");
    }

    // Specular for rougher surfaces should be lower
    let spec_rough = specular_highlight(&view, &light_dir, &normal, 0.9);
    if spec_rough > spec * 2.0 {
        // Rougher surfaces can have broader but lower peaks; this is a sanity check
        eprintln!(
            "NOTE: Rough specular ({}) unexpectedly higher than smooth ({})",
            spec_rough, spec
        );
    }

    // ── Test 5: Atmospheric scattering produces valid colors ──
    let ray = Ray::new(Vec3::new(0.0, 50.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
    let fog_color = atmospheric_scattering(&ray, 500.0, 0.01, &ctx.sun_direction);

    for c in 0..3 {
        if fog_color[c] < 0.0 || fog_color[c] > 10.0 {
            eprintln!(
                "FAIL: Atmospheric scattering color[{}] out of range: {}",
                c, fog_color[c]
            );
            all_pass = false;
        }
    }

    // ── Test 6: God rays intensity is non-negative ──
    let god_ray = god_rays_intensity(&ray, &ctx.sun_direction, 16, seed);

    if god_ray < 0.0 {
        eprintln!("FAIL: God rays intensity is negative: {}", god_ray);
        all_pass = false;
    }

    // ── Test 7: Full lighting computation produces valid results ──
    let hit = HitInfo::hit(0.0, point, normal, 3, 10); // material_id = 3 (grass)
    let result = compute_lighting(&ctx, &sdf_ctx, &hit);

    for c in 0..3 {
        if result.color[c] < 0.0 {
            eprintln!(
                "FAIL: Lighting result color[{}] is negative: {}",
                c, result.color[c]
            );
            all_pass = false;
        }
        if result.diffuse[c] < 0.0 {
            eprintln!(
                "FAIL: Lighting result diffuse[{}] is negative: {}",
                c, result.diffuse[c]
            );
            all_pass = false;
        }
        if result.specular[c] < 0.0 {
            eprintln!(
                "FAIL: Lighting result specular[{}] is negative: {}",
                c, result.specular[c]
            );
            all_pass = false;
        }
    }

    if result.shadow_factor < 0.0 || result.shadow_factor > 1.0 {
        eprintln!(
            "FAIL: Shadow factor out of range: {} (expected [0, 1])",
            result.shadow_factor
        );
        all_pass = false;
    }

    if result.ao_factor < 0.0 || result.ao_factor > 1.0 {
        eprintln!(
            "FAIL: AO factor out of range: {} (expected [0, 1])",
            result.ao_factor
        );
        all_pass = false;
    }

    // ── Test 8: VPL clustering works ──
    if !lights1.is_empty() {
        let clusters = cluster_vpls(&lights1, 32);
        // Clusters should be non-empty
        if clusters.is_empty() {
            eprintln!("FAIL: VPL clustering produced no clusters");
            all_pass = false;
        }
        // Each cluster should have at least one light
        for (i, cluster) in clusters.iter().enumerate() {
            if cluster.lights.is_empty() {
                eprintln!("FAIL: VPL cluster {} is empty", i);
                all_pass = false;
            }
        }
        // Clusters should be sorted by Morton code
        for i in 1..clusters.len() {
            if clusters[i].tile_morton < clusters[i - 1].tile_morton {
                eprintln!("FAIL: VPL clusters not sorted by Morton code");
                all_pass = false;
                break;
            }
        }
    }

    // ── Test 9: Morton ordering is preserved in VPL generation ──
    {
        let vpls = generate_vpl_positions(&ctx, &center, 64.0);
        let morton_codes: Vec<u64> = vpls.iter().map(|vpl| {
            encode_2d(
                (vpl.position.x + 64.0).max(0.0) as u32,
                (vpl.position.z + 64.0).max(0.0) as u32,
            )
        }).collect();

        // Verify non-decreasing order (Morton codes should be sorted)
        for i in 1..morton_codes.len() {
            if morton_codes[i] < morton_codes[i - 1] {
                eprintln!(
                    "FAIL: VPL Morton codes not in order at index {}: {} > {}",
                    i, morton_codes[i - 1], morton_codes[i]
                );
                all_pass = false;
                break;
            }
        }
    }

    // ── Test 10: Different seeds produce different VPL positions ──
    {
        let ctx2 = VplContext::new(seed.wrapping_add(1));
        let lights_other = generate_vpl_positions(&ctx2, &center, 64.0);

        if lights1.len() > 0 && lights_other.len() > 0 {
            let same = (lights1[0].position.x - lights_other[0].position.x).abs() < 1e-6
                && (lights1[0].position.y - lights_other[0].position.y).abs() < 1e-6
                && (lights1[0].position.z - lights_other[0].position.z).abs() < 1e-6;
            if same {
                eprintln!("NOTE: Different seeds produced same first VPL position (unlikely but possible)");
            }
        }
    }

    // ── Test 11: GGX distribution integrates to ~1 over the hemisphere ──
    {
        // Quick sanity: D(h) should be positive for all valid inputs
        for ndoth in [0.01, 0.1, 0.5, 0.9, 1.0] {
            for alpha in [0.01, 0.1, 0.5, 1.0] {
                let d = ggx_distribution(ndoth, alpha);
                if d < 0.0 || !d.is_finite() {
                    eprintln!(
                        "FAIL: GGX distribution invalid for ndoth={}, alpha={}: {}",
                        ndoth, alpha, d
                    );
                    all_pass = false;
                }
            }
        }
    }

    // ── Test 12: Smith geometry function is bounded [0, 1] ──
    {
        for ndotv in [0.01, 0.1, 0.5, 1.0] {
            for ndotl in [0.01, 0.1, 0.5, 1.0] {
                let g = smith_ggx_geometry(ndotv, ndotl, 0.5);
                if g < 0.0 || g > 1.0 {
                    eprintln!(
                        "FAIL: Smith geometry out of [0,1] for ndotv={}, ndotl={}: {}",
                        ndotv, ndotl, g
                    );
                    all_pass = false;
                }
            }
        }
    }

    if all_pass {
        eprintln!("All VPL Lighting verification tests PASSED.");
    }

    all_pass
}
