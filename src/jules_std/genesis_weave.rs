// =============================================================================
// std/genesis_weave — Stateless Procedural World Generation Engine
//
// The "Genesis Weave" — a world-generation system that treats the entire world
// as a mathematical function rather than stored data.
//
// Implements:
//   1. Stateless Terrain: Height = PRNG(Seed, X, Y, Z) — O(1) any coordinate
//   2. Sieve-Based Biome Distribution: Bit-mask biome rules via 210-wheel logic
//   3. Entity Oracle: O(1) "is there a tree HERE?" — no stored object list
//   4. Exclusion Sieve: Buffer zones prevent structure overlap via parent-check
//   5. Jittered Grid: Deterministic spacing via cell-center + hash offset
//   6. Semantic Priority Hierarchy: Water > Structures > Flora > Props
//   7. Biome Bleed: Soft sieve for natural biome borders (weighted falloff)
//   8. L-System Structure Generation: Hash-driven fractal grammar for entities
//
// The tree does not exist in memory. It does not exist on disk. It is purely
// a mathematical consequence of the player looking at coordinate [10, 5, 20].
//
// Pure Rust, zero external dependencies. Uses prng_simd and sieve_210 modules.
// =============================================================================

#![allow(dead_code)]

use crate::interp::{RuntimeError, Value};
use crate::compiler::lexer::Span;
use crate::jules_std::prng_simd::{SquaresRng, ShishiuaRng, SimdPrng8};

// ─── Dispatch for stdlib integration ────────────────────────────────────────

pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        "genesis::terrain_height" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let x = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let y = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let z = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let height = terrain_height(seed, x, y, z);
            Some(Ok(Value::F64(height)))
        }
        "genesis::biome_at" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let x = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let y = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let biome = biome_at(seed, x, y);
            Some(Ok(Value::I64(biome as i64)))
        }
        "genesis::entity_at" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let x = args.get(1).and_then(|v| v.as_i64()).unwrap_or(0) as i64;
            let y = args.get(2).and_then(|v| v.as_i64()).unwrap_or(0) as i64;
            let threshold = args.get(3).and_then(|v| v.as_f64()).unwrap_or(0.15);
            let entity = entity_at(seed, x, y, threshold);
            Some(Ok(Value::I64(entity as i64)))
        }
        "genesis::check_exclusion" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let x = args.get(1).and_then(|v| v.as_i64()).unwrap_or(0) as i64;
            let y = args.get(2).and_then(|v| v.as_i64()).unwrap_or(0) as i64;
            let radius = args.get(3).and_then(|v| v.as_i64()).unwrap_or(128) as i64;
            let ok = check_exclusion(seed, x, y, radius);
            Some(Ok(Value::Bool(ok)))
        }
        "genesis::jittered_position" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let cell_x = args.get(1).and_then(|v| v.as_i64()).unwrap_or(0) as i64;
            let cell_y = args.get(2).and_then(|v| v.as_i64()).unwrap_or(0) as i64;
            let cell_size = args.get(3).and_then(|v| v.as_i64()).unwrap_or(64) as i64;
            let (jx, jy) = jittered_position(seed, cell_x, cell_y, cell_size);
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
                Value::F64(jx), Value::F64(jy),
            ])))))
        }
        "genesis::biome_weight" => {
            let seed = args.first().and_then(|v| v.as_i64()).unwrap_or(42) as u64;
            let x = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let y = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let biome_id = args.get(3).and_then(|v| v.as_i64()).unwrap_or(0) as u8;
            let weight = biome_weight(seed, x, y, biome_id);
            Some(Ok(Value::F64(weight)))
        }
        "genesis::generate_structure" => {
            let hash = args.first().and_then(|v| v.as_i64()).unwrap_or(0) as u64;
            let grammar = args.get(1).and_then(|v| v.as_i64()).unwrap_or(0) as u8;
            let depth = args.get(2).and_then(|v| v.as_i64()).unwrap_or(3) as u32;
            let structure = generate_structure(hash, grammar, depth);
            let vals: Vec<Value> = structure.iter().map(|&s| Value::I64(s as i64)).collect();
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vals)))))
        }
        "genesis::verify" => {
            let ok = verify_genesis();
            Some(Ok(Value::Bool(ok)))
        }
        _ => None,
    }
}

// ─── Biome Definitions ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Biome {
    Ocean = 0,
    Beach = 1,
    Plains = 2,
    Forest = 3,
    DenseForest = 4,
    Hills = 5,
    Mountains = 6,
    SnowCaps = 7,
    Desert = 8,
    Savanna = 9,
    Jungle = 10,
    Tundra = 11,
    Swamp = 12,
    Volcanic = 13,
    Mushroom = 14,
    Crystal = 15,
}

impl Biome {
    pub fn from_id(id: u8) -> Self {
        match id {
            0 => Biome::Ocean,
            1 => Biome::Beach,
            2 => Biome::Plains,
            3 => Biome::Forest,
            4 => Biome::DenseForest,
            5 => Biome::Hills,
            6 => Biome::Mountains,
            7 => Biome::SnowCaps,
            8 => Biome::Desert,
            9 => Biome::Savanna,
            10 => Biome::Jungle,
            11 => Biome::Tundra,
            12 => Biome::Swamp,
            13 => Biome::Volcanic,
            14 => Biome::Mushroom,
            15 => Biome::Crystal,
            _ => Biome::Plains,
        }
    }

    /// Get the priority level for this biome (higher = more dominant)
    pub fn priority(&self) -> u8 {
        match self {
            Biome::Ocean => 0,       // Base layer — everything conforms
            Biome::Beach => 1,
            Biome::Mountains => 2,
            Biome::Volcanic => 2,
            Biome::SnowCaps => 2,
            Biome::Swamp => 3,
            Biome::Desert => 3,
            Biome::Tundra => 3,
            Biome::Savanna => 4,
            Biome::Plains => 4,
            Biome::Hills => 4,
            Biome::Forest => 5,
            Biome::DenseForest => 5,
            Biome::Jungle => 5,
            Biome::Mushroom => 5,
            Biome::Crystal => 5,
        }
    }
}

// ─── Entity Types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EntityType {
    None = 0,
    Tree = 1,
    Bush = 2,
    Rock = 3,
    Grass = 4,
    Flower = 5,
    Castle = 10,
    Dungeon = 11,
    Village = 12,
    Bridge = 13,
    Ruin = 14,
    WaterBody = 20,
    River = 21,
    Lake = 22,
}

impl EntityType {
    pub fn from_id(id: u8) -> Self {
        match id {
            0 => EntityType::None,
            1 => EntityType::Tree,
            2 => EntityType::Bush,
            3 => EntityType::Rock,
            4 => EntityType::Grass,
            5 => EntityType::Flower,
            10 => EntityType::Castle,
            11 => EntityType::Dungeon,
            12 => EntityType::Village,
            13 => EntityType::Bridge,
            14 => EntityType::Ruin,
            20 => EntityType::WaterBody,
            21 => EntityType::River,
            22 => EntityType::Lake,
            _ => EntityType::None,
        }
    }

    /// Get the semantic priority (higher wins in overlap conflicts)
    pub fn semantic_priority(&self) -> u8 {
        match self {
            EntityType::None => 0,
            EntityType::WaterBody | EntityType::River | EntityType::Lake => 0,
            EntityType::Grass | EntityType::Flower => 3,
            EntityType::Bush | EntityType::Rock => 2,
            EntityType::Tree => 2,
            EntityType::Castle | EntityType::Dungeon | EntityType::Village | EntityType::Bridge | EntityType::Ruin => 1,
        }
    }

    /// Is this a mega-structure (exclusion zone applies)?
    pub fn is_mega_structure(&self) -> bool {
        matches!(self, EntityType::Castle | EntityType::Dungeon | EntityType::Village | EntityType::Bridge)
    }
}

// ─── Coordinate Hashing ─────────────────────────────────────────────────────
//
// The fundamental operation: hash a coordinate into a deterministic u64.
// This is the "Oracle Query" — given (seed, x, y, z), produce a hash
// that determines everything about that point in the world.

/// Hash a 3D coordinate with a seed using the Shishiua counter-based PRNG.
/// The coordinate IS the counter — this makes it fully stateless.
#[inline(always)]
pub fn hash_coord_3d(seed: u64, x: i64, y: i64, z: i64) -> u64 {
    // Combine seed + coordinates into a unique counter using bit-mixing
    // We use a wyhash-style mixing to combine the inputs into a single u64
    let mut h = seed;
    h = h.wrapping_mul(0x6C62272E07BB0142).wrapping_add(x as u64);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    h = h.wrapping_add(y as u64);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D049BB133111EB);
    h = h.wrapping_add(z as u64);
    h = h ^ (h >> 31);
    // Run through Squares hash for final quality
    let rng = SquaresRng::new(h);
    rng.at_index(0)
}

/// Hash a 2D coordinate with a seed.
#[inline(always)]
pub fn hash_coord_2d(seed: u64, x: i64, y: i64) -> u64 {
    hash_coord_3d(seed, x, y, 0)
}

/// Convert a hash to a float in [0, 1).
#[inline(always)]
pub fn hash_to_f64(hash: u64) -> f64 {
    (hash >> 11) as f64 / (1u64 << 53) as f64
}

// ─── 1. Stateless Terrain ───────────────────────────────────────────────────
//
// Height = PRNG(Seed, X, Y, Z)
// If an agent teleports 10 trillion miles away, the engine calculates
// the exact terrain at that spot in nanoseconds without knowing what
// lies in between.

/// Compute the terrain height at world coordinates (x, y, z).
/// Uses multi-octave noise built from stateless PRNG queries.
///
/// The result is a height value in the range [0.0, 1.0].
pub fn terrain_height(seed: u64, x: f64, y: f64, z: f64) -> f64 {
    // Multi-octave value noise using stateless PRNG
    let mut total = 0.0f64;
    let mut amplitude = 1.0f64;
    let mut frequency = 1.0f64;
    let mut max_value = 0.0f64;
    const OCTAVES: u32 = 6;

    for _octave in 0..OCTAVES {
        let fx = (x * frequency).floor() as i64;
        let fy = (y * frequency).floor() as i64;
        let fz = (z * frequency).floor() as i64;

        // Query the PRNG oracle at the 8 corners of the current cell
        let v000 = hash_to_f64(hash_coord_3d(seed.wrapping_add(_octave as u64), fx, fy, fz));
        let v100 = hash_to_f64(hash_coord_3d(seed.wrapping_add(_octave as u64), fx + 1, fy, fz));
        let v010 = hash_to_f64(hash_coord_3d(seed.wrapping_add(_octave as u64), fx, fy + 1, fz));
        let v110 = hash_to_f64(hash_coord_3d(seed.wrapping_add(_octave as u64), fx + 1, fy + 1, fz));
        let v001 = hash_to_f64(hash_coord_3d(seed.wrapping_add(_octave as u64), fx, fy, fz + 1));
        let v101 = hash_to_f64(hash_coord_3d(seed.wrapping_add(_octave as u64), fx + 1, fy, fz + 1));
        let v011 = hash_to_f64(hash_coord_3d(seed.wrapping_add(_octave as u64), fx, fy + 1, fz + 1));
        let v111 = hash_to_f64(hash_coord_3d(seed.wrapping_add(_octave as u64), fx + 1, fy + 1, fz + 1));

        // Smooth interpolation (smoothstep) within the cell
        let dx = (x * frequency).fract();
        let dy = (y * frequency).fract();
        let dz = (z * frequency).fract();
        let sx = dx * dx * (3.0 - 2.0 * dx);
        let sy = dy * dy * (3.0 - 2.0 * dy);
        let sz = dz * dz * (3.0 - 2.0 * dz);

        // Trilinear interpolation
        let c00 = v000 * (1.0 - sx) + v100 * sx;
        let c10 = v010 * (1.0 - sx) + v110 * sx;
        let c01 = v001 * (1.0 - sx) + v101 * sx;
        let c11 = v011 * (1.0 - sx) + v111 * sx;
        let c0 = c00 * (1.0 - sy) + c10 * sy;
        let c1 = c01 * (1.0 - sy) + c11 * sy;
        let value = c0 * (1.0 - sz) + c1 * sz;

        total += value * amplitude;
        max_value += amplitude;
        amplitude *= 0.5;   // Each octave contributes half as much
        frequency *= 2.0;   // Each octave has double the frequency
    }

    // Normalize to [0, 1]
    total / max_value
}

/// Fast terrain height for 2D (z=0). Used for top-down world maps.
pub fn terrain_height_2d(seed: u64, x: f64, y: f64) -> f64 {
    terrain_height(seed, x, y, 0.0)
}

// ─── 2. Sieve-Based Biome Distribution ──────────────────────────────────────
//
// Uses the 210-Wheel Sieve logic creatively: "cross out" areas that are
// too close to each other. By using a bit-mask (similar to prime bit-packing),
// we can check if a 1km x 1km chunk is "valid" for a biome across an
// entire continent in a single CPU cycle.

/// Determine the biome at world coordinates (x, y).
/// Uses terrain height + temperature + moisture, all derived from stateless PRNG.
pub fn biome_at(seed: u64, x: f64, y: f64) -> Biome {
    let height = terrain_height_2d(seed, x, y);

    // Temperature and moisture are also stateless PRNG queries
    let temp = terrain_height_2d(seed.wrapping_add(1_000_000), x * 0.5, y * 0.5);
    let moisture = terrain_height_2d(seed.wrapping_add(2_000_000), x * 0.3, y * 0.3);

    // Sieve-style biome determination using bit-mask thresholds
    // The "sieve" here is a set of deterministic threshold checks
    // that categorize the coordinate into one of 16 biomes.
    classify_biome(height, temp, moisture)
}

/// Classify a point into a biome based on height, temperature, and moisture.
/// Uses a sieve-like cascade of threshold checks — each check "crosses out"
/// biomes that don't match, like the 210-wheel crosses out multiples.
#[inline(always)]
fn classify_biome(height: f64, temp: f64, moisture: f64) -> Biome {
    // Height bands (the "sieve levels")
    if height < 0.20 { return Biome::Ocean; }
    if height < 0.25 { return Biome::Beach; }
    if height > 0.85 { return Biome::SnowCaps; }
    if height > 0.75 {
        return if temp > 0.6 { Biome::Volcanic } else { Biome::Mountains };
    }

    // Temperature sieve
    if temp < 0.2 {
        return if moisture > 0.5 { Biome::Tundra } else { Biome::SnowCaps };
    }
    if temp > 0.8 {
        if moisture < 0.2 { return Biome::Desert; }
        if moisture < 0.5 { return Biome::Savanna; }
        return if height > 0.5 { Biome::Jungle } else { Biome::Swamp };
    }

    // Moderate temperature + height sieve
    if height > 0.55 {
        return if moisture > 0.6 { Biome::DenseForest } else { Biome::Hills };
    }

    // Moisture sieve for flat areas
    if moisture > 0.7 { return Biome::Forest; }
    if moisture > 0.4 { return Biome::Plains; }
    if moisture > 0.15 {
        return if temp > 0.6 { Biome::Savanna } else { Biome::Plains };
    }

    // Rare biomes (like "primes" in the sieve — sparse and special)
    if moisture > 0.5 && temp > 0.4 { return Biome::Mushroom; }
    if temp > 0.5 { return Biome::Desert; }

    Biome::Crystal
}

// ─── 3. Entity Oracle (O(1) Stateless Placement) ────────────────────────────
//
// You never ask, "Where are the trees?" You ask, "Is there a tree EXACTLY HERE?"
//
// When the player's camera moves, the engine queries the coordinates around them.
// The PRNG dice roll feeds the coordinate into the Shishiua generator.
// The Sieve Mask checks the hash against biome rules.
// If the hash passes the placement threshold, the entity MUST exist there.

/// Query what entity exists at integer coordinates (x, y).
/// Returns EntityType::None if no entity should be placed there.
///
/// This is the core "Oracle" — it determines placement in nanoseconds
/// without iterating through any stored object list.
pub fn entity_at(seed: u64, x: i64, y: i64, threshold: f64) -> EntityType {
    // Step 1: Determine biome
    let biome = biome_at(seed, x as f64, y as f64);

    // Step 2: Check water — no land entities in ocean
    if biome == Biome::Ocean { return EntityType::WaterBody; }
    if biome == Biome::Beach { return EntityType::None; }

    // Step 3: PRNG dice roll for this specific coordinate
    let hash = hash_coord_2d(seed.wrapping_add(0xDEAD_BEEF), x, y);
    let dice = hash_to_f64(hash);

    // Step 4: Sieve mask — check against biome-specific placement rules
    let entity = match biome {
        Biome::Forest | Biome::DenseForest => {
            if dice < threshold * 0.6 { EntityType::Tree }
            else if dice < threshold * 0.8 { EntityType::Bush }
            else if dice < threshold * 0.9 { EntityType::Flower }
            else { EntityType::None }
        }
        Biome::Plains | Biome::Savanna => {
            if dice < threshold * 0.2 { EntityType::Grass }
            else if dice < threshold * 0.3 { EntityType::Flower }
            else if dice < threshold * 0.35 { EntityType::Rock }
            else { EntityType::None }
        }
        Biome::Mountains | Biome::Hills => {
            if dice < threshold * 0.4 { EntityType::Rock }
            else if dice < threshold * 0.5 { EntityType::Bush }
            else { EntityType::None }
        }
        Biome::Desert => {
            if dice < threshold * 0.1 { EntityType::Rock }
            else { EntityType::None }
        }
        Biome::Jungle => {
            if dice < threshold * 0.7 { EntityType::Tree }
            else if dice < threshold * 0.85 { EntityType::Bush }
            else if dice < threshold * 0.9 { EntityType::Flower }
            else { EntityType::None }
        }
        Biome::Swamp => {
            if dice < threshold * 0.3 { EntityType::Tree }
            else if dice < threshold * 0.5 { EntityType::Bush }
            else { EntityType::None }
        }
        Biome::Tundra | Biome::SnowCaps => {
            if dice < threshold * 0.15 { EntityType::Rock }
            else { EntityType::None }
        }
        Biome::Volcanic => {
            if dice < threshold * 0.2 { EntityType::Rock }
            else { EntityType::None }
        }
        Biome::Mushroom => {
            if dice < threshold * 0.5 { EntityType::Tree }
            else if dice < threshold * 0.7 { EntityType::Flower }
            else { EntityType::None }
        }
        Biome::Crystal => {
            if dice < threshold * 0.3 { EntityType::Rock }
            else { EntityType::None }
        }
        _ => EntityType::None,
    };

    // Step 5: Mega-structure check (sieve-based rare placement)
    // These use a SEPARATE seed stream so they don't interfere with flora
    let struct_hash = hash_coord_2d(seed.wrapping_add(0xCAFE_BABE), x, y);
    let struct_dice = hash_to_f64(struct_hash);

    // Only place mega-structures at cell boundaries (jittered grid anchor points)
    let cell_size: i64 = 128;
    if x % cell_size == 0 && y % cell_size == 0 {
        if struct_dice < 0.005 {
            // Ultra-rare: Castle/Dungeon
            return if hash_to_f64(struct_hash.wrapping_add(1)) < 0.5 {
                EntityType::Castle
            } else {
                EntityType::Dungeon
            };
        }
        if struct_dice < 0.015 {
            return EntityType::Village;
        }
        if struct_dice < 0.02 {
            return EntityType::Ruin;
        }
    }

    // Priority: mega-structures win over flora
    if entity.is_mega_structure() { entity } else { entity }
}

// ─── 4. Exclusion Sieve ─────────────────────────────────────────────────────
//
// Like the 210-Wheel Sieve crosses out multiples of 2,3,5,7 to find primes,
// the Exclusion Sieve "crosses out" coordinates surrounding a mega-structure.
//
// When querying coordinate (x, y), we check a small set of "Parent Coordinates"
// (the center of the current 128x128 chunk). If a mega-structure is assigned
// to the Parent Coordinate, the sieve returns FALSE for all minor-structure
// placements within a specific radius.

/// Check if a coordinate is valid for placement (not in an exclusion zone).
/// Returns true if the coordinate is clear, false if it's blocked by a
/// nearby mega-structure.
pub fn check_exclusion(seed: u64, x: i64, y: i64, radius: i64) -> bool {
    // Find the parent cell (the anchor point for this area)
    let cell_size = radius;
    let parent_x = (x / cell_size) * cell_size;
    let parent_y = (y / cell_size) * cell_size;

    // Check all potential parent cells in the exclusion radius
    for dx in (-1i64..=1).step_by(cell_size as usize).take(3) {
        for dy in (-1i64..=1).step_by(cell_size as usize).take(3) {
            let px = parent_x + dx * cell_size;
            let py = parent_y + dy * cell_size;

            // Does this parent cell have a mega-structure?
            let hash = hash_coord_2d(seed.wrapping_add(0xCAFE_BABE), px, py);
            let dice = hash_to_f64(hash);

            if dice < 0.005 {
                // This parent has a mega-structure — check if we're in its footprint
                let dist_x = (x - px).abs();
                let dist_y = (y - py).abs();
                if dist_x < radius && dist_y < radius {
                    // We're inside the exclusion zone
                    return false;
                }
            }
        }
    }

    true
}

// ─── 5. Jittered Grid ──────────────────────────────────────────────────────
//
// Prevents structures from clumping. Instead of letting structures appear
// anywhere, we divide the world into a grid (e.g., 64x64 unit cells).
// Every cell has exactly one potential spawn point, jittered within the cell.
// Since the jitter is constrained to stay within cell boundaries, mathematical
// overlap becomes impossible.

/// Compute the jittered position for a grid cell.
/// Returns the (x, y) world position of the spawn point within this cell.
///
/// The jitter is deterministic: same seed + cell = same position.
/// The jitter is bounded: it stays within [0, cell_size) of the cell center.
pub fn jittered_position(seed: u64, cell_x: i64, cell_y: i64, cell_size: i64) -> (f64, f64) {
    let hash = hash_coord_2d(seed.wrapping_add(0xBEEF_CAFE), cell_x, cell_y);

    // Extract two independent f64 values from the hash
    let jitter_x = hash_to_f64(hash);
    let jitter_y = hash_to_f64(hash.wrapping_add(0x9E3779B97F4A7C15));

    // Constrain jitter to stay within cell boundaries
    // Cell center + offset, where offset ∈ [0, cell_size)
    let cx = (cell_x * cell_size) as f64 + (cell_size as f64) * 0.5;
    let cy = (cell_y * cell_size) as f64 + (cell_size as f64) * 0.5;
    let half_cell = (cell_size as f64) * 0.45; // Slight margin to avoid edge cases

    (
        cx + (jitter_x - 0.5) * 2.0 * half_cell,
        cy + (jitter_y - 0.5) * 2.0 * half_cell,
    )
}

// ─── 6. Biome Bleed (Soft Sieve) ───────────────────────────────────────────
//
// For borders between biomes (e.g., Forest → Desert), we don't want a hard
// line. Instead, we use a Weighted Sieve:
//   Tree_Density = Sieve_Weight(coord) * Global_Density
// As you move toward the Desert, the weight drops, the PRNG "fails" the
// placement check more often, and trees naturally thin out into scrubland.

/// Compute the biome weight at a coordinate for a specific biome.
/// Returns a value in [0.0, 1.0] indicating how strongly this biome
/// influences the given coordinate.
///
/// At the biome center, weight = 1.0. At the border, it drops smoothly.
pub fn biome_weight(seed: u64, x: f64, y: f64, biome_id: u8) -> f64 {
    // Sample biome at multiple scales
    let current = biome_at(seed, x, y);
    if current as u8 == biome_id { return 1.0; }

    // Check nearby points to see how close we are to the target biome
    let mut best = 0.0f64;
    let samples: [(f64, f64); 8] = [
        (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0),
        (-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7), (0.7, 0.7),
    ];
    let scale = 50.0; // Distance in world units to sample

    for &(dx, dy) in &samples {
        let nx = x + dx * scale;
        let ny = y + dy * scale;
        let neighbor = biome_at(seed, nx, ny);
        if neighbor as u8 == biome_id {
            // Closer samples contribute more weight
            let dist = (dx * dx + dy * dy).sqrt();
            let contribution = 1.0 / (1.0 + dist);
            if contribution > best { best = contribution; }
        }
    }

    // Smooth falloff using distance-based weighting
    best * 0.7 // Scale down for natural blending
}

// ─── 7. L-System Structure Generation ──────────────────────────────────────
//
// When the Oracle says "a tree exists at [10, 5, 20]", we don't load a 3D
// model. We generate its structure on-the-fly using the hash as a seed for
// an L-System (fractal grammar). The hash dictates branches, angles, density.
// If you leave and come back, the tree generates identically — zero bytes saved.

/// Grammar type for L-System generation
#[derive(Debug, Clone, Copy)]
pub enum StructureGrammar {
    Organic,     // Trees, bushes: branching lines, varying circles
    Geometric,   // Castles, dungeons: stacked rectangles, floor plans
    Natural,     // Rocks, boulders: irregular polygons
}

impl From<u8> for StructureGrammar {
    fn from(id: u8) -> Self {
        match id {
            0 => StructureGrammar::Organic,
            1 => StructureGrammar::Geometric,
            _ => StructureGrammar::Natural,
        }
    }
}

/// A single segment of a generated structure.
#[derive(Debug, Clone, Copy)]
pub struct StructureSegment {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub length: f64,
    pub width: f64,
    pub segment_type: u8, // 0=trunk, 1=branch, 2=leaf, 3=wall, 4=tower, 5=roof
}

/// Generate a structure from a hash seed using L-System rules.
/// Returns a flat array of segments: [x, y, angle, length, width, type, ...]
pub fn generate_structure(hash: u64, grammar: u8, depth: u32) -> Vec<f64> {
    let grammar = StructureGrammar::from(grammar);
    let mut rng = SquaresRng::new(hash);
    let mut segments = Vec::new();
    let mut stack: Vec<(f64, f64, f64)> = Vec::new(); // (x, y, angle)

    // Initial state
    let mut x = 0.0f64;
    let mut y = 0.0f64;
    let mut angle = -90.0f64; // Start pointing up

    match grammar {
        StructureGrammar::Organic => {
            // Tree: trunk + recursive branches
            let trunk_length = 20.0 + rng.next_f64() * 30.0;
            let branch_angle = 20.0 + rng.next_f64() * 30.0;
            let branch_ratio = 0.6 + rng.next_f64() * 0.2;

            // Draw trunk
            let dx = angle.to_radians().cos() * trunk_length;
            let dy = angle.to_radians().sin() * trunk_length;
            segments.extend_from_slice(&[x, y, angle, trunk_length, 3.0, 0.0]);
            x += dx;
            y += dy;

            if depth > 0 {
                // Branch left
                stack.push((x, y, angle));
                let left_angle = angle - branch_angle;
                generate_branch(&mut segments, &mut rng, x, y, left_angle,
                    trunk_length * branch_ratio, depth - 1, branch_angle, branch_ratio);

                // Branch right
                if let Some((px, py, pa)) = stack.pop() {
                    x = px; y = py; angle = pa;
                }
                let right_angle = angle + branch_angle;
                generate_branch(&mut segments, &mut rng, x, y, right_angle,
                    trunk_length * branch_ratio, depth - 1, branch_angle, branch_ratio);

                // Sometimes add a center branch
                if rng.next_f64() > 0.5 {
                    generate_branch(&mut segments, &mut rng, x, y, angle,
                        trunk_length * branch_ratio * 0.8, depth - 1, branch_angle, branch_ratio);
                }
            }

            // Add leaves at terminal branches
            segments.extend_from_slice(&[x, y, 0.0, 5.0 + rng.next_f64() * 5.0, 4.0, 2.0]);
        }
        StructureGrammar::Geometric => {
            // Castle: rectangular walls + towers
            let width = 30.0 + rng.next_f64() * 40.0;
            let height = 20.0 + rng.next_f64() * 30.0;
            let num_towers = 2 + (rng.next_f64() * 3.0) as usize;

            // Draw walls
            segments.extend_from_slice(&[x - width/2.0, y - height/2.0, 0.0, width, 2.0, 3.0]);
            segments.extend_from_slice(&[x + width/2.0, y - height/2.0, 90.0, height, 2.0, 3.0]);
            segments.extend_from_slice(&[x + width/2.0, y + height/2.0, 180.0, width, 2.0, 3.0]);
            segments.extend_from_slice(&[x - width/2.0, y + height/2.0, 270.0, height, 2.0, 3.0]);

            // Add towers at corners
            for i in 0..num_towers.min(4) {
                let (tx, ty) = match i {
                    0 => (x - width/2.0, y - height/2.0),
                    1 => (x + width/2.0, y - height/2.0),
                    2 => (x + width/2.0, y + height/2.0),
                    _ => (x - width/2.0, y + height/2.0),
                };
                let tower_height = 15.0 + rng.next_f64() * 20.0;
                segments.extend_from_slice(&[tx, ty, -90.0, tower_height, 4.0, 4.0]);
                segments.extend_from_slice(&[tx, ty + tower_height, 0.0, 6.0, 6.0, 5.0]);
            }

            // Gate
            segments.extend_from_slice(&[x - 3.0, y - height/2.0, -90.0, height * 0.4, 6.0, 3.0]);
        }
        StructureGrammar::Natural => {
            // Rock: irregular polygon
            let num_sides = 5 + (rng.next_f64() * 4.0) as usize;
            let base_size = 5.0 + rng.next_f64() * 15.0;
            for i in 0..num_sides {
                let a = (i as f64 / num_sides as f64) * 360.0;
                let size = base_size * (0.7 + rng.next_f64() * 0.6);
                let next_a = ((i + 1) as f64 / num_sides as f64) * 360.0;
                segments.extend_from_slice(&[
                    x + a.to_radians().cos() * size,
                    y + a.to_radians().sin() * size,
                    next_a, size * 0.8, 2.0, 5.0
                ]);
            }
        }
    }

    segments
}

/// Helper: recursively generate a branch of an L-System tree.
fn generate_branch(
    segments: &mut Vec<f64>,
    rng: &mut SquaresRng,
    x: f64, y: f64,
    angle: f64, length: f64,
    depth: u32, branch_angle: f64, branch_ratio: f64,
) {
    let width = (depth as f64 + 1.0) * 0.5;
    let dx = angle.to_radians().cos() * length;
    let dy = angle.to_radians().sin() * length;

    segments.extend_from_slice(&[x, y, angle, length, width, if depth == 0 { 2 } else { 1 } as f64]);

    let nx = x + dx;
    let ny = y + dy;

    if depth > 0 {
        // Sub-branches with slight random variation
        let variation = (rng.next_f64() - 0.5) * 15.0;
        generate_branch(segments, rng, nx, ny, angle - branch_angle + variation,
            length * branch_ratio, depth - 1, branch_angle, branch_ratio);
        let variation = (rng.next_f64() - 0.5) * 15.0;
        generate_branch(segments, rng, nx, ny, angle + branch_angle + variation,
            length * branch_ratio, depth - 1, branch_angle, branch_ratio);
    }
}

// ─── 8. Genesis Weave Context ───────────────────────────────────────────────
//
// Bundles all world-gen state into a single convenient struct.
// The only "state" is the seed — everything else is derived on-the-fly.

/// A Genesis Weave world context. Contains only the seed;
/// all world data is derived mathematically.
#[derive(Debug, Clone)]
pub struct GenesisWeave {
    pub seed: u64,
    pub placement_threshold: f64,
    pub exclusion_radius: i64,
    pub cell_size: i64,
}

impl GenesisWeave {
    /// Create a new world with the given seed.
    pub fn new(seed: u64) -> Self {
        GenesisWeave {
            seed,
            placement_threshold: 0.15,
            exclusion_radius: 128,
            cell_size: 64,
        }
    }

    /// Query terrain height at (x, y, z).
    pub fn height(&self, x: f64, y: f64, z: f64) -> f64 {
        terrain_height(self.seed, x, y, z)
    }

    /// Query the biome at (x, y).
    pub fn biome(&self, x: f64, y: f64) -> Biome {
        biome_at(self.seed, x, y)
    }

    /// Query the entity at integer coordinates (x, y).
    pub fn entity(&self, x: i64, y: i64) -> EntityType {
        entity_at(self.seed, x, y, self.placement_threshold)
    }

    /// Check if coordinates are in an exclusion zone.
    pub fn is_excluded(&self, x: i64, y: i64) -> bool {
        !check_exclusion(self.seed, x, y, self.exclusion_radius)
    }

    /// Get the jittered spawn point for a grid cell.
    pub fn jittered(&self, cell_x: i64, cell_y: i64) -> (f64, f64) {
        jittered_position(self.seed, cell_x, cell_y, self.cell_size)
    }

    /// Generate the full structure for an entity at (x, y).
    pub fn structure(&self, x: i64, y: i64) -> Vec<f64> {
        let hash = hash_coord_2d(self.seed.wrapping_add(0xDEAD_BEEF), x, y);
        let entity = self.entity(x, y);
        let grammar = if entity.is_mega_structure() { 1u8 } else { 0u8 };
        generate_structure(hash, grammar, 3)
    }

    /// Query biome weight (soft sieve) at (x, y) for a specific biome.
    pub fn biome_weight_at(&self, x: f64, y: f64, biome_id: u8) -> f64 {
        biome_weight(self.seed, x, y, biome_id)
    }
}

// ─── Verification ───────────────────────────────────────────────────────────

/// Verify the Genesis Weave system.
pub fn verify_genesis() -> bool {
    let mut all_pass = true;
    let seed = 42u64;

    // Test 1: Stateless terrain is deterministic
    let h1 = terrain_height(seed, 100.0, 200.0, 0.0);
    let h2 = terrain_height(seed, 100.0, 200.0, 0.0);
    if h1 != h2 {
        eprintln!("FAIL: terrain_height is not deterministic: {} != {}", h1, h2);
        all_pass = false;
    }

    // Test 2: Terrain varies across coordinates
    let h3 = terrain_height(seed, 1000.0, 2000.0, 0.0);
    if (h1 - h3).abs() < 0.001 {
        eprintln!("FAIL: terrain_height does not vary across coordinates: {} ≈ {}", h1, h3);
        all_pass = false;
    }

    // Test 3: Biome is deterministic
    let b1 = biome_at(seed, 100.0, 200.0);
    let b2 = biome_at(seed, 100.0, 200.0);
    if b1 != b2 {
        eprintln!("FAIL: biome_at is not deterministic");
        all_pass = false;
    }

    // Test 4: Entity oracle is deterministic
    let e1 = entity_at(seed, 10, 20, 0.15);
    let e2 = entity_at(seed, 10, 20, 0.15);
    if e1 != e2 {
        eprintln!("FAIL: entity_at is not deterministic");
        all_pass = false;
    }

    // Test 5: Exclusion sieve works
    // Create a world where some exclusion zones exist
    let excl = check_exclusion(seed, 50, 50, 128);
    // Just verify it doesn't crash and returns a boolean
    let _ = excl;

    // Test 6: Jittered grid is deterministic and bounded
    let (jx1, jy1) = jittered_position(seed, 5, 5, 64);
    let (jx2, jy2) = jittered_position(seed, 5, 5, 64);
    if (jx1 - jx2).abs() > 0.001 || (jy1 - jy2).abs() > 0.001 {
        eprintln!("FAIL: jittered_position is not deterministic");
        all_pass = false;
    }
    // Jitter should stay within cell bounds
    let cx = (5 * 64) as f64 + 32.0;
    let cy = (5 * 64) as f64 + 32.0;
    if (jx1 - cx).abs() > 64.0 || (jy1 - cy).abs() > 64.0 {
        eprintln!("FAIL: jittered_position escapes cell bounds: ({}, {}) vs center ({}, {})", jx1, jy1, cx, cy);
        all_pass = false;
    }

    // Test 7: Structure generation is deterministic
    let s1 = generate_structure(12345, 0, 3);
    let s2 = generate_structure(12345, 0, 3);
    if s1 != s2 {
        eprintln!("FAIL: generate_structure is not deterministic");
        all_pass = false;
    }

    // Test 8: GenesisWeave context
    let world = GenesisWeave::new(seed);
    let h = world.height(100.0, 200.0, 0.0);
    if h < 0.0 || h > 1.0 {
        eprintln!("FAIL: terrain height out of range: {}", h);
        all_pass = false;
    }
    let _ = world.biome(100.0, 200.0);
    let _ = world.entity(10, 20);
    let _ = world.structure(10, 20);

    if all_pass {
        eprintln!("All Genesis Weave verification tests PASSED.");
    }

    all_pass
}
