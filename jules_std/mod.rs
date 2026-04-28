// =============================================================================
// std — Jules Standard Library
//
// Module registry: all stdlib dispatch functions centralized here.
// Each submodule exposes `pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>>`.
// =============================================================================

#![allow(dead_code)]

use rustc_hash::FxHashMap;

mod ai;
mod alloc;
mod collections;
mod geometry;
mod math;
mod net;
mod noise;
mod pathfind;
mod random;
mod spatial;

use crate::interp::{RuntimeError, Value};

/// Master dispatch: routes `module::function` calls to the right stdlib module.
pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    // Try each module in order of likely frequency.
    // Most-used modules first for shorter dispatch chains.
    math::dispatch(name, args)
        .or_else(|| geometry::dispatch(name, args))
        .or_else(|| random::dispatch(name, args))
        .or_else(|| noise::dispatch(name, args))
        .or_else(|| spatial::dispatch(name, args))
        .or_else(|| pathfind::dispatch(name, args))
        .or_else(|| ai::dispatch(name, args))
        .or_else(|| net::dispatch(name, args))
        .or_else(|| alloc::dispatch(name, args))
        .or_else(|| collections::dispatch(name, args))
}

/// List all available stdlib modules and functions (for introspection).
pub fn modules_value() -> Value {
    let mut out = FxHashMap::default();

    let math_fns = vec![
        "vec2",
        "vec3",
        "vec4",
        "dot2",
        "dot3",
        "dot4",
        "cross3",
        "length2",
        "length3",
        "length4",
        "distance2",
        "distance3",
        "normalize2",
        "normalize3",
        "normalize4",
        "lerp",
        "mix",
        "reflect2",
        "reflect3",
        "refract2",
        "min",
        "max",
        "clamp",
        "sign",
        "step",
        "smoothstep",
        "smootherstep",
        "radians",
        "degrees",
        "quat",
        "quat_identity",
        "quat_mul",
        "quat_conjugate",
        "quat_inverse",
        "quat_normalize",
        "quat_axis_angle",
        "quat_rotate_vec3",
        "quat_slerp",
        "quat_from_euler",
        "mat4_identity",
        "mat4_perspective",
        "mat4_look_at",
        "mat4_translate",
        "mat4_scale",
        "mat4_rotate_x",
        "mat4_rotate_y",
        "mat4_rotate_z",
        "mat4_mul",
        "mat4_mul_vec4",
        "mat4_inverse",
        "mat4_transpose",
        "transform",
        "transform_from_mat4",
        "prime_count_segmented",
    ];
    out.insert(
        "math".into(),
        Value::Array(std::sync::Arc::new(std::sync::Mutex::new(
            math_fns
                .into_iter()
                .map(|s| Value::Str(format!("math::{}", s)))
                .collect(),
        ))),
    );

    let geom_fns = vec![
        "ray",
        "ray_at",
        "aabb",
        "aabb_contains",
        "aabb_intersects",
        "aabb_ray_intersect",
        "aabb_center",
        "aabb_extent",
        "sphere",
        "sphere_intersects",
        "sphere_ray_intersect",
        "plane",
        "plane_distance",
        "plane_ray_intersect",
        "frustum_from_perspective",
        "frustum_contains_aabb",
        "frustum_contains_sphere",
        "sdf_sphere",
        "sdf_box",
        "sdf_plane",
        "sdf_torus",
        "gjk_intersects",
        "closest_point_on_segment",
        "triangle_normal",
        "point_in_triangle",
    ];
    out.insert(
        "geom".into(),
        Value::Array(std::sync::Arc::new(std::sync::Mutex::new(
            geom_fns
                .into_iter()
                .map(|s| Value::Str(format!("geom::{}", s)))
                .collect(),
        ))),
    );

    let random_fns = vec![
        "seed",
        "rand",
        "rand_f32",
        "rand_f64",
        "rand_range",
        "rand_int",
        "rand_normal",
        "rand_bool",
        "choice",
        "weighted_choice",
        "shuffle",
        "sample",
        "pcg32",
        "pcg32_next",
        "pcg32_next_f32",
        "pcg32_range",
    ];
    out.insert(
        "random".into(),
        Value::Array(std::sync::Arc::new(std::sync::Mutex::new(
            random_fns
                .into_iter()
                .map(|s| Value::Str(format!("random::{}", s)))
                .collect(),
        ))),
    );

    let noise_fns = vec![
        "perlin2",
        "perlin3",
        "value2",
        "value3",
        "simplex2",
        "worley2",
        "worley2_cellular",
        "fbm2",
        "fbm3",
        "turbulence2",
        "ridged2",
    ];
    out.insert(
        "noise".into(),
        Value::Array(std::sync::Arc::new(std::sync::Mutex::new(
            noise_fns
                .into_iter()
                .map(|s| Value::Str(format!("noise::{}", s)))
                .collect(),
        ))),
    );

    let spatial_fns = vec![
        "hash_new",
        "hash_insert",
        "hash_query",
        "hash_clear",
        "quadtree_new",
        "quadtree_insert",
        "quadtree_query",
        "octree_new",
        "octree_insert",
        "octree_query",
        "bvh_new",
        "bvh_build",
        "bvh_query",
    ];
    out.insert(
        "spatial".into(),
        Value::Array(std::sync::Arc::new(std::sync::Mutex::new(
            spatial_fns
                .into_iter()
                .map(|s| Value::Str(format!("spatial::{}", s)))
                .collect(),
        ))),
    );

    let path_fns = vec![
        "astar",
        "dijkstra",
        "flow_field",
        "distance",
        "heuristic_manhattan",
        "heuristic_chebyshev",
        "heuristic_octile",
    ];
    out.insert(
        "path".into(),
        Value::Array(std::sync::Arc::new(std::sync::Mutex::new(
            path_fns
                .into_iter()
                .map(|s| Value::Str(format!("path::{}", s)))
                .collect(),
        ))),
    );

    let ai_fns = vec![
        "seek",
        "flee",
        "arrive",
        "pursue",
        "evade",
        "boid_separation",
        "boid_cohesion",
        "boid_alignment",
        "boid_update",
        "utility_score",
        "bt_sequence",
        "bt_selector",
    ];
    out.insert(
        "ai".into(),
        Value::Array(std::sync::Arc::new(std::sync::Mutex::new(
            ai_fns
                .into_iter()
                .map(|s| Value::Str(format!("ai::{}", s)))
                .collect(),
        ))),
    );

    let net_fns = vec![
        "tcp_listen",
        "tcp_accept",
        "tcp_send",
        "tcp_connect",
        "tcp_client_send",
        "tcp_client_recv",
        "udp_bind",
        "udp_send_to",
        "udp_recv",
        "url_encode",
        "url_decode",
        "parse_url",
    ];
    out.insert(
        "net".into(),
        Value::Array(std::sync::Arc::new(std::sync::Mutex::new(
            net_fns
                .into_iter()
                .map(|s| Value::Str(format!("net::{}", s)))
                .collect(),
        ))),
    );

    let alloc_fns = vec![
        "arena_new",
        "arena_alloc",
        "arena_reset",
        "arena_used",
        "arena_write_f32",
        "arena_read_f32",
        "pool_new",
        "pool_alloc",
        "pool_free",
        "pool_used",
        "slab_new",
        "slab_insert",
        "slab_remove",
        "slab_len",
    ];
    out.insert(
        "alloc".into(),
        Value::Array(std::sync::Arc::new(std::sync::Mutex::new(
            alloc_fns
                .into_iter()
                .map(|s| Value::Str(format!("alloc::{}", s)))
                .collect(),
        ))),
    );

    let coll_fns = vec![
        "mpsc_new",
        "mpsc_push",
        "mpsc_try_pop",
        "mpsc_is_empty",
        "ring_new",
        "ring_push",
        "ring_pop",
        "ring_len",
        "pq_new",
        "pq_push",
        "pq_pop",
        "pq_len",
        "sorted_set_new",
        "sorted_set_insert",
        "sorted_set_remove",
        "sorted_set_contains",
        "sorted_set_len",
        "sorted_set_to_array",
        "par_map",
        "par_reduce",
    ];
    out.insert(
        "collections".into(),
        Value::Array(std::sync::Arc::new(std::sync::Mutex::new(
            coll_fns
                .into_iter()
                .map(|s| Value::Str(format!("collections::{}", s)))
                .collect(),
        ))),
    );

    Value::HashMap(std::sync::Arc::new(std::sync::Mutex::new(out)))
}
