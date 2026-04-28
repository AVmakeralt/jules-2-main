// =============================================================================
// std/geometry — Jules Standard Library: Geometry Module
//
// Ray, AABB, Sphere, Plane, Frustum, SDF primitives, collision queries.
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

fn vec3_arg(args: &[Value], i: usize) -> Option<[f32; 3]> {
    match args.get(i) {
        Some(Value::Vec3(v)) => Some(*v),
        _ => None,
    }
}

fn vec2_arg(args: &[Value], i: usize) -> Option<[f32; 2]> {
    match args.get(i) {
        Some(Value::Vec2(v)) => Some(*v),
        _ => None,
    }
}

/// Dispatch a geometry:: builtin. Returns None if not a geometry builtin.
pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    let v = match name {
        // ── Ray ──────────────────────────────────────────────────────────
        "geom::ray" => {
            if args.len() < 6 {
                return Some(Err(rt_err!("geom::ray() requires origin(xyz) + dir(xyz)")));
            }
            let ox = f64_arg(args, 0).unwrap_or(0.0) as f32;
            let oy = f64_arg(args, 1).unwrap_or(0.0) as f32;
            let oz = f64_arg(args, 2).unwrap_or(0.0) as f32;
            let dx = f64_arg(args, 3).unwrap_or(0.0) as f32;
            let dy = f64_arg(args, 4).unwrap_or(0.0) as f32;
            let dz = f64_arg(args, 5).unwrap_or(1.0) as f32;
            // Return as tuple: (origin: vec3, dir: vec3)
            Some(Ok(Value::Tuple(vec![Value::Vec3([ox, oy, oz]), Value::Vec3([dx, dy, dz])])))
        }
        "geom::ray_at" => {
            if args.len() < 2 {
                return Some(Err(rt_err!("geom::ray_at() requires ray, t")));
            }
            let (origin, dir) = match args.get(0) {
                Some(Value::Tuple(t)) if t.len() >= 2 => {
                    if let (Value::Vec3(o), Value::Vec3(d)) = (&t[0], &t[1]) {
                        (*o, *d)
                    } else { return Some(Err(rt_err!("geom::ray_at(): expected (vec3, vec3)"))); }
                }
                _ => return Some(Err(rt_err!("geom::ray_at(): expected ray tuple"))),
            };
            let t = f64_arg(args, 1).unwrap_or(0.0) as f32;
            Some(Ok(Value::Vec3([origin[0]+dir[0]*t, origin[1]+dir[1]*t, origin[2]+dir[2]*t])))
        }

        // ── AABB ─────────────────────────────────────────────────────────
        "geom::aabb" => {
            if args.len() < 6 {
                return Some(Err(rt_err!("geom::aabb() requires min(xyz) + max(xyz)")));
            }
            let min = [f64_arg(args,0).unwrap_or(0.0) as f32,
                       f64_arg(args,1).unwrap_or(0.0) as f32,
                       f64_arg(args,2).unwrap_or(0.0) as f32];
            let max = [f64_arg(args,3).unwrap_or(1.0) as f32,
                       f64_arg(args,4).unwrap_or(1.0) as f32,
                       f64_arg(args,5).unwrap_or(1.0) as f32];
            Some(Ok(Value::Tuple(vec![Value::Vec3(min), Value::Vec3(max)])))
        }
        "geom::aabb_contains" => {
            if args.len() < 2 {
                return Some(Err(rt_err!("geom::aabb_contains() requires aabb, point")));
            }
            if let (Some((min, max)), Some(p)) = (aabb_from_arg(&args[0]), vec3_arg(args, 1)) {
                let inside = p[0]>=min[0] && p[0]<=max[0] && p[1]>=min[1] && p[1]<=max[1] && p[2]>=min[2] && p[2]<=max[2];
                Some(Ok(Value::Bool(inside)))
            } else { Some(Err(rt_err!("geom::aabb_contains() requires aabb, vec3"))) }
        }
        "geom::aabb_intersects" => {
            if args.len() < 2 {
                return Some(Err(rt_err!("geom::aabb_intersects() requires two aabbs")));
            }
            if let (Some((a_min, a_max)), Some((b_min, b_max))) = (aabb_from_arg(&args[0]), aabb_from_arg(&args[1])) {
                let hits = a_min[0]<=b_max[0] && a_max[0]>=b_min[0]
                        && a_min[1]<=b_max[1] && a_max[1]>=b_min[1]
                        && a_min[2]<=b_max[2] && a_max[2]>=b_min[2];
                Some(Ok(Value::Bool(hits)))
            } else { Some(Err(rt_err!("geom::aabb_intersects() requires two aabbs"))) }
        }
        "geom::aabb_ray_intersect" => {
            if args.len() < 2 {
                return Some(Err(rt_err!("geom::aabb_ray_intersect() requires aabb, ray")));
            }
            if let (Some((a_min, a_max)), Some((origin, dir))) = (aabb_from_arg(&args[0]), ray_from_arg(&args[1])) {
                let result = ray_aabb_intersect(origin, dir, a_min, a_max);
                match result {
                    Some(t) => Some(Ok(Value::Tuple(vec![Value::Bool(true), Value::F32(t)]))),
                    None => Some(Ok(Value::Tuple(vec![Value::Bool(false), Value::F32(0.0)]))),
                }
            } else { Some(Err(rt_err!("geom::aabb_ray_intersect() requires aabb, ray"))) }
        }
        "geom::aabb_center" => {
            if let Some((min, max)) = aabb_from_arg(&args[0]) {
                Some(Ok(Value::Vec3([(min[0]+max[0])*0.5, (min[1]+max[1])*0.5, (min[2]+max[2])*0.5])))
            } else { Some(Err(rt_err!("geom::aabb_center() requires aabb"))) }
        }
        "geom::aabb_extent" => {
            if let Some((min, max)) = aabb_from_arg(&args[0]) {
                Some(Ok(Value::Vec3([max[0]-min[0], max[1]-min[1], max[2]-min[2]])))
            } else { Some(Err(rt_err!("geom::aabb_extent() requires aabb"))) }
        }

        // ── Sphere ───────────────────────────────────────────────────────
        "geom::sphere" => {
            if args.len() < 4 {
                return Some(Err(rt_err!("geom::sphere() requires center(xyz) + radius")));
            }
            if let Some(c) = vec3_arg(args, 0) {
                let r = f64_arg(args, 1).unwrap_or(1.0) as f32;
                Some(Ok(Value::Tuple(vec![Value::Vec3(c), Value::F32(r)])))
            } else if args.len() >= 6 {
                let c = [f64_arg(args,0).unwrap_or(0.0) as f32,
                         f64_arg(args,1).unwrap_or(0.0) as f32,
                         f64_arg(args,2).unwrap_or(0.0) as f32];
                let r = f64_arg(args,3).unwrap_or(1.0) as f32;
                Some(Ok(Value::Tuple(vec![Value::Vec3(c), Value::F32(r)])))
            } else { Some(Err(rt_err!("geom::sphere() requires center + radius"))) }
        }
        "geom::sphere_intersects" => {
            if args.len() < 2 { return Some(Err(rt_err!("geom::sphere_intersects() requires two spheres"))); }
            if let (Some((c1, r1)), Some((c2, r2))) = (sphere_from_arg(&args[0]), sphere_from_arg(&args[1])) {
                let dx = c1[0]-c2[0]; let dy = c1[1]-c2[1]; let dz = c1[2]-c2[2];
                let dist_sq = dx*dx + dy*dy + dz*dz;
                Some(Ok(Value::Bool(dist_sq <= (r1+r2)*(r1+r2))))
            } else { Some(Err(rt_err!("geom::sphere_intersects() requires two spheres"))) }
        }
        "geom::sphere_ray_intersect" => {
            if args.len() < 2 { return Some(Err(rt_err!("geom::sphere_ray_intersect() requires sphere, ray"))); }
            if let (Some((center, radius)), Some((origin, dir))) = (sphere_from_arg(&args[0]), ray_from_arg(&args[1])) {
                let result = ray_sphere_intersect(origin, dir, center, radius);
                match result {
                    Some(t) => Some(Ok(Value::Tuple(vec![Value::Bool(true), Value::F32(t)]))),
                    None => Some(Ok(Value::Tuple(vec![Value::Bool(false), Value::F32(0.0)]))),
                }
            } else { Some(Err(rt_err!("geom::sphere_ray_intersect() requires sphere, ray"))) }
        }

        // ── Plane ────────────────────────────────────────────────────────
        "geom::plane" => {
            if args.len() < 4 { return Some(Err(rt_err!("geom::plane() requires normal(xyz) + d"))); }
            if let Some(n) = vec3_arg(args, 0) {
                let d = f64_arg(args, 1).unwrap_or(0.0) as f32;
                Some(Ok(Value::Tuple(vec![Value::Vec3(n), Value::F32(d)])))
            } else { Some(Err(rt_err!("geom::plane() requires normal + d"))) }
        }
        "geom::plane_distance" => {
            if args.len() < 2 { return Some(Err(rt_err!("geom::plane_distance() requires plane, point"))); }
            if let (Some((n, d)), Some(p)) = (plane_from_arg(&args[0]), vec3_arg(args, 1)) {
                let dist = n[0]*p[0] + n[1]*p[1] + n[2]*p[2] + d;
                Some(Ok(Value::F32(dist)))
            } else { Some(Err(rt_err!("geom::plane_distance() requires plane, point"))) }
        }
        "geom::plane_ray_intersect" => {
            if args.len() < 2 { return Some(Err(rt_err!("geom::plane_ray_intersect() requires plane, ray"))); }
            if let (Some((n, d)), Some((origin, dir))) = (plane_from_arg(&args[0]), ray_from_arg(&args[1])) {
                let denom = n[0]*dir[0] + n[1]*dir[1] + n[2]*dir[2];
                if denom.abs() < 1e-8 {
                    Some(Ok(Value::Tuple(vec![Value::Bool(false), Value::F32(0.0)])))
                } else {
                    let t = -(n[0]*origin[0] + n[1]*origin[1] + n[2]*origin[2] + d) / denom;
                    Some(Ok(Value::Tuple(vec![Value::Bool(t >= 0.0), Value::F32(t)])))
                }
            } else { Some(Err(rt_err!("geom::plane_ray_intersect() requires plane, ray"))) }
        }

        // ── Frustum ──────────────────────────────────────────────────────
        "geom::frustum_from_perspective" => {
            if args.len() < 4 { return Some(Err(rt_err!("geom::frustum_from_perspective() requires fov, aspect, near, far"))); }
            if let (Some(fov), Some(aspect), Some(near), Some(far)) =
                (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3)) {
                let planes = frustum_planes(fov as f32, aspect as f32, near as f32, far as f32);
                // Return as array of 6 planes: each plane is (normal: vec3, d: f32)
                let mut out = Vec::with_capacity(12);
                for p in &planes {
                    out.push(Value::Vec3(p.0));
                    out.push(Value::F32(p.1));
                }
                Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(out)))))
            } else { Some(Err(rt_err!("frustum_from_perspective() requires 4 floats"))) }
        }
        "geom::frustum_contains_aabb" => {
            if args.len() < 2 { return Some(Err(rt_err!("frustum_contains_aabb() requires planes, aabb"))); }
            if let (Some(planes), Some((min, max))) = (frustum_planes_from_arg(&args[0]), aabb_from_arg(&args[1])) {
                let inside = frustum_aabb_test(&planes, min, max);
                Some(Ok(Value::Bool(inside)))
            } else { Some(Err(rt_err!("frustum_contains_aabb() requires planes, aabb"))) }
        }
        "geom::frustum_contains_sphere" => {
            if args.len() < 2 { return Some(Err(rt_err!("frustum_contains_sphere() requires planes, sphere"))); }
            if let (Some(planes), Some((center, radius))) = (frustum_planes_from_arg(&args[0]), sphere_from_arg(&args[1])) {
                let inside = frustum_sphere_test(&planes, center, radius);
                Some(Ok(Value::Bool(inside)))
            } else { Some(Err(rt_err!("frustum_contains_sphere() requires planes, sphere"))) }
        }

        // ── SDF Primitives ───────────────────────────────────────────────
        "geom::sdf_sphere" => {
            if args.len() < 2 { return Some(Err(rt_err!("sdf_sphere() requires point, radius"))); }
            if let (Some(p), Some(r)) = (vec3_arg(args, 0), f64_arg(args, 1)) {
                let l = (p[0]*p[0]+p[1]*p[1]+p[2]*p[2]).sqrt();
                Some(Ok(Value::F32(l - r as f32)))
            } else { Some(Err(rt_err!("sdf_sphere() requires vec3 + float"))) }
        }
        "geom::sdf_box" => {
            if args.len() < 2 { return Some(Err(rt_err!("sdf_box() requires point, half_size"))); }
            if let (Some(p), Some(b)) = (vec3_arg(args, 0), vec3_arg(args, 1)) {
                let q = [p[0].abs()-b[0], p[1].abs()-b[1], p[2].abs()-b[2]];
                let outside = q[0].max(0.0).hypot(q[1].max(0.0)).hypot(q[2].max(0.0));
                let inside = q[0].max(q[1].max(q[2])).min(0.0);
                Some(Ok(Value::F32(outside + inside)))
            } else { Some(Err(rt_err!("sdf_box() requires vec3 + vec3"))) }
        }
        "geom::sdf_plane" => {
            if args.len() < 2 { return Some(Err(rt_err!("sdf_plane() requires point, plane_normal"))); }
            if let (Some(p), Some(n)) = (vec3_arg(args, 0), vec3_arg(args, 1)) {
                let l = (n[0]*n[0]+n[1]*n[1]+n[2]*n[2]).sqrt();
                if l > 1e-8 {
                    let nn = [n[0]/l, n[1]/l, n[2]/l];
                    Some(Ok(Value::F32(p[0]*nn[0] + p[1]*nn[1] + p[2]*nn[2])))
                } else { Some(Err(rt_err!("sdf_plane(): zero normal"))) }
            } else { Some(Err(rt_err!("sdf_plane() requires vec3 + vec3"))) }
        }
        "geom::sdf_torus" => {
            if args.len() < 3 { return Some(Err(rt_err!("sdf_torus() requires point, major_r, minor_r"))); }
            if let (Some(p), Some(r1), Some(r2)) = (vec3_arg(args,0), f64_arg(args,1), f64_arg(args,2)) {
                let q = [(p[0]*p[0]+p[2]*p[2]).sqrt()-r1 as f32, p[1]];
                let l = (q[0]*q[0]+q[1]*q[1]).sqrt();
                Some(Ok(Value::F32(l - r2 as f32)))
            } else { Some(Err(rt_err!("sdf_torus() requires vec3 + 2 floats"))) }
        }

        // ── Collision: GJK ───────────────────────────────────────────────
        "geom::gjk_intersects" => {
            if args.len() < 2 { return Some(Err(rt_err!("gjk_intersects() requires two convex shapes"))); }
            // Simplified: for now supports AABB vs AABB, Sphere vs Sphere
            if let (Some((a_min, a_max)), Some((b_min, b_max))) = (aabb_from_arg(&args[0]), aabb_from_arg(&args[1])) {
                let hits = a_min[0]<=b_max[0] && a_max[0]>=b_min[0]
                        && a_min[1]<=b_max[1] && a_max[1]>=b_min[1]
                        && a_min[2]<=b_max[2] && a_max[2]>=b_min[2];
                Some(Ok(Value::Bool(hits)))
            } else if let (Some((c1, r1)), Some((c2, r2))) = (sphere_from_arg(&args[0]), sphere_from_arg(&args[1])) {
                let dx=c1[0]-c2[0]; let dy=c1[1]-c2[1]; let dz=c1[2]-c2[2];
                Some(Ok(Value::Bool(dx*dx+dy*dy+dz*dz <= (r1+r2)*(r1+r2))))
            } else { Some(Err(rt_err!("gjk_intersects(): unsupported shape pair (use aabbs or spheres)"))) }
        }

        // ── Helpers ──────────────────────────────────────────────────────
        "geom::closest_point_on_segment" => {
            if args.len() < 3 { return Some(Err(rt_err!("closest_point_on_segment() requires point, a, b"))); }
            if let (Some(p), Some(a), Some(b)) = (vec3_arg(args,0), vec3_arg(args,1), vec3_arg(args,2)) {
                let ab = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
                let ab_len_sq = ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2];
                if ab_len_sq < 1e-8 { Some(Ok(Value::Vec3(a))) }
                else {
                    let t = ((p[0]-a[0])*ab[0]+(p[1]-a[1])*ab[1]+(p[2]-a[2])*ab[2]) / ab_len_sq;
                    let t = t.clamp(0.0, 1.0);
                    Some(Ok(Value::Vec3([a[0]+ab[0]*t, a[1]+ab[1]*t, a[2]+ab[2]*t])))
                }
            } else { Some(Err(rt_err!("closest_point_on_segment() requires 3 vec3s"))) }
        }
        "geom::triangle_normal" => {
            if args.len() < 3 { return Some(Err(rt_err!("triangle_normal() requires 3 vertices"))); }
            if let (Some(a), Some(b), Some(c)) = (vec3_arg(args,0), vec3_arg(args,1), vec3_arg(args,2)) {
                let e1 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
                let e2 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
                let mut n = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
                let l = (n[0]*n[0]+n[1]*n[1]+n[2]*n[2]).sqrt();
                if l > 1e-8 { n[0]/=l; n[1]/=l; n[2]/=l; }
                Some(Ok(Value::Vec3(n)))
            } else { Some(Err(rt_err!("triangle_normal() requires 3 vec3s"))) }
        }
        "geom::point_in_triangle" => {
            if args.len() < 4 { return Some(Err(rt_err!("point_in_triangle() requires point + 3 vertices"))); }
            if let (Some(p), Some(a), Some(b), Some(c)) = (vec3_arg(args,0), vec3_arg(args,1), vec3_arg(args,2), vec3_arg(args,3)) {
                // Barycentric technique (2D projection on dominant plane)
                let ab = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
                let ac = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
                let ap = [p[0]-a[0], p[1]-a[1], p[2]-a[2]];
                let d1 = ab[0]*ap[1]-ab[1]*ap[0];
                let d2 = ac[0]*ap[1]-ac[1]*ap[0];
                let d3 = ab[0]*ac[1]-ab[1]*ac[0];
                let has_neg = (d1 < 0.0) != (d2 < 0.0) && d1.abs() > 1e-8 && d2.abs() > 1e-8;
                let inside = if d3 > 0.0 { !has_neg } else { d1 <= 0.0 && d2 >= d3 };
                Some(Ok(Value::Bool(inside)))
            } else { Some(Err(rt_err!("point_in_triangle() requires point + 3 vec3s"))) }
        }

        _ => return None,
    };
    v
}

// ─── Internal helpers ───────────────────────────────────────────────────────

fn ray_aabb_intersect(origin: [f32;3], dir: [f32;3], min: [f32;3], max: [f32;3]) -> Option<f32> {
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;
    for i in 0..3 {
        if dir[i].abs() < 1e-8 {
            if origin[i] < min[i] || origin[i] > max[i] { return None; }
        } else {
            let mut t0 = (min[i] - origin[i]) / dir[i];
            let mut t1 = (max[i] - origin[i]) / dir[i];
            if t0 > t1 { std::mem::swap(&mut t0, &mut t1); }
            t_min = t_min.max(t0);
            t_max = t_max.min(t1);
            if t_min > t_max { return None; }
        }
    }
    if t_min < 0.0 { if t_max < 0.0 { return None; } Some(t_max) } else { Some(t_min) }
}

fn ray_sphere_intersect(origin: [f32;3], dir: [f32;3], center: [f32;3], radius: f32) -> Option<f32> {
    let oc = [origin[0]-center[0], origin[1]-center[1], origin[2]-center[2]];
    let a = dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2];
    let b = 2.0*(oc[0]*dir[0]+oc[1]*dir[1]+oc[2]*dir[2]);
    let c = oc[0]*oc[0]+oc[1]*oc[1]+oc[2]*oc[2] - radius*radius;
    let disc = b*b - 4.0*a*c;
    if disc < 0.0 { return None; }
    let sqrt_disc = disc.sqrt();
    let t = (-b - sqrt_disc) / (2.0*a);
    if t >= 0.0 { Some(t) } else {
        let t2 = (-b + sqrt_disc) / (2.0*a);
        if t2 >= 0.0 { Some(t2) } else { None }
    }
}

fn aabb_from_arg(v: &Value) -> Option<([f32;3], [f32;3])> {
    match v {
        Value::Tuple(t) if t.len() >= 2 => {
            if let (Value::Vec3(min), Value::Vec3(max)) = (&t[0], &t[1]) {
                Some((*min, *max))
            } else { None }
        }
        _ => None,
    }
}

fn ray_from_arg(v: &Value) -> Option<([f32;3], [f32;3])> {
    match v {
        Value::Tuple(t) if t.len() >= 2 => {
            if let (Value::Vec3(origin), Value::Vec3(dir)) = (&t[0], &t[1]) {
                Some((*origin, *dir))
            } else { None }
        }
        _ => None,
    }
}

fn sphere_from_arg(v: &Value) -> Option<([f32;3], f32)> {
    match v {
        Value::Tuple(t) if t.len() >= 2 => {
            if let (Value::Vec3(c), Value::F32(r)) = (&t[0], &t[1]) {
                Some((*c, *r))
            } else { None }
        }
        _ => None,
    }
}

fn plane_from_arg(v: &Value) -> Option<([f32;3], f32)> {
    match v {
        Value::Tuple(t) if t.len() >= 2 => {
            if let (Value::Vec3(n), Value::F32(d)) = (&t[0], &t[1]) {
                Some((*n, *d))
            } else { None }
        }
        _ => None,
    }
}

fn frustum_planes_from_arg(v: &Value) -> Option<Vec<([f32;3], f32)>> {
    match v {
        Value::Array(arr) => {
            let arr = arr.lock().ok()?;
            if arr.len() < 12 { return None; }
            let mut planes = Vec::with_capacity(6);
            for i in (0..12).step_by(2) {
                if let (Value::Vec3(n), Value::F32(d)) = (&arr[i], &arr[i+1]) {
                    planes.push((*n, *d));
                } else { return None; }
            }
            Some(planes)
        }
        _ => None,
    }
}

fn frustum_planes(fov_y: f32, aspect: f32, near: f32, far: f32) -> [([f32;3], f32); 6] {
    // Near, Far, Left, Right, Top, Bottom
    let tan_half_fov = (fov_y * 0.5).tan();
    let near_h = near * tan_half_fov;
    let near_w = near_h * aspect;
    let far_h = far * tan_half_fov;
    let far_w = far_h * aspect;

    // Normalized normals for each plane
    let near_plane = ([0.0, 0.0, 1.0], -near);
    let far_plane = ([0.0, 0.0, -1.0], far);

    let l = (near_w*near_w + near*near).sqrt();
    let left_plane = ([near/l, 0.0, near_w/l], 0.0);

    let r = (near_w*near_w + near*near).sqrt();
    let right_plane = ([-near/r, 0.0, near_w/r], 0.0);

    let t = (near_h*near_h + near*near).sqrt();
    let top_plane = ([0.0, -near/t, near_h/t], 0.0);

    let b = (near_h*near_h + near*near).sqrt();
    let bottom_plane = ([0.0, near/b, near_h/b], 0.0);

    [near_plane, far_plane, left_plane, right_plane, top_plane, bottom_plane]
}

fn frustum_aabb_test(planes: &[([f32;3], f32)], min: [f32;3], max: [f32;3]) -> bool {
    for &(n, d) in planes {
        // Find the positive vertex (furthest in the direction of the plane normal)
        let px = if n[0] > 0.0 { max[0] } else { min[0] };
        let py = if n[1] > 0.0 { max[1] } else { min[1] };
        let pz = if n[2] > 0.0 { max[2] } else { min[2] };
        if n[0]*px + n[1]*py + n[2]*pz + d < 0.0 { return false; }
    }
    true
}

fn frustum_sphere_test(planes: &[([f32;3], f32)], center: [f32;3], radius: f32) -> bool {
    for &(n, d) in planes {
        let dist = n[0]*center[0] + n[1]*center[1] + n[2]*center[2] + d;
        if dist < -radius { return false; }
    }
    true
}
