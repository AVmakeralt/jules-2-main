// =============================================================================
// std/math — Jules Standard Library: Math Module
//
// Full vector, matrix, quaternion, and transform operations.
// All operations are pure Rust — zero external dependencies.
// Performance: AVX2/FMA-ready via #[inline(always)] hot paths.
// =============================================================================

#![allow(dead_code)]

use std::f64::consts as f64c;

// ─── Public API: called from interp.rs eval_builtin ─────────────────────────

/// Dispatch a math:: builtin call. Returns None if the name is not a math builtin.
pub fn dispatch(
    name: &str,
    args: &[crate::interp::Value],
) -> Option<Result<crate::interp::Value, crate::interp::RuntimeError>> {
    macro_rules! rt_err {
        ($msg:expr) => {
            Some(Err(crate::interp::RuntimeError {
                span: Some(crate::lexer::Span::dummy()),
                message: $msg.to_string(),
            }))
        };
    }

    fn get_num(args: &[crate::interp::Value], idx: usize) -> Option<f64> {
        args.get(idx).and_then(|v| v.as_f64())
    }

    let v = match name {
        // ── Scalar math (many already in interp.rs; extras here) ──────────
        "math::min" | "math::max" | "math::clamp" | "math::sign" | "math::fract" | "math::mix"
        | "math::step" | "math::smoothstep" | "math::smootherstep" | "math::radians"
        | "math::degrees" => dispatch_scalar(name, args, &|i| get_num(args, i)),

        // ── vec2 operations ──────────────────────────────────────────────
        "math::vec2" => {
            let x = get_num(args, 0).unwrap_or(0.0) as f32;
            let y = get_num(args, 1).unwrap_or(0.0) as f32;
            Some(Ok(crate::interp::Value::Vec2([x, y])))
        }
        "math::dot2" => {
            if let (Some(a), Some(b)) = (arg_vec2(0, args), arg_vec2(1, args)) {
                Some(Ok(crate::interp::Value::F32(a[0] * b[0] + a[1] * b[1])))
            } else {
                rt_err!("dot2() requires two vec2 arguments")
            }
        }
        "math::length2" => {
            if let Some(a) = arg_vec2(0, args) {
                Some(Ok(crate::interp::Value::F32(
                    (a[0] * a[0] + a[1] * a[1]).sqrt(),
                )))
            } else {
                rt_err!("length2() requires a vec2 argument")
            }
        }
        "math::distance2" => {
            if let (Some(a), Some(b)) = (arg_vec2(0, args), arg_vec2(1, args)) {
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                Some(Ok(crate::interp::Value::F32((dx * dx + dy * dy).sqrt())))
            } else {
                rt_err!("distance2() requires two vec2 arguments")
            }
        }
        "math::normalize2" => {
            if let Some(a) = arg_vec2(0, args) {
                let l = (a[0] * a[0] + a[1] * a[1]).sqrt();
                if l > 1e-8 {
                    Some(Ok(crate::interp::Value::Vec2([a[0] / l, a[1] / l])))
                } else {
                    rt_err!("normalize2(): zero-length vector")
                }
            } else {
                rt_err!("normalize2() requires a vec2 argument")
            }
        }
        "math::reflect2" => {
            if let (Some(i), Some(n)) = (arg_vec2(0, args), arg_vec2(1, args)) {
                let d = 2.0 * (i[0] * n[0] + i[1] * n[1]);
                Some(Ok(crate::interp::Value::Vec2([
                    i[0] - d * n[0],
                    i[1] - d * n[1],
                ])))
            } else {
                rt_err!("reflect2() requires two vec2 arguments")
            }
        }
        "math::refract2" => {
            if args.len() < 3 {
                return rt_err!("refract2() requires vec2, vec2, float");
            }
            if let (Some(i), Some(n), Some(eta)) =
                (arg_vec2(0, args), arg_vec2(1, args), get_num(args, 2))
            {
                let d = i[0] * n[0] + i[1] * n[1];
                let k = 1.0 - eta as f32 * eta as f32 * (1.0 - d * d);
                if k < 0.0 {
                    return Some(Ok(crate::interp::Value::Vec2([0.0, 0.0])));
                }
                let e = eta as f32;
                Some(Ok(crate::interp::Value::Vec2([
                    e * i[0] - (e * d + k.sqrt()) * n[0],
                    e * i[1] - (e * d + k.sqrt()) * n[1],
                ])))
            } else {
                rt_err!("refract2() requires vec2, vec2, float")
            }
        }

        // ── vec3 operations ──────────────────────────────────────────────
        "math::vec3" => {
            let x = get_num(args, 0).unwrap_or(0.0) as f32;
            let y = get_num(args, 1).unwrap_or(0.0) as f32;
            let z = get_num(args, 2).unwrap_or(0.0) as f32;
            Some(Ok(crate::interp::Value::Vec3([x, y, z])))
        }
        "math::dot3" => {
            if let (Some(a), Some(b)) = (arg_vec3(0, args), arg_vec3(1, args)) {
                Some(Ok(crate::interp::Value::F32(
                    a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
                )))
            } else {
                rt_err!("dot3() requires two vec3 arguments")
            }
        }
        "math::cross3" => {
            if let (Some(a), Some(b)) = (arg_vec3(0, args), arg_vec3(1, args)) {
                Some(Ok(crate::interp::Value::Vec3([
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ])))
            } else {
                rt_err!("cross3() requires two vec3 arguments")
            }
        }
        "math::length3" => {
            if let Some(a) = arg_vec3(0, args) {
                Some(Ok(crate::interp::Value::F32(
                    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt(),
                )))
            } else {
                rt_err!("length3() requires a vec3 argument")
            }
        }
        "math::distance3" => {
            if let (Some(a), Some(b)) = (arg_vec3(0, args), arg_vec3(1, args)) {
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                let dz = a[2] - b[2];
                Some(Ok(crate::interp::Value::F32(
                    (dx * dx + dy * dy + dz * dz).sqrt(),
                )))
            } else {
                rt_err!("distance3() requires two vec3 arguments")
            }
        }
        "math::normalize3" => {
            if let Some(a) = arg_vec3(0, args) {
                let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                if l > 1e-8 {
                    Some(Ok(crate::interp::Value::Vec3([
                        a[0] / l,
                        a[1] / l,
                        a[2] / l,
                    ])))
                } else {
                    rt_err!("normalize3(): zero-length vector")
                }
            } else {
                rt_err!("normalize3() requires a vec3 argument")
            }
        }
        "math::reflect3" => {
            if let (Some(i), Some(n)) = (arg_vec3(0, args), arg_vec3(1, args)) {
                let d = 2.0 * (i[0] * n[0] + i[1] * n[1] + i[2] * n[2]);
                Some(Ok(crate::interp::Value::Vec3([
                    i[0] - d * n[0],
                    i[1] - d * n[1],
                    i[2] - d * n[2],
                ])))
            } else {
                rt_err!("reflect3() requires two vec3 arguments")
            }
        }

        // ── vec4 operations ──────────────────────────────────────────────
        "math::vec4" => {
            let x = get_num(args, 0).unwrap_or(0.0) as f32;
            let y = get_num(args, 1).unwrap_or(0.0) as f32;
            let z = get_num(args, 2).unwrap_or(0.0) as f32;
            let w = get_num(args, 3).unwrap_or(1.0) as f32;
            Some(Ok(crate::interp::Value::Vec4([x, y, z, w])))
        }
        "math::dot4" => {
            if let (Some(a), Some(b)) = (arg_vec4(0, args), arg_vec4(1, args)) {
                Some(Ok(crate::interp::Value::F32(
                    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3],
                )))
            } else {
                rt_err!("dot4() requires two vec4 arguments")
            }
        }
        "math::length4" => {
            if let Some(a) = arg_vec4(0, args) {
                Some(Ok(crate::interp::Value::F32(
                    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt(),
                )))
            } else {
                rt_err!("length4() requires a vec4 argument")
            }
        }
        "math::normalize4" => {
            if let Some(a) = arg_vec4(0, args) {
                let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt();
                if l > 1e-8 {
                    Some(Ok(crate::interp::Value::Vec4([
                        a[0] / l,
                        a[1] / l,
                        a[2] / l,
                        a[3] / l,
                    ])))
                } else {
                    rt_err!("normalize4(): zero-length vector")
                }
            } else {
                rt_err!("normalize4() requires a vec4 argument")
            }
        }

        // ── quaternion operations ────────────────────────────────────────
        "math::quat" => {
            let x = get_num(args, 0).unwrap_or(0.0) as f32;
            let y = get_num(args, 1).unwrap_or(0.0) as f32;
            let z = get_num(args, 2).unwrap_or(0.0) as f32;
            let w = get_num(args, 3).unwrap_or(1.0) as f32;
            Some(Ok(crate::interp::Value::Quat([x, y, z, w])))
        }
        "math::quat_identity" => Some(Ok(crate::interp::Value::Quat([0.0, 0.0, 0.0, 1.0]))),
        "math::quat_mul" => {
            if let (Some(a), Some(b)) = (arg_quat(0, args), arg_quat(1, args)) {
                Some(Ok(crate::interp::Value::Quat([
                    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
                    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
                    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
                ])))
            } else {
                rt_err!("quat_mul() requires two quat arguments")
            }
        }
        "math::quat_conjugate" => {
            if let Some(q) = arg_quat(0, args) {
                Some(Ok(crate::interp::Value::Quat([-q[0], -q[1], -q[2], q[3]])))
            } else {
                rt_err!("quat_conjugate() requires a quat argument")
            }
        }
        "math::quat_inverse" => {
            if let Some(q) = arg_quat(0, args) {
                let len_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
                if len_sq > 1e-8 {
                    let inv = 1.0 / len_sq;
                    Some(Ok(crate::interp::Value::Quat([
                        -q[0] * inv,
                        -q[1] * inv,
                        -q[2] * inv,
                        q[3] * inv,
                    ])))
                } else {
                    rt_err!("quat_inverse(): zero-length quaternion")
                }
            } else {
                rt_err!("quat_inverse() requires a quat argument")
            }
        }
        "math::quat_normalize" => {
            if let Some(q) = arg_quat(0, args) {
                let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                if l > 1e-8 {
                    Some(Ok(crate::interp::Value::Quat([
                        q[0] / l,
                        q[1] / l,
                        q[2] / l,
                        q[3] / l,
                    ])))
                } else {
                    rt_err!("quat_normalize(): zero-length quaternion")
                }
            } else {
                rt_err!("quat_normalize() requires a quat argument")
            }
        }
        "math::quat_axis_angle" => {
            if args.len() < 4 {
                return rt_err!("quat_axis_angle() requires vec3 axis + angle");
            }
            if let (Some(axis), Some(angle)) = (arg_vec3(0, args), get_num(args, 3)) {
                let half = (angle as f32) * 0.5;
                let s = half.sin();
                let l = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
                if l > 1e-8 {
                    let inv = 1.0 / l;
                    Some(Ok(crate::interp::Value::Quat([
                        axis[0] * inv * s,
                        axis[1] * inv * s,
                        axis[2] * inv * s,
                        half.cos(),
                    ])))
                } else {
                    rt_err!("quat_axis_angle(): zero-length axis")
                }
            } else {
                rt_err!("quat_axis_angle() requires vec3 axis + angle")
            }
        }
        "math::quat_rotate_vec3" => {
            if args.len() < 2 {
                return rt_err!("quat_rotate_vec3() requires quat + vec3");
            }
            if let (Some(q), Some(v)) = (arg_quat(0, args), arg_vec3(1, args)) {
                let qv = [v[0], v[1], v[2], 0.0];
                let q_conj = [-q[0], -q[1], -q[2], q[3]];
                let tmp = quat_mul_internal(q, qv);
                let result = quat_mul_internal(tmp, q_conj);
                Some(Ok(crate::interp::Value::Vec3([
                    result[0], result[1], result[2],
                ])))
            } else {
                rt_err!("quat_rotate_vec3() requires quat + vec3")
            }
        }
        "math::quat_slerp" => {
            if args.len() < 3 {
                return rt_err!("quat_slerp() requires two quats + t");
            }
            if let (Some(a), Some(b), Some(t)) =
                (arg_quat(0, args), arg_quat(1, args), get_num(args, 2))
            {
                Some(Ok(crate::interp::Value::Quat(quat_slerp(a, b, t as f32))))
            } else {
                rt_err!("quat_slerp() requires two quats + t")
            }
        }
        "math::quat_from_euler" => {
            if args.len() < 3 {
                return rt_err!("quat_from_euler() requires pitch, yaw, roll");
            }
            if let (Some(pitch), Some(yaw), Some(roll)) =
                (get_num(args, 0), get_num(args, 1), get_num(args, 2))
            {
                let (cp, sp) = ((pitch * 0.5).cos() as f32, (pitch * 0.5).sin() as f32);
                let (cy, sy) = ((yaw * 0.5).cos() as f32, (yaw * 0.5).sin() as f32);
                let (cr, sr) = ((roll * 0.5).cos() as f32, (roll * 0.5).sin() as f32);
                Some(Ok(crate::interp::Value::Quat([
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                    cr * cp * cy + sr * sp * sy,
                ])))
            } else {
                rt_err!("quat_from_euler() requires pitch, yaw, roll")
            }
        }

        // ── mat4 operations ──────────────────────────────────────────────
        "math::mat4_identity" => Some(Ok(crate::interp::Value::Mat4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]))),
        "math::mat4_perspective" => {
            if args.len() < 4 {
                return rt_err!("mat4_perspective() requires fov_y, aspect, near, far");
            }
            if let (Some(fov), Some(aspect), Some(near), Some(far)) = (
                get_num(args, 0),
                get_num(args, 1),
                get_num(args, 2),
                get_num(args, 3),
            ) {
                Some(Ok(crate::interp::Value::Mat4(make_perspective(
                    fov as f32,
                    aspect as f32,
                    near as f32,
                    far as f32,
                ))))
            } else {
                rt_err!("mat4_perspective() requires fov_y, aspect, near, far")
            }
        }
        "math::mat4_look_at" => {
            if args.len() < 6 {
                return rt_err!("mat4_look_at() requires eye/target/up (6 floats)");
            }
            if let (Some(eyex), Some(eyey), Some(eyez), Some(tx), Some(ty), Some(tz)) = (
                get_num(args, 0),
                get_num(args, 1),
                get_num(args, 2),
                get_num(args, 3),
                get_num(args, 4),
                get_num(args, 5),
            ) {
                let upx = get_num(args, 6).unwrap_or(0.0) as f32;
                let upy = get_num(args, 7).unwrap_or(1.0) as f32;
                let upz = get_num(args, 8).unwrap_or(0.0) as f32;
                let eye = [eyex as f32, eyey as f32, eyez as f32];
                let target = [tx as f32, ty as f32, tz as f32];
                Some(Ok(crate::interp::Value::Mat4(make_look_at(
                    eye,
                    target,
                    [upx, upy, upz],
                ))))
            } else {
                rt_err!("mat4_look_at() requires eye/target/up")
            }
        }
        "math::mat4_translate" => {
            if args.len() < 3 {
                return rt_err!("mat4_translate() requires x, y, z");
            }
            if let (Some(x), Some(y), Some(z)) =
                (get_num(args, 0), get_num(args, 1), get_num(args, 2))
            {
                Some(Ok(crate::interp::Value::Mat4(make_translate(
                    x as f32, y as f32, z as f32,
                ))))
            } else {
                rt_err!("mat4_translate() requires x, y, z")
            }
        }
        "math::mat4_scale" => {
            if args.len() < 3 {
                return rt_err!("mat4_scale() requires x, y, z");
            }
            if let (Some(x), Some(y), Some(z)) =
                (get_num(args, 0), get_num(args, 1), get_num(args, 2))
            {
                Some(Ok(crate::interp::Value::Mat4(make_scale(
                    x as f32, y as f32, z as f32,
                ))))
            } else {
                rt_err!("mat4_scale() requires x, y, z")
            }
        }
        "math::mat4_rotate_x" => {
            if let Some(a) = get_num(args, 0) {
                Some(Ok(crate::interp::Value::Mat4(make_rot_x(a as f32))))
            } else {
                rt_err!("mat4_rotate_x() requires angle")
            }
        }
        "math::mat4_rotate_y" => {
            if let Some(a) = get_num(args, 0) {
                Some(Ok(crate::interp::Value::Mat4(make_rot_y(a as f32))))
            } else {
                rt_err!("mat4_rotate_y() requires angle")
            }
        }
        "math::mat4_rotate_z" => {
            if let Some(a) = get_num(args, 0) {
                Some(Ok(crate::interp::Value::Mat4(make_rot_z(a as f32))))
            } else {
                rt_err!("mat4_rotate_z() requires angle")
            }
        }
        "math::mat4_mul" => {
            if let (Some(a), Some(b)) = (arg_mat4(0, args), arg_mat4(1, args)) {
                Some(Ok(crate::interp::Value::Mat4(mat4_mul(a, b))))
            } else {
                rt_err!("mat4_mul() requires two mat4 arguments")
            }
        }
        "math::mat4_mul_vec4" => {
            if let (Some(m), Some(v)) = (arg_mat4(0, args), arg_vec4(1, args)) {
                Some(Ok(crate::interp::Value::Vec4(mat4_mul_vec4(m, v))))
            } else {
                rt_err!("mat4_mul_vec4() requires mat4 + vec4")
            }
        }
        "math::mat4_inverse" => {
            if let Some(m) = arg_mat4(0, args) {
                if let Some(inv) = mat4_inverse(m) {
                    Some(Ok(crate::interp::Value::Mat4(inv)))
                } else {
                    rt_err!("mat4_inverse(): matrix is singular")
                }
            } else {
                rt_err!("mat4_inverse() requires a mat4 argument")
            }
        }
        "math::mat4_transpose" => {
            if let Some(m) = arg_mat4(0, args) {
                Some(Ok(crate::interp::Value::Mat4(mat4_transpose(m))))
            } else {
                rt_err!("mat4_transpose() requires a mat4 argument")
            }
        }

        // ── Transform (composite) ────────────────────────────────────────
        "math::transform" => {
            // transform(pos: vec3, rot: quat, scale: vec3) -> mat4
            if args.len() < 3 {
                return rt_err!("transform() requires pos: vec3, rot: quat, scale: vec3");
            }
            if let (Some(pos), Some(rot), Some(scale)) =
                (arg_vec3(0, args), arg_quat(1, args), arg_vec3(2, args))
            {
                let t = make_translate(pos[0], pos[1], pos[2]);
                let r = quat_to_mat4(rot);
                let s = make_scale(scale[0], scale[1], scale[2]);
                Some(Ok(crate::interp::Value::Mat4(mat4_mul(mat4_mul(t, r), s))))
            } else {
                rt_err!("transform() requires pos: vec3, rot: quat, scale: vec3")
            }
        }
        "math::transform_from_mat4" => {
            if let Some(m) = arg_mat4(0, args) {
                let pos = [m[3][0], m[3][1], m[3][2]];
                let scale = [
                    (m[0][0] * m[0][0] + m[1][0] * m[1][0] + m[2][0] * m[2][0]).sqrt(),
                    (m[0][1] * m[0][1] + m[1][1] * m[1][1] + m[2][1] * m[2][1]).sqrt(),
                    (m[0][2] * m[0][2] + m[1][2] * m[1][2] + m[2][2] * m[2][2]).sqrt(),
                ];
                // Extract rotation for quaternion
                let rm = [
                    [m[0][0] / scale[0], m[0][1] / scale[0], m[0][2] / scale[0]],
                    [m[1][0] / scale[1], m[1][1] / scale[1], m[1][2] / scale[1]],
                    [m[2][0] / scale[2], m[2][1] / scale[2], m[2][2] / scale[2]],
                ];
                let q = mat3_to_quat(rm);
                let tup = crate::interp::Value::Tuple(vec![
                    crate::interp::Value::Vec3(pos),
                    crate::interp::Value::Quat(q),
                    crate::interp::Value::Vec3(scale),
                ]);
                Some(Ok(tup))
            } else {
                rt_err!("transform_from_mat4() requires a mat4 argument")
            }
        }

        // ── Additional vector helpers ────────────────────────────────────
        "math::lerp" | "mix" => {
            if args.len() < 3 {
                return rt_err!("lerp() requires a, b, t");
            }
            match (args.get(0), args.get(1), get_num(args, 2)) {
                (
                    Some(crate::interp::Value::F32(a)),
                    Some(crate::interp::Value::F32(b)),
                    Some(t),
                ) => Some(Ok(crate::interp::Value::F32(a + (b - a) * t as f32))),
                (
                    Some(crate::interp::Value::Vec2(a)),
                    Some(crate::interp::Value::Vec2(b)),
                    Some(t),
                ) => {
                    let t = t as f32;
                    Some(Ok(crate::interp::Value::Vec2([
                        a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t,
                    ])))
                }
                (
                    Some(crate::interp::Value::Vec3(a)),
                    Some(crate::interp::Value::Vec3(b)),
                    Some(t),
                ) => {
                    let t = t as f32;
                    Some(Ok(crate::interp::Value::Vec3([
                        a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t,
                        a[2] + (b[2] - a[2]) * t,
                    ])))
                }
                (
                    Some(crate::interp::Value::Vec4(a)),
                    Some(crate::interp::Value::Vec4(b)),
                    Some(t),
                ) => {
                    let t = t as f32;
                    Some(Ok(crate::interp::Value::Vec4([
                        a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t,
                        a[2] + (b[2] - a[2]) * t,
                        a[3] + (b[3] - a[3]) * t,
                    ])))
                }
                (
                    Some(crate::interp::Value::Quat(a)),
                    Some(crate::interp::Value::Quat(b)),
                    Some(t),
                ) => Some(Ok(crate::interp::Value::Quat(quat_slerp(*a, *b, t as f32)))),
                _ => rt_err!("lerp() requires matching types for a, b + float t"),
            }
        }
        "math::prime_count_segmented" => {
            let limit = args.first().and_then(|v| v.as_i64()).unwrap_or(0);
            let seg_size = args.get(1).and_then(|v| v.as_i64()).unwrap_or(1 << 20);
            if limit < 0 {
                return rt_err!("prime_count_segmented(limit): limit must be >= 0");
            }
            if seg_size <= 0 {
                return rt_err!("prime_count_segmented(limit, seg_size): seg_size must be > 0");
            }
            let count = prime_count_segmented(limit as usize, seg_size as usize);
            Some(Ok(crate::interp::Value::I64(count as i64)))
        }

        _ => None,
    };

    // ── Fallback: try scalar helpers ─────────────────────────────────────
    if v.is_none() {
        dispatch_scalar(name, args, &|i| get_num(args, i))
    } else {
        v
    }
}

// ─── Scalar math helpers ────────────────────────────────────────────────────

fn dispatch_scalar(
    name: &str,
    args: &[crate::interp::Value],
    num_arg: &impl Fn(usize) -> Option<f64>,
) -> Option<Result<crate::interp::Value, crate::interp::RuntimeError>> {
    macro_rules! rt_err {
        ($msg:expr) => {
            Some(Err(crate::interp::RuntimeError {
                span: Some(crate::lexer::Span::dummy()),
                message: $msg.to_string(),
            }))
        };
    }

    match name {
        "math::min" => {
            if let (Some(a), Some(b)) = (num_arg(0), num_arg(1)) {
                Some(Ok(crate::interp::Value::F32(a.min(b) as f32)))
            } else {
                rt_err!("min() requires two numbers")
            }
        }
        "math::max" => {
            if let (Some(a), Some(b)) = (num_arg(0), num_arg(1)) {
                Some(Ok(crate::interp::Value::F32(a.max(b) as f32)))
            } else {
                rt_err!("max() requires two numbers")
            }
        }
        "math::clamp" => {
            if let (Some(v), Some(lo), Some(hi)) = (num_arg(0), num_arg(1), num_arg(2)) {
                Some(Ok(crate::interp::Value::F32(v.max(lo).min(hi) as f32)))
            } else {
                rt_err!("clamp() requires three numbers")
            }
        }
        "math::sign" => {
            if let Some(v) = num_arg(0) {
                Some(Ok(crate::interp::Value::F32(if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                })))
            } else {
                rt_err!("sign() requires a number")
            }
        }
        "math::fract" => {
            if let Some(v) = num_arg(0) {
                Some(Ok(crate::interp::Value::F32((v - v.floor()) as f32)))
            } else {
                rt_err!("fract() requires a number")
            }
        }
        "math::mix" => {
            if let (Some(a), Some(b), Some(t)) = (num_arg(0), num_arg(1), num_arg(2)) {
                Some(Ok(crate::interp::Value::F32((a + (b - a) * t) as f32)))
            } else {
                rt_err!("mix() requires a, b, t")
            }
        }
        "math::step" => {
            if let (Some(edge), Some(x)) = (num_arg(0), num_arg(1)) {
                Some(Ok(crate::interp::Value::F32(if x < edge {
                    0.0
                } else {
                    1.0
                })))
            } else {
                rt_err!("step() requires edge, x")
            }
        }
        "math::smoothstep" => {
            if let (Some(edge0), Some(edge1), Some(x)) = (num_arg(0), num_arg(1), num_arg(2)) {
                let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
                Some(Ok(crate::interp::Value::F32(
                    (t * t * (3.0 - 2.0 * t)) as f32,
                )))
            } else {
                rt_err!("smoothstep() requires edge0, edge1, x")
            }
        }
        "math::smootherstep" => {
            if let (Some(edge0), Some(edge1), Some(x)) = (num_arg(0), num_arg(1), num_arg(2)) {
                let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
                Some(Ok(crate::interp::Value::F32(
                    (t * t * t * (t * (t * 6.0 - 15.0) + 10.0)) as f32,
                )))
            } else {
                rt_err!("smootherstep() requires edge0, edge1, x")
            }
        }
        "math::radians" => {
            if let Some(d) = num_arg(0) {
                Some(Ok(crate::interp::Value::F32((d * f64c::PI / 180.0) as f32)))
            } else {
                rt_err!("radians() requires degrees")
            }
        }
        "math::degrees" => {
            if let Some(r) = num_arg(0) {
                Some(Ok(crate::interp::Value::F32((r * 180.0 / f64c::PI) as f32)))
            } else {
                rt_err!("degrees() requires radians")
            }
        }
        _ => None,
    }
}

// ─── Argument extraction helpers ────────────────────────────────────────────

fn arg_vec2(i: usize, args: &[crate::interp::Value]) -> Option<[f32; 2]> {
    match args.get(i) {
        Some(crate::interp::Value::Vec2(v)) => Some(*v),
        _ => None,
    }
}

fn arg_vec3(i: usize, args: &[crate::interp::Value]) -> Option<[f32; 3]> {
    match args.get(i) {
        Some(crate::interp::Value::Vec3(v)) => Some(*v),
        _ => None,
    }
}

fn arg_vec4(i: usize, args: &[crate::interp::Value]) -> Option<[f32; 4]> {
    match args.get(i) {
        Some(crate::interp::Value::Vec4(v)) => Some(*v),
        _ => None,
    }
}

fn arg_quat(i: usize, args: &[crate::interp::Value]) -> Option<[f32; 4]> {
    match args.get(i) {
        Some(crate::interp::Value::Quat(v)) => Some(*v),
        _ => None,
    }
}

fn arg_mat4(i: usize, args: &[crate::interp::Value]) -> Option<[[f32; 4]; 4]> {
    match args.get(i) {
        Some(crate::interp::Value::Mat4(m)) => Some(*m),
        _ => None,
    }
}

fn prime_count_segmented(limit: usize, seg_size: usize) -> usize {
    if limit < 2 {
        return 0;
    }

    let sqrt_limit = (limit as f64).sqrt() as usize + 1;
    let mut is_prime_small = vec![true; sqrt_limit + 1];
    is_prime_small[0] = false;
    is_prime_small[1] = false;
    for p in 2..=sqrt_limit {
        if is_prime_small[p] {
            let mut multiple = p * p;
            while multiple <= sqrt_limit {
                is_prime_small[multiple] = false;
                multiple += p;
            }
        }
    }
    let small_primes: Vec<usize> = (2..=sqrt_limit).filter(|&p| is_prime_small[p]).collect();
    let mut count = small_primes.len();

    let odd_seg = seg_size.max(1024);
    let mut segment = vec![0u8; odd_seg.div_ceil(8)];

    let mut low = sqrt_limit + 1;
    if low % 2 == 0 {
        low += 1;
    }

    while low <= limit {
        let high = (low + odd_seg * 2 - 1).min(limit);
        let seg_len_odd = ((high - low) / 2) + 1;
        let bytes_needed = seg_len_odd.div_ceil(8);
        segment[..bytes_needed].fill(0);

        for &p in &small_primes {
            if p == 2 {
                continue;
            }
            let start = if low <= p { p * p } else { low.div_ceil(p) * p };
            let start = if start % 2 == 0 { start + p } else { start };
            let mut multiple = start;
            while multiple <= high {
                let idx = (multiple - low) / 2;
                segment[idx >> 3] |= 1 << (idx & 7);
                multiple += p * 2;
            }
        }

        for idx in 0..seg_len_odd {
            if (segment[idx >> 3] & (1 << (idx & 7))) == 0 {
                count += 1;
            }
        }
        low += odd_seg * 2;
    }
    count
}

// ─── Quaternion helpers ─────────────────────────────────────────────────────

fn quat_mul_internal(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn quat_slerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let (b, d) = if dot < 0.0 {
        ([-b[0], -b[1], -b[2], -b[3]], -dot)
    } else {
        (b, dot)
    };
    if d > 0.9995 {
        // Fall back to lerp for near-parallel quaternions
        let s = 1.0 - t;
        let l = (
            a[0] * s + b[0] * t,
            a[1] * s + b[1] * t,
            a[2] * s + b[2] * t,
            a[3] * s + b[3] * t,
        );
        let len = (l.0 * l.0 + l.1 * l.1 + l.2 * l.2 + l.3 * l.3).sqrt();
        if len > 1e-8 {
            [l.0 / len, l.1 / len, l.2 / len, l.3 / len]
        } else {
            a
        }
    } else {
        let theta = d.acos();
        let sin_t = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_t;
        let wb = (t * theta).sin() / sin_t;
        [
            a[0] * wa + b[0] * wb,
            a[1] * wa + b[1] * wb,
            a[2] * wa + b[2] * wb,
            a[3] * wa + b[3] * wb,
        ]
    }
}

// ─── Matrix helpers ─────────────────────────────────────────────────────────

fn make_perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, -1.0],
        [0.0, 0.0, 2.0 * far * near * nf, 0.0],
    ]
}

fn make_look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let mut z = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]];
    let l = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2]).sqrt();
    if l > 1e-8 {
        z[0] /= l;
        z[1] /= l;
        z[2] /= l;
    }
    let mut x = [
        up[1] * z[2] - up[2] * z[1],
        up[2] * z[0] - up[0] * z[2],
        up[0] * z[1] - up[1] * z[0],
    ];
    let l = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    if l > 1e-8 {
        x[0] /= l;
        x[1] /= l;
        x[2] /= l;
    }
    let y = [
        z[1] * x[2] - z[2] * x[1],
        z[2] * x[0] - z[0] * x[2],
        z[0] * x[1] - z[1] * x[0],
    ];
    [
        [x[0], y[0], z[0], 0.0],
        [x[1], y[1], z[1], 0.0],
        [x[2], y[2], z[2], 0.0],
        [
            -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
            -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
            -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]),
            1.0,
        ],
    ]
}

fn make_translate(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0],
    ]
}

fn make_scale(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [x, 0.0, 0.0, 0.0],
        [0.0, y, 0.0, 0.0],
        [0.0, 0.0, z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn make_rot_x(a: f32) -> [[f32; 4]; 4] {
    let (s, c) = (a.sin(), a.cos());
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, s, 0.0],
        [0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn make_rot_y(a: f32) -> [[f32; 4]; 4] {
    let (s, c) = (a.sin(), a.cos());
    [
        [c, 0.0, -s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn make_rot_z(a: f32) -> [[f32; 4]; 4] {
    let (s, c) = (a.sin(), a.cos());
    [
        [c, s, 0.0, 0.0],
        [-s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn quat_to_mat4(q: [f32; 4]) -> [[f32; 4]; 4] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut r = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    r
}

fn mat4_mul_vec4(m: [[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
        m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3],
    ]
}

fn mat4_transpose(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut r = m;
    for i in 0..4 {
        for j in i + 1..4 {
            r[i][j] = m[j][i];
            r[j][i] = m[i][j];
        }
    }
    r
}

fn mat4_inverse(m: [[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    let mut inv = [[0.0; 4]; 4];
    inv[0][0] =
        m[1][1] * m[2][2] * m[3][3] - m[1][1] * m[2][3] * m[3][2] - m[2][1] * m[1][2] * m[3][3]
            + m[2][1] * m[1][3] * m[3][2]
            + m[3][1] * m[1][2] * m[2][3]
            - m[3][1] * m[1][3] * m[2][2];
    inv[1][0] =
        -m[1][0] * m[2][2] * m[3][3] + m[1][0] * m[2][3] * m[3][2] + m[2][0] * m[1][2] * m[3][3]
            - m[2][0] * m[1][3] * m[3][2]
            - m[3][0] * m[1][2] * m[2][3]
            + m[3][0] * m[1][3] * m[2][2];
    inv[2][0] =
        m[1][0] * m[2][1] * m[3][3] - m[1][0] * m[2][3] * m[3][1] - m[2][0] * m[1][1] * m[3][3]
            + m[2][0] * m[1][3] * m[3][1]
            + m[3][0] * m[1][1] * m[2][3]
            - m[3][0] * m[1][3] * m[2][1];
    inv[3][0] =
        -m[1][0] * m[2][1] * m[3][2] + m[1][0] * m[2][2] * m[3][1] + m[2][0] * m[1][1] * m[3][2]
            - m[2][0] * m[1][2] * m[3][1]
            - m[3][0] * m[1][1] * m[2][2]
            + m[3][0] * m[1][2] * m[2][1];
    inv[0][1] =
        -m[0][1] * m[2][2] * m[3][3] + m[0][1] * m[2][3] * m[3][2] + m[2][1] * m[0][2] * m[3][3]
            - m[2][1] * m[0][3] * m[3][2]
            - m[3][1] * m[0][2] * m[2][3]
            + m[3][1] * m[0][3] * m[2][2];
    inv[1][1] =
        m[0][0] * m[2][2] * m[3][3] - m[0][0] * m[2][3] * m[3][2] - m[2][0] * m[0][2] * m[3][3]
            + m[2][0] * m[0][3] * m[3][2]
            + m[3][0] * m[0][2] * m[2][3]
            - m[3][0] * m[0][3] * m[2][2];
    inv[2][1] =
        -m[0][0] * m[2][1] * m[3][3] + m[0][0] * m[2][3] * m[3][1] + m[2][0] * m[0][1] * m[3][3]
            - m[2][0] * m[0][3] * m[3][1]
            - m[3][0] * m[0][1] * m[2][3]
            + m[3][0] * m[0][3] * m[2][1];
    inv[3][1] =
        m[0][0] * m[2][1] * m[3][2] - m[0][0] * m[2][2] * m[3][1] - m[2][0] * m[0][1] * m[3][2]
            + m[2][0] * m[0][2] * m[3][1]
            + m[3][0] * m[0][1] * m[2][2]
            - m[3][0] * m[0][2] * m[2][1];
    inv[0][2] =
        m[0][1] * m[1][2] * m[3][3] - m[0][1] * m[1][3] * m[3][2] - m[1][1] * m[0][2] * m[3][3]
            + m[1][1] * m[0][3] * m[3][2]
            + m[3][1] * m[0][2] * m[1][3]
            - m[3][1] * m[0][3] * m[1][2];
    inv[1][2] =
        -m[0][0] * m[1][2] * m[3][3] + m[0][0] * m[1][3] * m[3][2] + m[1][0] * m[0][2] * m[3][3]
            - m[1][0] * m[0][3] * m[3][2]
            - m[3][0] * m[0][2] * m[1][3]
            + m[3][0] * m[0][3] * m[1][2];
    inv[2][2] =
        m[0][0] * m[1][1] * m[3][3] - m[0][0] * m[1][3] * m[3][1] - m[1][0] * m[0][1] * m[3][3]
            + m[1][0] * m[0][3] * m[3][1]
            + m[3][0] * m[0][1] * m[1][3]
            - m[3][0] * m[0][3] * m[1][1];
    inv[3][2] =
        -m[0][0] * m[1][1] * m[3][2] + m[0][0] * m[1][2] * m[3][1] + m[1][0] * m[0][1] * m[3][2]
            - m[1][0] * m[0][2] * m[3][1]
            - m[3][0] * m[0][1] * m[1][2]
            + m[3][0] * m[0][2] * m[1][1];
    inv[0][3] =
        -m[0][1] * m[1][2] * m[2][3] + m[0][1] * m[1][3] * m[2][2] + m[1][1] * m[0][2] * m[2][3]
            - m[1][1] * m[0][3] * m[2][2]
            - m[2][1] * m[0][2] * m[1][3]
            + m[2][1] * m[0][3] * m[1][2];
    inv[1][3] =
        m[0][0] * m[1][2] * m[2][3] - m[0][0] * m[1][3] * m[2][2] - m[1][0] * m[0][2] * m[2][3]
            + m[1][0] * m[0][3] * m[2][2]
            + m[2][0] * m[0][2] * m[1][3]
            - m[2][0] * m[0][3] * m[1][2];
    inv[2][3] =
        -m[0][0] * m[1][1] * m[2][3] + m[0][0] * m[1][3] * m[2][1] + m[1][0] * m[0][1] * m[2][3]
            - m[1][0] * m[0][3] * m[2][1]
            - m[2][0] * m[0][1] * m[1][3]
            + m[2][0] * m[0][3] * m[1][1];
    inv[3][3] =
        m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[1][2] * m[2][1] - m[1][0] * m[0][1] * m[2][2]
            + m[1][0] * m[0][2] * m[2][1]
            + m[2][0] * m[0][1] * m[1][2]
            - m[2][0] * m[0][2] * m[1][1];

    let det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0] + m[0][3] * inv[3][0];
    if det.abs() < 1e-8 {
        return None;
    }
    let inv_det = 1.0 / det;
    for i in 0..4 {
        for j in 0..4 {
            inv[i][j] *= inv_det;
        }
    }
    Some(inv)
}

fn mat3_to_quat(m: [[f32; 3]; 3]) -> [f32; 4] {
    let tr = m[0][0] + m[1][1] + m[2][2];
    if tr > 0.0 {
        let s = 0.5 / (tr + 1.0).sqrt();
        [
            (m[2][1] - m[1][2]) * s,
            (m[0][2] - m[2][0]) * s,
            (m[1][0] - m[0][1]) * s,
            0.25 / s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
        [
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
        [
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        ]
    } else {
        let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
        [
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s,
        ]
    }
}
