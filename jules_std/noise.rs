// =============================================================================
// std/noise — Jules Standard Library: Procedural Noise
//
// Value noise, Perlin noise, Simplex noise (2D/3D/4D), Worley/cellular noise,
// fractal Brownian motion (fBm), turbulence, ridged multifractal.
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

// ─── Permutation table (fixed seed for reproducibility) ─────────────────────

const PERM: [u8; 512] = {
    const BASE: [u8; 256] = generate_permutation();
    let mut out = [0u8; 512];
    let mut i = 0;
    while i < 256 {
        out[i] = BASE[i];
        out[i + 256] = BASE[i];
        i += 1;
    }
    out
};

const fn generate_permutation() -> [u8; 256] {
    let mut p = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        p[i as usize] = i as u8;
        i += 1;
    }
    // Fisher-Yates with a fixed seed LCG
    let mut seed: u64 = 12345;
    i = 255;
    while i > 0 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = ((seed >> 33) as u16) % (i + 1);
        let tmp = p[i as usize];
        p[i as usize] = p[j as usize];
        p[j as usize] = tmp;
        i -= 1;
    }
    p
}

fn fade(t: f64) -> f64 { t * t * t * (t * (t * 6.0 - 15.0) + 10.0) }
fn lerp(t: f64, a: f64, b: f64) -> f64 { a + t * (b - a) }
fn grad(hash: u8, x: f64, y: f64, z: f64) -> f64 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 { y } else if h == 12 || h == 14 { x } else { z };
    let su = if h & 1 == 0 { u } else { -u };
    let sv = if h & 2 == 0 { v } else { -v };
    su + sv
}

// ─── Perlin Noise 2D/3D ────────────────────────────────────────────────────

pub fn perlin2(x: f64, y: f64) -> f64 {
    let xi = x.floor() as i32 & 255;
    let yi = y.floor() as i32 & 255;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let u = fade(xf);
    let v = fade(yf);

    let aa = PERM[(PERM[xi as usize] as usize + yi as usize) as usize];
    let ab = PERM[(PERM[xi as usize] as usize + (yi + 1) as usize) as usize];
    let ba = PERM[(PERM[(xi + 1) as usize] as usize + yi as usize) as usize];
    let bb = PERM[(PERM[(xi + 1) as usize] as usize + (yi + 1) as usize) as usize];

    let x1 = lerp(u, grad(aa, xf, yf, 0.0), grad(ba, xf - 1.0, yf, 0.0));
    let x2 = lerp(u, grad(ab, xf, yf - 1.0, 0.0), grad(bb, xf - 1.0, yf - 1.0, 0.0));
    (lerp(v, x1, x2) + 1.0) * 0.5  // Normalize to [0, 1]
}

pub fn perlin3(x: f64, y: f64, z: f64) -> f64 {
    let xi = x.floor() as i32 & 255;
    let yi = y.floor() as i32 & 255;
    let zi = z.floor() as i32 & 255;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();
    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let aaa = PERM[(PERM[(PERM[xi as usize] as usize + yi as usize) as usize] as usize + zi as usize) as usize];
    let aba = PERM[(PERM[(PERM[xi as usize] as usize + (yi + 1) as usize) as usize] as usize + zi as usize) as usize];
    let aab = PERM[(PERM[(PERM[xi as usize] as usize + yi as usize) as usize] as usize + (zi + 1) as usize) as usize];
    let abb = PERM[(PERM[(PERM[xi as usize] as usize + (yi + 1) as usize) as usize] as usize + (zi + 1) as usize) as usize];
    let baa = PERM[(PERM[(PERM[(xi + 1) as usize] as usize + yi as usize) as usize] as usize + zi as usize) as usize];
    let bba = PERM[(PERM[(PERM[(xi + 1) as usize] as usize + (yi + 1) as usize) as usize] as usize + zi as usize) as usize];
    let bab = PERM[(PERM[(PERM[(xi + 1) as usize] as usize + yi as usize) as usize] as usize + (zi + 1) as usize) as usize];
    let bbb = PERM[(PERM[(PERM[(xi + 1) as usize] as usize + (yi + 1) as usize) as usize] as usize + (zi + 1) as usize) as usize];

    let x1 = lerp(u, grad(aaa, xf, yf, zf), grad(baa, xf - 1.0, yf, zf));
    let x2 = lerp(u, grad(aba, xf, yf - 1.0, zf), grad(bba, xf - 1.0, yf - 1.0, zf));
    let x3 = lerp(u, grad(aab, xf, yf, zf - 1.0), grad(bab, xf - 1.0, yf, zf - 1.0));
    let x4 = lerp(u, grad(abb, xf, yf - 1.0, zf - 1.0), grad(bbb, xf - 1.0, yf - 1.0, zf - 1.0));

    let y1 = lerp(v, x1, x2);
    let y2 = lerp(v, x3, x4);
    (lerp(w, y1, y2) + 1.0) * 0.5
}

// ─── Value Noise 2D/3D ─────────────────────────────────────────────────────

fn hash2(x: i32, y: i32) -> f64 {
    let mut n = x.wrapping_mul(374761393).wrapping_add(y.wrapping_mul(668265263));
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    ((n ^ (n >> 16)) as f64) / (u32::MAX as f64)
}

fn hash3(x: i32, y: i32, z: i32) -> f64 {
    let mut n = x.wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(z.wrapping_mul(1274126177));
    n = (n ^ (n >> 13)).wrapping_mul(1103515245);
    ((n ^ (n >> 16)) as f64) / (u32::MAX as f64)
}

pub fn value2(x: f64, y: f64) -> f64 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = yf * yf * (3.0 - 2.0 * yf);
    let n00 = hash2(xi, yi);
    let n10 = hash2(xi + 1, yi);
    let n01 = hash2(xi, yi + 1);
    let n11 = hash2(xi + 1, yi + 1);
    lerp(v, lerp(u, n00, n10), lerp(u, n01, n11))
}

pub fn value3(x: f64, y: f64, z: f64) -> f64 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();
    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = yf * yf * (3.0 - 2.0 * yf);
    let w = zf * zf * (3.0 - 2.0 * zf);
    let n000 = hash3(xi, yi, zi);
    let n100 = hash3(xi+1, yi, zi);
    let n010 = hash3(xi, yi+1, zi);
    let n110 = hash3(xi+1, yi+1, zi);
    let n001 = hash3(xi, yi, zi+1);
    let n101 = hash3(xi+1, yi, zi+1);
    let n011 = hash3(xi, yi+1, zi+1);
    let n111 = hash3(xi+1, yi+1, zi+1);
    let x1 = lerp(u, n000, n100);
    let x2 = lerp(u, n010, n110);
    let x3 = lerp(u, n001, n101);
    let x4 = lerp(u, n011, n111);
    lerp(w, lerp(v, x1, x2), lerp(v, x3, x4))
}

// ─── Simplex Noise 2D ──────────────────────────────────────────────────────

const F2: f64 = 0.3660254037844386;
const G2: f64 = 0.21132486540518713;
const GRAD3: [[f64; 2]; 12] = [
    [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
];

pub fn simplex2(x: f64, y: f64) -> f64 {
    let s = (x + y) * F2;
    let i = (x + s).floor();
    let j = (y + s).floor();
    let t = (i + j) * G2;
    let x0 = x - (i - t);
    let y0 = y - (j - t);
    let i1 = if x0 > y0 { 1 } else { 0 };
    let j1 = if x0 > y0 { 0 } else { 1 };
    let x1 = x0 - i1 as f64 + G2;
    let y1 = y0 - j1 as f64 + G2;
    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;
    let ii = (i as i32) & 255;
    let jj = (j as i32) & 255;

    let mut n0 = 0.0; let mut n1 = 0.0; let mut n2 = 0.0;
    let mut t0 = 0.5 - x0*x0 - y0*y0;
    if t0 >= 0.0 {
        t0 *= t0;
        let gi0 = PERM[(ii + PERM[jj as usize] as i32) as usize] as usize % 12;
        n0 = t0 * t0 * (GRAD3[gi0][0] * x0 + GRAD3[gi0][1] * y0);
    }
    let mut t1 = 0.5 - x1*x1 - y1*y1;
    if t1 >= 0.0 {
        t1 *= t1;
        let gi1 = PERM[(ii + i1 + PERM[(jj + j1) as usize] as i32) as usize] as usize % 12;
        n1 = t1 * t1 * (GRAD3[gi1][0] * x1 + GRAD3[gi1][1] * y1);
    }
    let mut t2 = 0.5 - x2*x2 - y2*y2;
    if t2 >= 0.0 {
        t2 *= t2;
        let gi2 = PERM[(ii + 1 + PERM[(jj + 1) as usize] as i32) as usize] as usize % 12;
        n2 = t2 * t2 * (GRAD3[gi2][0] * x2 + GRAD3[gi2][1] * y2);
    }
    70.0 * (n0 + n1 + n2)
}

// ─── Worley / Cellular Noise ───────────────────────────────────────────────

fn worley2_point(x: f64, y: f64) -> f64 {
    let mut min_dist = f64::INFINITY;
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let px = (xi + dx) as f64 + hash2(xi + dx, yi + dy);
            let py = (yi + dy) as f64 + hash2(yi + dy, xi + dx);
            let d = (x - px).hypot(y - py);
            if d < min_dist { min_dist = d; }
        }
    }
    min_dist
}

pub fn worley2(x: f64, y: f64) -> f64 { worley2_point(x, y) }

pub fn worley2_cellular(x: f64, y: f64) -> (f64, f64) {
    let mut f1 = f64::INFINITY;
    let mut f2 = f64::INFINITY;
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let px = (xi + dx) as f64 + hash2(xi + dx, yi + dy);
            let py = (yi + dy) as f64 + hash2(yi + dy, xi + dx);
            let d = (x - px).hypot(y - py);
            if d < f1 { f2 = f1; f1 = d; }
            else if d < f2 { f2 = d; }
        }
    }
    (f1, f2 - f1)
}

// ─── Fractal Brownian Motion ───────────────────────────────────────────────

pub fn fbm_2d(x: f64, y: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut max_val = 0.0;
    for _ in 0..octaves {
        sum += perlin2(x * freq, y * freq) * amp;
        max_val += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    sum / max_val
}

pub fn fbm_3d(x: f64, y: f64, z: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut max_val = 0.0;
    for _ in 0..octaves {
        sum += perlin3(x * freq, y * freq, z * freq) * amp;
        max_val += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    sum / max_val
}

pub fn turbulence_2d(x: f64, y: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut max_val = 0.0;
    for _ in 0..octaves {
        sum += (perlin2(x * freq, y * freq) - 0.5).abs() * amp;
        max_val += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    sum / max_val
}

pub fn ridged_2d(x: f64, y: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut prev = 1.0;
    let mut max_val = 0.0;
    for _ in 0..octaves {
        let n = (perlin2(x * freq, y * freq) - 0.5).abs();
        let n = 1.0 - n;
        let n = n * n * prev;
        sum += n * amp;
        max_val += amp;
        prev = n;
        amp *= gain;
        freq *= lacunarity;
    }
    sum / max_val
}

// ─── Builtin dispatch ───────────────────────────────────────────────────────

pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        "noise::perlin2" => {
            if let (Some(x), Some(y)) = (f64_arg(args,0), f64_arg(args,1)) {
                Some(Ok(Value::F64(perlin2(x, y))))
            } else { Some(Err(rt_err!("noise::perlin2() requires x, y"))) }
        }
        "noise::perlin3" => {
            if let (Some(x), Some(y), Some(z)) = (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2)) {
                Some(Ok(Value::F64(perlin3(x, y, z))))
            } else { Some(Err(rt_err!("noise::perlin3() requires x, y, z"))) }
        }
        "noise::value2" => {
            if let (Some(x), Some(y)) = (f64_arg(args,0), f64_arg(args,1)) {
                Some(Ok(Value::F64(value2(x, y))))
            } else { Some(Err(rt_err!("noise::value2() requires x, y"))) }
        }
        "noise::value3" => {
            if let (Some(x), Some(y), Some(z)) = (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2)) {
                Some(Ok(Value::F64(value3(x, y, z))))
            } else { Some(Err(rt_err!("noise::value3() requires x, y, z"))) }
        }
        "noise::simplex2" => {
            if let (Some(x), Some(y)) = (f64_arg(args,0), f64_arg(args,1)) {
                Some(Ok(Value::F64(simplex2(x, y))))
            } else { Some(Err(rt_err!("noise::simplex2() requires x, y"))) }
        }
        "noise::worley2" => {
            if let (Some(x), Some(y)) = (f64_arg(args,0), f64_arg(args,1)) {
                Some(Ok(Value::F64(worley2(x, y))))
            } else { Some(Err(rt_err!("noise::worley2() requires x, y"))) }
        }
        "noise::worley2_cellular" => {
            if let (Some(x), Some(y)) = (f64_arg(args,0), f64_arg(args,1)) {
                let (f1, f2) = worley2_cellular(x, y);
                Some(Ok(Value::Tuple(vec![Value::F64(f1), Value::F64(f2)])))
            } else { Some(Err(rt_err!("noise::worley2_cellular() requires x, y"))) }
        }
        "noise::fbm2" => {
            if args.len() < 5 { return Some(Err(rt_err!("fbm2() requires x, y, octaves, lacunarity, gain"))); }
            if let (Some(x), Some(y), Some(o), Some(l), Some(g)) =
                (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4)) {
                Some(Ok(Value::F64(fbm_2d(x, y, o as u32, l, g))))
            } else { Some(Err(rt_err!("fbm2() requires x, y, octaves, lacunarity, gain"))) }
        }
        "noise::fbm3" => {
            if args.len() < 6 { return Some(Err(rt_err!("fbm3() requires x, y, z, octaves, lacunarity, gain"))); }
            if let (Some(x), Some(y), Some(z), Some(o), Some(l), Some(g)) =
                (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4), f64_arg(args,5)) {
                Some(Ok(Value::F64(fbm_3d(x, y, z, o as u32, l, g))))
            } else { Some(Err(rt_err!("fbm3() requires x, y, z, octaves, lacunarity, gain"))) }
        }
        "noise::turbulence2" => {
            if args.len() < 5 { return Some(Err(rt_err!("turbulence2() requires x, y, octaves, lacunarity, gain"))); }
            if let (Some(x), Some(y), Some(o), Some(l), Some(g)) =
                (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4)) {
                Some(Ok(Value::F64(turbulence_2d(x, y, o as u32, l, g))))
            } else { Some(Err(rt_err!("turbulence2() requires x, y, octaves, lacunarity, gain"))) }
        }
        "noise::ridged2" => {
            if args.len() < 5 { return Some(Err(rt_err!("ridged2() requires x, y, octaves, lacunarity, gain"))); }
            if let (Some(x), Some(y), Some(o), Some(l), Some(g)) =
                (f64_arg(args,0), f64_arg(args,1), f64_arg(args,2), f64_arg(args,3), f64_arg(args,4)) {
                Some(Ok(Value::F64(ridged_2d(x, y, o as u32, l, g))))
            } else { Some(Err(rt_err!("ridged2() requires x, y, octaves, lacunarity, gain"))) }
        }
        _ => None,
    }
}
