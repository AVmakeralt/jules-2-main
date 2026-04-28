use crate::gpu_backend::{GpuBackend, GpuMemoryManager};
use std::time::{Duration, Instant};

const FEAT_DIM: usize = 8;
const MAX_ABS_FEATURE: f32 = 64.0;
const MAX_ABS_WEIGHT: f32 = 1_000_000.0;
const MAX_ABS_REWARD: f32 = 2.0;
const TACTICAL_LANES: [f32; 4] = [0.00, 0.02, 0.06, 0.10];

#[derive(Clone, Debug)]
struct GradBuffer {
    acc: [f32; FEAT_DIM],
    count: usize,
}

impl GradBuffer {
    #[inline]
    fn new() -> Self {
        Self {
            acc: [0.0; FEAT_DIM],
            count: 0,
        }
    }

    #[inline]
    fn push(&mut self, features: &[f32; FEAT_DIM], reward: f32) {
        // unrolled update keeps this tiny hot path branch-light and bounds-check free.
        self.acc[0] += reward * features[0];
        self.acc[1] += reward * features[1];
        self.acc[2] += reward * features[2];
        self.acc[3] += reward * features[3];
        self.acc[4] += reward * features[4];
        self.acc[5] += reward * features[5];
        self.acc[6] += reward * features[6];
        self.acc[7] += reward * features[7];
        self.count += 1;
    }

    #[inline]
    fn apply(&mut self, weights: &mut [f32; FEAT_DIM], lr: f32) {
        if self.count == 0 {
            return;
        }
        let scale = lr / self.count as f32;
        weights[0] += self.acc[0] * scale;
        weights[1] += self.acc[1] * scale;
        weights[2] += self.acc[2] * scale;
        weights[3] += self.acc[3] * scale;
        weights[4] += self.acc[4] * scale;
        weights[5] += self.acc[5] * scale;
        weights[6] += self.acc[6] * scale;
        weights[7] += self.acc[7] * scale;
        self.acc = [0.0; FEAT_DIM];
        self.count = 0;
    }
}

#[inline]
fn infer_action_noalloc(
    weights: &[f32; FEAT_DIM],
    f: &[f32; FEAT_DIM],
    white_to_move: bool,
    rng: &mut XorShift64,
) -> u8 {
    // explicit vector-like action lanes (4 macro-actions), no heap allocations.
    let tactical = if white_to_move {
        TACTICAL_LANES
    } else {
        [0.10f32, 0.06, 0.02, 0.00]
    };

    let mut base = 0.0f32;
    for i in 0..FEAT_DIM {
        base += weights[i] * f[i];
    }

    let mut best_idx = 0u8;
    let mut best = f32::MIN;
    // unrolled 4-lane scoring
    for lane in 0..4 {
        let noise = (rng.next_f32() - 0.5) * 0.01;
        let s = base + tactical[lane] + noise;
        if s > best {
            best = s;
            best_idx = lane as u8;
        }
    }
    best_idx
}

#[inline(always)]
fn dot8(a: &[f32; FEAT_DIM], b: &[f32; FEAT_DIM]) -> f32 {
    a[0] * b[0]
        + a[1] * b[1]
        + a[2] * b[2]
        + a[3] * b[3]
        + a[4] * b[4]
        + a[5] * b[5]
        + a[6] * b[6]
        + a[7] * b[7]
}

#[derive(Clone)]
struct EnvSoA {
    white_pawns: Vec<f32>,
    black_pawns: Vec<f32>,
    white_adv: Vec<f32>,
    black_adv: Vec<f32>,
}

impl EnvSoA {
    fn new(envs: usize) -> Self {
        Self {
            white_pawns: vec![8.0; envs],
            black_pawns: vec![8.0; envs],
            white_adv: vec![0.0; envs],
            black_adv: vec![0.0; envs],
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.white_pawns.fill(8.0);
        self.black_pawns.fill(8.0);
        self.white_adv.fill(0.0);
        self.black_adv.fill(0.0);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BenchResult {
    pub elapsed: Duration,
    pub episodes: usize,
    pub total_steps: usize,
    pub steps_per_sec: f64,
    pub win_rate: f32,
}

#[derive(Clone, Copy, Debug)]
struct Move {
    from: u8,
    to: u8,
}

#[derive(Clone, Copy, Debug)]
struct Board {
    white_king: u8,
    black_king: u8,
    white_pawns: u64,
    black_pawns: u64,
    white_to_move: bool,
    done: bool,
    white_pawn_count: i32,
    black_pawn_count: i32,
    white_advance_sum: f32,
    black_advance_sum: f32,
}

#[derive(Clone, Copy, Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as u32 as f32) / (u32::MAX as f32)
    }
}

impl Board {
    #[inline]
    fn reset() -> Self {
        Self {
            white_king: 4,
            black_king: 60,
            white_pawns: 0x0000_0000_0000_ff00,
            black_pawns: 0x00ff_0000_0000_0000,
            white_to_move: true,
            done: false,
            white_pawn_count: 8,
            black_pawn_count: 8,
            white_advance_sum: 8.0,
            black_advance_sum: 8.0,
        }
    }

    #[inline]
    fn occ(&self) -> u64 {
        self.white_pawns | self.black_pawns | (1u64 << self.white_king) | (1u64 << self.black_king)
    }

    #[inline]
    fn features(&self) -> [f32; 8] {
        let wp = self.white_pawn_count;
        let bp = self.black_pawn_count;
        let white_advance = self.white_advance_sum;
        let black_advance = self.black_advance_sum;
        [
            1.0,
            wp as f32,
            bp as f32,
            (wp - bp) as f32,
            white_advance,
            black_advance,
            (self.white_king as i32 / 8) as f32,
            (self.black_king as i32 / 8) as f32,
        ]
    }

    #[inline]
    fn legal_moves(&self, out: &mut [Move; 128]) -> usize {
        if self.done {
            return 0;
        }
        let mut n = 0usize;
        let occ = self.occ();
        if self.white_to_move {
            let mut bb = self.white_pawns;
            while bb != 0 {
                let sq = bb.trailing_zeros() as i32;
                bb &= bb - 1;
                let fwd = sq + 8;
                if fwd < 64 && (occ & (1u64 << fwd)) == 0 {
                    out[n] = Move {
                        from: sq as u8,
                        to: fwd as u8,
                    };
                    n += 1;
                }
                let cap_l = if sq % 8 != 0 { sq + 7 } else { -1 };
                let cap_r = if sq % 8 != 7 { sq + 9 } else { -1 };
                if cap_l >= 0 && cap_l < 64 {
                    let b = 1u64 << cap_l;
                    if (self.black_pawns & b) != 0 || self.black_king == cap_l as u8 {
                        out[n] = Move {
                            from: sq as u8,
                            to: cap_l as u8,
                        };
                        n += 1;
                    }
                }
                if cap_r >= 0 && cap_r < 64 {
                    let b = 1u64 << cap_r;
                    if (self.black_pawns & b) != 0 || self.black_king == cap_r as u8 {
                        out[n] = Move {
                            from: sq as u8,
                            to: cap_r as u8,
                        };
                        n += 1;
                    }
                }
            }
            n += king_moves(self.white_king, self.black_king, occ, true, &mut out[n..]);
        } else {
            let mut bb = self.black_pawns;
            while bb != 0 {
                let sq = bb.trailing_zeros() as i32;
                bb &= bb - 1;
                let fwd = sq - 8;
                if fwd >= 0 && (occ & (1u64 << fwd)) == 0 {
                    out[n] = Move {
                        from: sq as u8,
                        to: fwd as u8,
                    };
                    n += 1;
                }
                let cap_l = if sq % 8 != 0 { sq - 9 } else { -1 };
                let cap_r = if sq % 8 != 7 { sq - 7 } else { -1 };
                if cap_l >= 0 {
                    let b = 1u64 << cap_l;
                    if (self.white_pawns & b) != 0 || self.white_king == cap_l as u8 {
                        out[n] = Move {
                            from: sq as u8,
                            to: cap_l as u8,
                        };
                        n += 1;
                    }
                }
                if cap_r >= 0 && cap_r < 64 {
                    let b = 1u64 << cap_r;
                    if (self.white_pawns & b) != 0 || self.white_king == cap_r as u8 {
                        out[n] = Move {
                            from: sq as u8,
                            to: cap_r as u8,
                        };
                        n += 1;
                    }
                }
            }
            n += king_moves(self.black_king, self.white_king, occ, false, &mut out[n..]);
        }
        n
    }

    #[inline]
    fn apply_move(&mut self, mv: Move) -> f32 {
        let from_b = 1u64 << mv.from;
        let to_b = 1u64 << mv.to;
        let from_rank = (mv.from >> 3) as f32;
        let to_rank = (mv.to >> 3) as f32;
        let mut reward = 0.0;
        if self.white_to_move {
            if self.white_king == mv.from {
                self.white_king = mv.to;
            } else {
                self.white_pawns &= !from_b;
                self.white_pawns |= to_b;
                self.white_advance_sum += to_rank - from_rank;
            }
            if (self.black_pawns & to_b) != 0 {
                self.black_pawns &= !to_b;
                self.black_pawn_count -= 1;
                self.black_advance_sum -= 7.0 - to_rank;
                reward += 0.25;
            }
            if self.black_king == mv.to {
                self.done = true;
                reward += 1.0;
            }
            if mv.to >= 56 {
                reward += 0.4;
            }
        } else {
            if self.black_king == mv.from {
                self.black_king = mv.to;
            } else {
                self.black_pawns &= !from_b;
                self.black_pawns |= to_b;
                self.black_advance_sum += from_rank - to_rank;
            }
            if (self.white_pawns & to_b) != 0 {
                self.white_pawns &= !to_b;
                self.white_pawn_count -= 1;
                self.white_advance_sum -= to_rank;
                reward -= 0.25;
            }
            if self.white_king == mv.to {
                self.done = true;
                reward -= 1.0;
            }
            if mv.to < 8 {
                reward -= 0.4;
            }
        }
        self.white_to_move = !self.white_to_move;
        reward
    }
}

#[inline]
fn king_moves(own_king: u8, enemy_king: u8, occ: u64, white: bool, out: &mut [Move]) -> usize {
    let _ = white;
    let r = (own_king / 8) as i32;
    let c = (own_king % 8) as i32;
    let mut n = 0;
    for dr in -1..=1 {
        for dc in -1..=1 {
            if dr == 0 && dc == 0 {
                continue;
            }
            let nr = r + dr;
            let nc = c + dc;
            if !(0..8).contains(&nr) || !(0..8).contains(&nc) {
                continue;
            }
            let to = (nr * 8 + nc) as u8;
            let b = 1u64 << to;
            if to != enemy_king && (occ & b) != 0 {
                continue;
            }
            out[n] = Move { from: own_king, to };
            n += 1;
        }
    }
    n
}

pub fn train_chess_policy(episodes: usize, max_steps: usize, seed: u64) -> BenchResult {
    train_chess_policy_batched(episodes, max_steps, 1, seed)
}

pub fn train_chess_policy_batched(
    episodes: usize,
    max_steps: usize,
    batch_size: usize,
    seed: u64,
) -> BenchResult {
    train_chess_policy_batched_from(
        episodes,
        max_steps,
        batch_size,
        seed,
        [0.03f32, 0.0, 0.0, 0.2, 0.01, -0.01, 0.0, 0.0],
    )
    .0
}

pub fn train_chess_policy_batched_from(
    episodes: usize,
    max_steps: usize,
    batch_size: usize,
    seed: u64,
    mut weights: [f32; FEAT_DIM],
) -> (BenchResult, [f32; FEAT_DIM]) {
    let mut rng = XorShift64::new(seed);
    let mut total_steps = 0usize;
    let mut wins = 0usize;
    let start = Instant::now();
    let mut mv_buf = [Move { from: 0, to: 0 }; 128];
    let mut grads = GradBuffer::new();
    let batch = batch_size.max(1);

    for _ in 0..episodes {
        let mut b = Board::reset();
        for _ in 0..max_steps {
            if b.done {
                break;
            }
            let n = b.legal_moves(&mut mv_buf);
            if n == 0 {
                b.done = true;
                break;
            }
            let f = b.features();
            let base_score = dot8(&f, &weights);
            let mut best_i = 0usize;
            let mut best_score = f32::MIN;
            let tactical_sign = if b.white_to_move { 1.0f32 } else { -1.0f32 };
            for i in 0..n {
                let mv = mv_buf[i];
                let score = base_score
                    + tactical_sign * (mv.to >> 3) as f32 * 0.02
                    + (rng.next_f32() - 0.5) * 0.01;
                if score > best_score {
                    best_score = score;
                    best_i = i;
                }
            }

            let reward = b.apply_move(mv_buf[best_i]);
            grads.push(&f, reward);
            total_steps += 1;

            if grads.count >= batch {
                grads.apply(&mut weights, 0.002f32);
            }
        }
        if b.black_king != 60 {
            wins += 1;
        } else if b.black_pawn_count == 0 {
            wins += 1;
        }
    }

    // flush partial batch
    grads.apply(&mut weights, 0.002f32);

    let elapsed = start.elapsed();
    let steps_per_sec = total_steps as f64 / elapsed.as_secs_f64().max(1e-9);
    (
        BenchResult {
            elapsed,
            episodes,
            total_steps,
            steps_per_sec,
            win_rate: wins as f32 / episodes.max(1) as f32,
        },
        weights,
    )
}

pub fn eval_policy_vs_random(
    weights: [f32; FEAT_DIM],
    games: usize,
    max_steps: usize,
    seed: u64,
) -> f32 {
    let mut rng = XorShift64::new(seed);
    let mut mv_buf = [Move { from: 0, to: 0 }; 128];
    let mut wins = 0usize;

    for _ in 0..games {
        let mut b = Board::reset();
        for _ in 0..max_steps {
            if b.done {
                break;
            }
            let n = b.legal_moves(&mut mv_buf);
            if n == 0 {
                b.done = true;
                break;
            }

            let chosen = if b.white_to_move {
                let f = b.features();
                let base_score = dot8(&f, &weights);
                let mut best_i = 0usize;
                let mut best_score = f32::MIN;
                for i in 0..n {
                    let mv = mv_buf[i];
                    let tactical = (mv.to >> 3) as f32 * 0.02;
                    let score = base_score + tactical;
                    if score > best_score {
                        best_score = score;
                        best_i = i;
                    }
                }
                mv_buf[best_i]
            } else {
                mv_buf[(rng.next_u64() as usize) % n]
            };
            b.apply_move(chosen);
        }
        if b.black_king != 60 || b.black_pawn_count == 0 {
            wins += 1;
        }
    }

    wins as f32 / games.max(1) as f32
}

pub fn train_chess_policy_soa(
    episodes: usize,
    envs: usize,
    max_steps: usize,
    batch_size: usize,
    seed: u64,
) -> BenchResult {
    let envs = envs.max(1).min(4096);
    let mut rng = XorShift64::new(seed);
    let mut weights = [0.03f32, 0.0, 0.0, 0.2, 0.01, -0.01, 0.0, 0.0];
    let mut grads = GradBuffer::new();
    let mut soa = EnvSoA::new(envs);
    let mut total_steps = 0usize;
    let start = Instant::now();

    for _ in 0..episodes {
        soa.reset();
        for _ in 0..max_steps {
            for e in 0..envs {
                let f = [
                    1.0,
                    soa.white_pawns[e],
                    soa.black_pawns[e],
                    soa.white_pawns[e] - soa.black_pawns[e],
                    soa.white_adv[e],
                    soa.black_adv[e],
                    soa.white_adv[e] * 0.1,
                    soa.black_adv[e] * 0.1,
                ];
                let action = infer_action_noalloc(&weights, &f, true, &mut rng);
                let mut reward = 0.0f32;
                match action {
                    0 => {
                        soa.white_adv[e] += 1.0;
                        reward += 0.01;
                    }
                    1 => {
                        soa.white_adv[e] += 2.0;
                        reward += 0.02;
                    }
                    2 | 3 => {
                        if soa.black_pawns[e] > 0.0 {
                            soa.black_pawns[e] -= 1.0;
                            reward += 0.20;
                        }
                    }
                    _ => {}
                }
                if rng.next_u64() & 7 == 0 && soa.white_pawns[e] > 0.0 {
                    soa.white_pawns[e] -= 1.0;
                    reward -= 0.1;
                }
                grads.push(&f, reward);
                total_steps += 1;
            }

            if grads.count >= batch_size.max(1) {
                grads.apply(&mut weights, 0.0015);
            }
        }
    }

    grads.apply(&mut weights, 0.0015);
    let elapsed = start.elapsed();
    BenchResult {
        elapsed,
        episodes,
        total_steps,
        steps_per_sec: total_steps as f64 / elapsed.as_secs_f64().max(1e-9),
        win_rate: 0.0,
    }
}

pub fn train_chess_policy_gpu(
    episodes: usize,
    envs: usize,
    max_steps: usize,
    batch_size: usize,
    seed: u64,
) -> Result<BenchResult, String> {
    let envs = envs.max(1).min(4096);
    let mut rng = XorShift64::new(seed);
    let mut weights = [0.03f32, 0.0, 0.0, 0.2, 0.01, -0.01, 0.0, 0.0];
    // 4-action projection weights [8,4]
    let proj = [
        0.0f32, 0.02, 0.06, 0.10, 1.0, 1.0, 1.2, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];

    let backend = GpuBackend::auto_select();
    let mm = GpuMemoryManager::new(backend);

    let mut grads = GradBuffer::new();
    let mut soa = EnvSoA::new(envs);
    let mut total_steps = 0usize;
    let start = Instant::now();
    let mut feats_flat = vec![0.0f32; envs * FEAT_DIM];
    let mut feats_rows = vec![[0.0f32; FEAT_DIM]; envs];
    let mut scores = vec![0.0f32; envs * 4];
    let a = mm.allocate(vec![envs, FEAT_DIM], 0.0)?;
    let b = mm.allocate_from_data(vec![FEAT_DIM, 4], proj.to_vec())?;
    let out = mm.allocate(vec![envs, 4], 0.0)?;

    for _ in 0..episodes {
        soa.reset();
        for _ in 0..max_steps {
            for e in 0..envs {
                let f = [
                    1.0,
                    soa.white_pawns[e],
                    soa.black_pawns[e],
                    soa.white_pawns[e] - soa.black_pawns[e],
                    soa.white_adv[e],
                    soa.black_adv[e],
                    soa.white_adv[e] * 0.1,
                    soa.black_adv[e] * 0.1,
                ];
                feats_rows[e] = f;
                for k in 0..FEAT_DIM {
                    feats_flat[e * FEAT_DIM + k] = f[k];
                }
            }

            mm.write(&a, &feats_flat)?;
            mm.matmul(&a, &b, &out)?;
            mm.download_into(&out, &mut scores)?;

            for e in 0..envs {
                let base = scores[e];
                let mut action = 0usize;
                let mut best = f32::MIN;
                for lane in 0..4 {
                    let tactical = TACTICAL_LANES[lane];
                    let s = base + tactical + (rng.next_f32() - 0.5) * 0.01;
                    if s > best {
                        best = s;
                        action = lane;
                    }
                }
                let mut reward = 0.0f32;
                match action {
                    0 => {
                        soa.white_adv[e] += 1.0;
                        reward += 0.01;
                    }
                    1 => {
                        soa.white_adv[e] += 2.0;
                        reward += 0.02;
                    }
                    2 | 3 => {
                        if soa.black_pawns[e] > 0.0 {
                            soa.black_pawns[e] -= 1.0;
                            reward += 0.20;
                        }
                    }
                    _ => {}
                }
                if rng.next_u64() & 7 == 0 && soa.white_pawns[e] > 0.0 {
                    soa.white_pawns[e] -= 1.0;
                    reward -= 0.1;
                }
                grads.push(&feats_rows[e], reward);
                total_steps += 1;
            }

            if grads.count >= batch_size.max(1) {
                grads.apply(&mut weights, 0.0015);
            }
        }
    }

    grads.apply(&mut weights, 0.0015);
    mm.free(&a);
    mm.free(&b);
    mm.free(&out);
    let elapsed = start.elapsed();
    Ok(BenchResult {
        elapsed,
        episodes,
        total_steps,
        steps_per_sec: total_steps as f64 / elapsed.as_secs_f64().max(1e-9),
        win_rate: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_position_has_moves() {
        let b = Board::reset();
        let mut mv = [Move { from: 0, to: 0 }; 128];
        assert!(b.legal_moves(&mut mv) > 0);
    }

    #[test]
    fn training_runs_and_counts_steps() {
        let r = train_chess_policy(200, 16, 7);
        assert!(r.total_steps > 0);
        assert!(r.steps_per_sec > 0.0);
    }

    #[test]
    fn batched_training_runs() {
        let r = train_chess_policy_batched(200, 16, 32, 7);
        assert!(r.total_steps > 0);
        assert!(r.steps_per_sec > 0.0);
    }

    #[test]
    fn soa_training_runs() {
        let r = train_chess_policy_soa(50, 8, 16, 32, 7);
        assert!(r.total_steps > 0);
        assert!(r.steps_per_sec > 0.0);
    }

    #[test]
    fn gpu_training_runs() {
        let r = train_chess_policy_gpu(5, 4, 4, 4, 7).unwrap();
        assert!(r.total_steps > 0);
    }
}
