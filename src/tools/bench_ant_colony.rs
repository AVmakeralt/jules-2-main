use std::sync::mpsc;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Colony {
    ants: f32,
    food: f32,
    pheromone: f32,
    brood: f32,
    energy: f32,
    checksum: f32,
}

impl Colony {
    fn new() -> Self {
        Self {
            ants: 200.0,
            food: 1200.0,
            pheromone: 1.0,
            brood: 40.0,
            energy: 600.0,
            checksum: 0.0,
        }
    }
}

#[inline(always)]
fn ant_step(mut c: Colony, seasonal: f32) -> Colony {
    let forage_eff = 0.004 * c.ants * (1.0 + 0.3 * c.pheromone);
    let gathered = forage_eff.min(c.food);
    c.food -= gathered;
    c.energy += gathered * 0.9;

    let burn = c.ants * 0.0016;
    c.energy -= burn;

    c.pheromone = (c.pheromone * 0.992 + gathered * 0.0009).max(0.01);

    if c.energy > 250.0 {
        c.brood += 0.06 * (1.0 + 0.1 * c.pheromone);
        c.energy -= 0.02;
    }

    let hatch = c.brood * 0.012;
    c.brood -= hatch;
    c.ants += hatch;

    let starvation = if c.energy < 150.0 {
        (150.0 - c.energy) * 0.0005
    } else {
        0.0
    };
    let deaths = c.ants * (0.0008 + starvation);
    c.ants = (c.ants - deaths).max(0.0);

    c.food += seasonal;
    c.checksum += c.ants * 0.00001 + c.pheromone * 0.0001 + c.energy * 0.000001;
    c
}

fn run_sequential(days: usize, micro_steps: usize) -> (f64, Colony) {
    let mut colony = Colony::new();
    let t0 = Instant::now();
    for day in 0..days {
        let seasonal = compute_seasonal(day);
        for _ in 0..micro_steps {
            colony = ant_step(colony, seasonal);
        }
    }
    (t0.elapsed().as_secs_f64(), colony)
}

fn run_pipelined(days: usize, micro_steps: usize, lookahead_days: usize) -> (f64, Colony) {
    let mut colony = Colony::new();
    let (tx, rx) = mpsc::sync_channel::<f32>(lookahead_days.max(1));
    let t0 = Instant::now();

    let producer = std::thread::spawn(move || {
        for day in 0..days {
            let seasonal = compute_seasonal(day);
            if tx.send(seasonal).is_err() {
                return;
            }
        }
    });

    for _day in 0..days {
        let seasonal = match rx.recv() {
            Ok(v) => v,
            Err(_) => break,
        };
        for _ in 0..micro_steps {
            colony = ant_step(colony, seasonal);
        }
    }

    let _ = producer.join();
    (t0.elapsed().as_secs_f64(), colony)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let days: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3650);
    let micro_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let lookahead: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);

    println!("bench-ant-colony days={days} micro_steps={micro_steps} lookahead={lookahead}");

    let (seq_t, seq) = run_sequential(days, micro_steps);
    let (pipe_t, pipe) = run_pipelined(days, micro_steps, lookahead);

    println!(
        "Sequential: {:.3}s total, checksum={:.6}, ants={:.3}, food={:.3}",
        seq_t, seq.checksum, seq.ants, seq.food
    );
    println!(
        "Pipelined : {:.3}s total, checksum={:.6}, ants={:.3}, food={:.3}",
        pipe_t, pipe.checksum, pipe.ants, pipe.food
    );
    println!("Pipeline speedup: {:.2}x", seq_t / pipe_t.max(1e-12));
}

#[inline]
fn compute_seasonal(day: usize) -> f32 {
    let mut x = day as f32 * 0.017;
    let mut acc = 0.0f32;
    // intentionally non-trivial to model upstream environment preprocessing
    for _ in 0..64 {
        x = (x * 1.00013 + 0.031).sin();
        acc += x * x;
    }
    let base = if (day % 90) < 45 { 0.8 } else { 0.3 };
    base + (acc * 0.0002)
}
