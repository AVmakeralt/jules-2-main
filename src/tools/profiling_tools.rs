// Profiling tools: simple scoped timers with aggregate reporting.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Instant, Duration};

use lazy_static::lazy_static;

lazy_static! {
    static ref AGG: Mutex<HashMap<String, (Duration, u64)>> = Mutex::new(HashMap::new());
}

pub fn available() -> bool { true }

pub struct ScopedTimer {
    name: String,
    start: Instant,
}

impl ScopedTimer {
    pub fn new(name: impl Into<String>) -> Self {
        ScopedTimer { name: name.into(), start: Instant::now() }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        let mut agg = AGG.lock().unwrap();
        let entry = agg.entry(self.name.clone()).or_insert((Duration::ZERO, 0));
        entry.0 += elapsed;
        entry.1 += 1;
    }
}

pub fn report() -> String {
    let agg = AGG.lock().unwrap();
    let mut out = String::new();
    for (k, (dur, cnt)) in agg.iter() {
        out.push_str(&format!("{}: {:.6}s over {} calls\n", k, dur.as_secs_f64(), cnt));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_timer_records() {
        {
            let _t = ScopedTimer::new("test-scope");
        }
        let r = report();
        assert!(r.contains("test-scope"));
    }
}
