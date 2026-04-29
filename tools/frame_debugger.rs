// Frame-by-frame debugger for Jules.
// Minimal, non-invasive API intended for future expansion into a full
// frame debugger with history, inspection, and rewind.

use crate::interp::{ComponentMap, EcsWorld, EntityId, WorldSnapshot};

/// Simple frame debugger storing a bounded history of world snapshots.
pub struct Debugger {
    history: Vec<WorldSnapshot>,
    max_history: usize,
    paused: bool,
}

impl Debugger {
    pub fn new(max_history: usize) -> Self {
        Debugger {
            history: Vec::new(),
            max_history,
            paused: false,
        }
    }

    /// Capture the current world state into the debugger history.
    pub fn capture(&mut self, world: &EcsWorld) {
        let snap = world.snapshot();
        self.history.push(snap);
        if self.history.len() > self.max_history {
            // Remove oldest entry efficiently by rotating and truncating
            // instead of O(n) remove(0) shift.
            let excess = self.history.len() - self.max_history;
            self.history.drain(0..excess);
        }
    }

    /// Step the world by invoking the provided tick function. The debugger
    /// will capture a snapshot before the tick so the user can inspect
    /// pre-step state and rewind if needed.
    pub fn step_with<F>(&mut self, world: &mut EcsWorld, mut tick: F)
    where
        F: FnMut(&mut EcsWorld),
    {
        self.capture(world);
        tick(world);
    }

    /// Inspect the most recent snapshot for a given entity's components.
    pub fn inspect_entity(&self, entity: EntityId) -> Option<ComponentMap> {
        if let Some(snap) = self.history.last() {
            for (id, comps) in &snap.entities {
                if *id == entity {
                    return Some(comps.clone());
                }
            }
        }
        None
    }

    /// Rewind the world to the given history index (0 = oldest, len-1 = newest).
    pub fn restore(&self, world: &mut EcsWorld, index: usize) -> Result<(), String> {
        if index >= self.history.len() {
            return Err("index out of range".into());
        }
        let snap = &self.history[index];
        world.restore_snapshot(snap);
        Ok(())
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }
    pub fn resume(&mut self) {
        self.paused = false;
    }
    pub fn is_paused(&self) -> bool {
        self.paused
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::{EcsWorld, Value};

    #[test]
    fn capture_and_restore() {
        let mut world = EcsWorld::default();
        let id = world.spawn();
        world.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
        let mut dbg = Debugger::new(8);
        dbg.capture(&world);
        // mutate world
        world.insert_component(id, "pos", Value::Vec3([1.0, 2.0, 3.0]));
        // restore
        dbg.restore(&mut world, 0).unwrap();
        let p = world.get_component(id, "pos").unwrap();
        match p {
            Value::Vec3(v) => assert_eq!(v[0], 0.0),
            _ => panic!("unexpected value"),
        }
    }
}
