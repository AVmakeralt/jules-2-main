// Visual scene editor (minimal CLI-based implementation).
// Save/load scenes as JSON using a small serializable representation.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Serialize, Deserialize};

use crate::interp::{ComponentMap, EcsWorld, EntityId, Value, WorldSnapshot};

#[derive(Debug, Serialize, Deserialize)]
pub enum SerializableValue {
    F32(f32),
    Vec2([f32;2]),
    Vec3([f32;3]),
    Vec4([f32;4]),
    Bool(bool),
    Str(String),
    Unit,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EntityData {
    pub id: EntityId,
    pub comps: HashMap<String, SerializableValue>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SceneFile {
    pub entities: Vec<EntityData>,
}

fn value_to_serializable(v: &Value) -> Option<SerializableValue> {
    match v {
        Value::F32(x) => Some(SerializableValue::F32(*x)),
        Value::Vec2(a) => Some(SerializableValue::Vec2(*a)),
        Value::Vec3(a) => Some(SerializableValue::Vec3(*a)),
        Value::Vec4(a) => Some(SerializableValue::Vec4(*a)),
        Value::Bool(b) => Some(SerializableValue::Bool(*b)),
        Value::Str(s) => Some(SerializableValue::Str(s.clone())),
        Value::Unit => Some(SerializableValue::Unit),
        _ => None,
    }
}

fn serializable_to_value(s: &SerializableValue) -> Value {
    match s {
        SerializableValue::F32(x) => Value::F32(*x),
        SerializableValue::Vec2(a) => Value::Vec2(*a),
        SerializableValue::Vec3(a) => Value::Vec3(*a),
        SerializableValue::Vec4(a) => Value::Vec4(*a),
        SerializableValue::Bool(b) => Value::Bool(*b),
        SerializableValue::Str(st) => Value::Str(st.clone()),
        SerializableValue::Unit => Value::Unit,
    }
}

/// Save the current world state to a JSON scene file. Only a limited set
/// of `Value` variants are supported for now (floats, vectors, bool, str).
pub fn save_scene<P: AsRef<Path>>(path: P, world: &EcsWorld) -> Result<(), String> {
    let snap = world.snapshot();
    let mut entities: Vec<EntityData> = Vec::new();
    for (id, comps) in snap.entities {
        let mut map: HashMap<String, SerializableValue> = HashMap::new();
        for (k, v) in comps {
            if let Some(sv) = value_to_serializable(&v) {
                map.insert(k, sv);
            }
        }
        entities.push(EntityData { id, comps: map });
    }
    let scene = SceneFile { entities };
    let txt = serde_json::to_string_pretty(&scene).map_err(|e| e.to_string())?;
    fs::write(path, txt).map_err(|e| e.to_string())
}

/// Load a scene file and restore it into the provided world (replacing state).
pub fn load_scene<P: AsRef<Path>>(path: P, world: &mut EcsWorld) -> Result<(), String> {
    let txt = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let scene: SceneFile = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
    let mut snap = WorldSnapshot { entities: Vec::new() };
    for ent in scene.entities {
        let mut comps = ComponentMap::default();
        for (k, sv) in ent.comps {
            comps.insert(k, serializable_to_value(&sv));
        }
        snap.entities.push((ent.id, comps));
    }
    world.restore_snapshot(&snap);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::EcsWorld;
    use crate::interp::Value;
    use std::fs;

    #[test]
    fn save_and_load_roundtrip() {
        let mut world = EcsWorld::default();
        let id = world.spawn();
        world.insert_component(id, "pos", Value::Vec3([1.0, 2.0, 3.0]));
        let tmp = "test_scene.json";
        save_scene(tmp, &world).unwrap();
        let mut world2 = EcsWorld::default();
        load_scene(tmp, &mut world2).unwrap();
        let p = world2.get_component(id, "pos").unwrap();
        match p {
            Value::Vec3(a) => assert_eq!(a[0], 1.0),
            _ => panic!("bad value"),
        }
        let _ = fs::remove_file(tmp);
    }
}
