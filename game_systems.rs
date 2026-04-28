// =========================================================================
// Physics Engine Integration for Jules
// Provides Rapier physics simulation with ECS integration
// =========================================================================

use rustc_hash::FxHashMap;


// Physics component that can be attached to entities
#[derive(Debug, Clone)]
pub struct PhysicsBody {
    pub body_id: u32,
    pub mass: f32,
    pub shape: PhysicsShape,
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub position: [f32; 3],
    pub rotation: [f32; 4], // Quaternion
}

#[derive(Debug, Clone)]
pub enum PhysicsShape {
    Sphere { radius: f32 },
    Box { width: f32, height: f32, depth: f32 },
    Capsule { radius: f32, height: f32 },
    Cylinder { radius: f32, height: f32 },
    Plane { normal: [f32; 3] },
}

#[derive(Debug, Clone)]
pub struct PhysicsCollider {
    pub body_id: u32,
    pub friction: f32,
    pub restitution: f32,
    pub density: f32,
}

#[derive(Debug)]
pub struct PhysicsWorld {
    bodies: FxHashMap<u32, PhysicsBody>,
    colliders: FxHashMap<u32, PhysicsCollider>,
    next_id: u32,
    gravity: [f32; 3],
    damping: f32,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        PhysicsWorld {
            bodies: FxHashMap::default(),
            colliders: FxHashMap::default(),
            next_id: 1,
            gravity: [0.0, -9.81, 0.0],
            damping: 0.99,
        }
    }

    pub fn create_rigid_body(&mut self, mass: f32, shape: PhysicsShape, position: [f32; 3]) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let body = PhysicsBody {
            body_id: id,
            mass,
            shape,
            velocity: [0.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.0, 0.0],
            position,
            rotation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
        };

        self.bodies.insert(id, body);
        id
    }

    pub fn add_collider(&mut self, body_id: u32, friction: f32, restitution: f32) {
        let collider = PhysicsCollider {
            body_id,
            friction,
            restitution,
            density: 1.0,
        };
        self.colliders.insert(body_id, collider);
    }

    pub fn set_velocity(&mut self, body_id: u32, vx: f32, vy: f32, vz: f32) {
        if let Some(body) = self.bodies.get_mut(&body_id) {
            body.velocity = [vx, vy, vz];
        }
    }

    pub fn get_position(&self, body_id: u32) -> Option<[f32; 3]> {
        self.bodies.get(&body_id).map(|b| b.position)
    }

    pub fn get_velocity(&self, body_id: u32) -> Option<[f32; 3]> {
        self.bodies.get(&body_id).map(|b| b.velocity)
    }

    pub fn step(&mut self, dt: f32) {
        // Simple Euler integration
        for body in self.bodies.values_mut() {
            if body.mass > 0.0 {
                // Apply gravity
                body.velocity[1] += self.gravity[1] * dt;

                // Apply damping
                body.velocity[0] *= self.damping;
                body.velocity[1] *= self.damping;
                body.velocity[2] *= self.damping;

                // Update position
                body.position[0] += body.velocity[0] * dt;
                body.position[1] += body.velocity[1] * dt;
                body.position[2] += body.velocity[2] * dt;
            }
        }

        // Simple collision detection (sphere-sphere)
        self.detect_collisions();
    }

    fn detect_collisions(&mut self) {
        // Collect body references once to avoid repeated HashMap lookups.
        let bodies: Vec<_> = self.bodies.values().cloned().collect();

        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                if self.bodies_colliding(&bodies[i], &bodies[j]) {
                    let id1 = bodies[i].body_id;
                    let id2 = bodies[j].body_id;
                    self.resolve_collision(id1, id2);
                }
            }
        }
    }

    fn bodies_colliding(&self, b1: &PhysicsBody, b2: &PhysicsBody) -> bool {
        let dx = b1.position[0] - b2.position[0];
        let dy = b1.position[1] - b2.position[1];
        let dz = b1.position[2] - b2.position[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;

        let r1 = match &b1.shape {
            PhysicsShape::Sphere { radius } => *radius,
            _ => 1.0,
        };
        let r2 = match &b2.shape {
            PhysicsShape::Sphere { radius } => *radius,
            _ => 1.0,
        };

        dist_sq < (r1 + r2) * (r1 + r2)
    }

    fn resolve_collision(&mut self, id1: u32, id2: u32) {
        // Simple impulse-based collision response
        let (pos1, vel1, mass1) = {
            let b = self.bodies.get(&id1).unwrap();
            (b.position, b.velocity, b.mass)
        };
        let (pos2, vel2, mass2) = {
            let b = self.bodies.get(&id2).unwrap();
            (b.position, b.velocity, b.mass)
        };

        let dx = pos2[0] - pos1[0];
        let dy = pos2[1] - pos1[1];
        let dz = pos2[2] - pos1[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist < 0.001 {
            return;
        }

        let nx = dx / dist;
        let ny = dy / dist;
        let nz = dz / dist;

        let dvx = vel2[0] - vel1[0];
        let dvy = vel2[1] - vel1[1];
        let dvz = vel2[2] - vel1[2];

        let dvn = dvx * nx + dvy * ny + dvz * nz;

        if dvn >= 0.0 {
            return;
        } // Already separating

        let inv_mass_sum = 1.0 / (mass1 + mass2);
        let impulse = -dvn / inv_mass_sum;

        if let Some(body1) = self.bodies.get_mut(&id1) {
            if body1.mass > 0.0 {
                body1.velocity[0] -= impulse * nx / mass1;
                body1.velocity[1] -= impulse * ny / mass1;
                body1.velocity[2] -= impulse * nz / mass1;
            }
        }

        if let Some(body2) = self.bodies.get_mut(&id2) {
            if body2.mass > 0.0 {
                body2.velocity[0] += impulse * nx / mass2;
                body2.velocity[1] += impulse * ny / mass2;
                body2.velocity[2] += impulse * nz / mass2;
            }
        }
    }
}

// =========================================================================
// Graphics / Rendering System
// =========================================================================

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct Material {
    pub color: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
    pub emission: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: [0.0, 5.0, 10.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 45.0,
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl Camera {
    pub fn look_at(&mut self, pos: [f32; 3], target: [f32; 3]) {
        self.position = pos;
        self.target = target;
    }
}

#[derive(Debug, Clone)]
pub struct Sprite {
    pub name: String,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub mesh_id: u32,
}

#[derive(Debug, Clone)]
pub enum SceneObjectKind {
    Sprite(u32),
    Model(u32),
}

#[derive(Debug, Clone)]
pub struct SceneObject {
    pub id: u32,
    pub kind: SceneObjectKind,
    pub material_id: Option<u32>,
}

#[derive(Debug, Clone)]
pub enum RenderCommand {
    Clear {
        color: [f32; 4],
    },
    Rect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: [f32; 4],
        layer: i32,
    },
    Sprite {
        sprite_id: u32,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        rotation_deg: f32,
        layer: i32,
    },
}

#[derive(Debug, Clone)]
pub struct GridMap {
    pub id: u32,
    pub width: usize,
    pub height: usize,
    pub cells: Vec<u32>, // object-id per tile; 0 = empty
}

#[derive(Debug, Clone)]
pub struct ChunkedGridMap {
    pub id: u32,
    pub width: usize,
    pub height: usize,
    pub chunk_size: usize,
    pub chunks: FxHashMap<(i32, i32), Vec<u32>>,
}

pub struct RenderState {
    pub camera: Camera,
    pub meshes: FxHashMap<u32, Mesh>,
    pub materials: FxHashMap<u32, Material>,
    pub sprites: FxHashMap<u32, Sprite>,
    pub models: FxHashMap<u32, Model>,
    pub objects: FxHashMap<u32, SceneObject>,
    pub maps: FxHashMap<u32, GridMap>,
    pub chunked_maps: FxHashMap<u32, ChunkedGridMap>,
    pub next_id: u32,
    // 8K rendering support
    pub width: u32,
    pub height: u32,
    pub command_buffer: Vec<RenderCommand>,
}

impl RenderState {
    pub fn new() -> Self {
        RenderState {
            camera: Camera::default(),
            meshes: FxHashMap::default(),
            materials: FxHashMap::default(),
            sprites: FxHashMap::default(),
            models: FxHashMap::default(),
            objects: FxHashMap::default(),
            maps: FxHashMap::default(),
            chunked_maps: FxHashMap::default(),
            next_id: 1,
            width: 1920,
            height: 1080,
            command_buffer: Vec::new(),
        }
    }

    pub fn create_mesh(&mut self, vertices: Vec<[f32; 3]>, indices: Vec<u32>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let normals = vec![[0.0, 1.0, 0.0]; vertices.len()];

        let mesh = Mesh {
            vertices,
            normals,
            indices,
        };

        self.meshes.insert(id, mesh);
        id
    }

    pub fn create_material(&mut self, color: [f32; 4]) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let material = Material {
            color,
            roughness: 0.5,
            metallic: 0.0,
            emission: [0.0, 0.0, 0.0],
        };

        self.materials.insert(id, material);
        id
    }

    pub fn update_camera_position(&mut self, x: f32, y: f32, z: f32) {
        self.camera.position = [x, y, z];
    }

    pub fn begin_frame(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.command_buffer.clear();
    }

    pub fn queue_clear(&mut self, color: [f32; 4]) {
        self.command_buffer.push(RenderCommand::Clear { color });
    }

    pub fn queue_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4], layer: i32) {
        self.command_buffer.push(RenderCommand::Rect {
            x,
            y,
            w,
            h,
            color,
            layer,
        });
    }

    pub fn queue_sprite(
        &mut self,
        sprite_id: u32,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        rotation_deg: f32,
        layer: i32,
    ) -> Result<(), String> {
        if !self.sprites.contains_key(&sprite_id) {
            return Err(format!("unknown sprite_id {}", sprite_id));
        }
        self.command_buffer.push(RenderCommand::Sprite {
            sprite_id,
            x,
            y,
            w,
            h,
            rotation_deg,
            layer,
        });
        Ok(())
    }

    pub fn drain_command_buffer(&mut self) -> Vec<RenderCommand> {
        std::mem::take(&mut self.command_buffer)
    }

    pub fn create_sprite(&mut self, name: String, width: f32, height: f32) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.sprites.insert(
            id,
            Sprite {
                name,
                width,
                height,
            },
        );
        id
    }

    pub fn create_model(&mut self, name: String, mesh_id: u32) -> Result<u32, String> {
        if !self.meshes.contains_key(&mesh_id) {
            return Err(format!("unknown mesh_id {}", mesh_id));
        }
        let id = self.next_id;
        self.next_id += 1;
        self.models.insert(id, Model { name, mesh_id });
        Ok(id)
    }

    pub fn create_object(
        &mut self,
        kind: SceneObjectKind,
        material_id: Option<u32>,
    ) -> Result<u32, String> {
        if let Some(mid) = material_id {
            if !self.materials.contains_key(&mid) {
                return Err(format!("unknown material_id {}", mid));
            }
        }
        match kind {
            SceneObjectKind::Sprite(id) if !self.sprites.contains_key(&id) => {
                return Err(format!("unknown sprite_id {}", id));
            }
            SceneObjectKind::Model(id) if !self.models.contains_key(&id) => {
                return Err(format!("unknown model_id {}", id));
            }
            _ => {}
        }

        let id = self.next_id;
        self.next_id += 1;
        self.objects.insert(
            id,
            SceneObject {
                id,
                kind,
                material_id,
            },
        );
        Ok(id)
    }

    pub fn create_grid_map(&mut self, width: usize, height: usize) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.maps.insert(
            id,
            GridMap {
                id,
                width,
                height,
                cells: vec![0; width.saturating_mul(height)],
            },
        );
        id
    }

    pub fn set_grid_cell(
        &mut self,
        map_id: u32,
        x: usize,
        y: usize,
        object_id: u32,
    ) -> Result<(), String> {
        if object_id != 0 && !self.objects.contains_key(&object_id) {
            return Err(format!("unknown object_id {}", object_id));
        }
        let map = self
            .maps
            .get_mut(&map_id)
            .ok_or_else(|| format!("unknown map_id {}", map_id))?;
        if x >= map.width || y >= map.height {
            return Err(format!(
                "grid index out of bounds ({}, {}) for {}x{}",
                x, y, map.width, map.height
            ));
        }
        map.cells[y * map.width + x] = object_id;
        Ok(())
    }

    pub fn get_grid_cell(&self, map_id: u32, x: usize, y: usize) -> Option<u32> {
        let map = self.maps.get(&map_id)?;
        if x >= map.width || y >= map.height {
            return None;
        }
        Some(map.cells[y * map.width + x])
    }

    pub fn render_grid_map(&self, map_id: u32) -> Result<usize, String> {
        let map = self
            .maps
            .get(&map_id)
            .ok_or_else(|| format!("unknown map_id {}", map_id))?;
        let mut drawn = 0usize;
        for object_id in &map.cells {
            if *object_id == 0 {
                continue;
            }
            if self.objects.contains_key(object_id) {
                drawn += 1;
            }
        }
        Ok(drawn)
    }

    pub fn create_chunked_grid_map(
        &mut self,
        width: usize,
        height: usize,
        chunk_size: usize,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.chunked_maps.insert(
            id,
            ChunkedGridMap {
                id,
                width,
                height,
                chunk_size: chunk_size.max(1),
                chunks: FxHashMap::default(),
            },
        );
        id
    }

    pub fn set_chunked_grid_cell(
        &mut self,
        map_id: u32,
        x: usize,
        y: usize,
        object_id: u32,
    ) -> Result<(), String> {
        if object_id != 0 && !self.objects.contains_key(&object_id) {
            return Err(format!("unknown object_id {}", object_id));
        }
        let map = self
            .chunked_maps
            .get_mut(&map_id)
            .ok_or_else(|| format!("unknown chunked map_id {}", map_id))?;
        if x >= map.width || y >= map.height {
            return Err(format!(
                "chunked grid index out of bounds ({}, {}) for {}x{}",
                x, y, map.width, map.height
            ));
        }

        let csize = map.chunk_size;
        let cx = (x / csize) as i32;
        let cy = (y / csize) as i32;
        let lx = x % csize;
        let ly = y % csize;
        let chunk = map
            .chunks
            .entry((cx, cy))
            .or_insert_with(|| vec![0; csize.saturating_mul(csize)]);
        chunk[ly * csize + lx] = object_id;
        Ok(())
    }

    pub fn render_chunked_grid_map(&self, map_id: u32) -> Result<usize, String> {
        let map = self
            .chunked_maps
            .get(&map_id)
            .ok_or_else(|| format!("unknown chunked map_id {}", map_id))?;
        let mut drawn = 0usize;
        for chunk in map.chunks.values() {
            drawn += chunk
                .iter()
                .filter(|&&id| id != 0 && self.objects.contains_key(&id))
                .count();
        }
        Ok(drawn)
    }
}

// Primitive mesh generators
pub fn create_cube_mesh(size: f32) -> Mesh {
    let s = size / 2.0;
    let vertices = vec![
        // Front
        [-s, -s, s],
        [s, -s, s],
        [s, s, s],
        [-s, s, s],
        // Back
        [-s, -s, -s],
        [s, -s, -s],
        [s, s, -s],
        [-s, s, -s],
        // Right
        [s, -s, s],
        [s, -s, -s],
        [s, s, -s],
        [s, s, s],
        // Left
        [-s, -s, s],
        [-s, -s, -s],
        [-s, s, -s],
        [-s, s, s],
        // Top
        [-s, s, s],
        [s, s, s],
        [s, s, -s],
        [-s, s, -s],
        // Bottom
        [-s, -s, s],
        [s, -s, s],
        [s, -s, -s],
        [-s, -s, -s],
    ];

    let indices = vec![
        // Front
        0, 1, 2, 2, 3, 0, // Back
        6, 5, 4, 4, 7, 6, // Right
        8, 9, 10, 10, 11, 8, // Left
        14, 13, 12, 12, 15, 14, // Top
        16, 17, 18, 18, 19, 16, // Bottom
        22, 21, 20, 20, 23, 22,
    ];

    Mesh {
        vertices,
        normals: vec![[0.0, 1.0, 0.0]; 24],
        indices,
    }
}

pub fn create_sphere_mesh(radius: f32, segments: u32) -> Mesh {
    // Clamp segments to prevent unbounded memory allocation (max 65K vertices ≈ 256 segments)
    let segments = segments.min(256);
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    let rings = segments;
    let sectors = segments;

    for i in 0..=rings {
        let lat = (std::f32::consts::PI / rings as f32) * i as f32;
        let sin_lat = lat.sin();
        let cos_lat = lat.cos();

        for j in 0..=sectors {
            let lon = (2.0 * std::f32::consts::PI / sectors as f32) * j as f32;
            let sin_lon = lon.sin();
            let cos_lon = lon.cos();

            let x = radius * sin_lat * cos_lon;
            let y = radius * cos_lat;
            let z = radius * sin_lat * sin_lon;

            vertices.push([x, y, z]);

            // Normal for sphere is simply the normalized position (points outward)
            let normal = [(sin_lat * cos_lon), (cos_lat), (sin_lat * sin_lon)];
            normals.push(normal);
        }
    }

    for i in 0..rings {
        for j in 0..sectors {
            let a = i * (sectors + 1) + j;
            let b = a + sectors + 1;

            indices.push(a as u32);
            indices.push(b as u32);
            indices.push((a + 1) as u32);

            indices.push((a + 1) as u32);
            indices.push(b as u32);
            indices.push((b + 1) as u32);
        }
    }

    Mesh {
        vertices,
        normals,
        indices,
    }
}

// =========================================================================
// Input System
// =========================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KeyCode {
    W,
    A,
    S,
    D,
    Left,
    Right,
    Up,
    Down,
    Space,
    Enter,
    Escape,
}

#[derive(Debug, Clone)]
pub struct InputState {
    pub keys_pressed: std::collections::HashSet<KeyCode>,
    pub mouse_x: f32,
    pub mouse_y: f32,
    pub mouse_scroll: f32,
    pub gamepad_axes: [f32; 6], // [LX, LY, RX, RY, LT, RT]
    pub gamepad_buttons: std::collections::HashSet<u8>,
}

impl InputState {
    pub fn new() -> Self {
        InputState {
            keys_pressed: std::collections::HashSet::new(),
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_scroll: 0.0,
            gamepad_axes: [0.0; 6],
            gamepad_buttons: std::collections::HashSet::new(),
        }
    }

    pub fn set_key_pressed(&mut self, key: KeyCode, pressed: bool) {
        if pressed {
            self.keys_pressed.insert(key);
        } else {
            self.keys_pressed.remove(&key);
        }
    }

    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    pub fn set_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse_x = x;
        self.mouse_y = y;
    }

    pub fn get_axis(&self, axis: &str) -> f32 {
        match axis {
            "horizontal" => {
                let mut val = 0.0;
                if self.is_key_pressed(KeyCode::D) {
                    val += 1.0;
                }
                if self.is_key_pressed(KeyCode::A) {
                    val -= 1.0;
                }
                if self.gamepad_axes[0].abs() > 0.1 {
                    val = self.gamepad_axes[0];
                }
                val.clamp(-1.0, 1.0)
            }
            "vertical" => {
                let mut val = 0.0;
                if self.is_key_pressed(KeyCode::W) {
                    val += 1.0;
                }
                if self.is_key_pressed(KeyCode::S) {
                    val -= 1.0;
                }
                if self.gamepad_axes[1].abs() > 0.1 {
                    val = self.gamepad_axes[1];
                }
                val.clamp(-1.0, 1.0)
            }
            _ => 0.0,
        }
    }
}

// =========================================================================
// Integration: These would be exposed as Jules built-in functions
// =========================================================================

pub fn physics_world_new() -> PhysicsWorld {
    PhysicsWorld::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_map_with_sprite_and_model_objects() {
        let mut render = RenderState::new();
        let mesh = render.create_mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
        );
        let mat = render.create_material([1.0, 1.0, 1.0, 1.0]);
        let sprite = render.create_sprite("grass".into(), 1.0, 1.0);
        let model = render.create_model("tree".into(), mesh).unwrap();
        let sprite_obj = render
            .create_object(SceneObjectKind::Sprite(sprite), Some(mat))
            .unwrap();
        let model_obj = render
            .create_object(SceneObjectKind::Model(model), Some(mat))
            .unwrap();

        let map = render.create_grid_map(4, 4);
        render.set_grid_cell(map, 0, 0, sprite_obj).unwrap();
        render.set_grid_cell(map, 1, 0, model_obj).unwrap();

        assert_eq!(render.get_grid_cell(map, 0, 0), Some(sprite_obj));
        assert_eq!(render.get_grid_cell(map, 1, 0), Some(model_obj));
        assert_eq!(render.render_grid_map(map).unwrap(), 2);
    }

    #[test]
    fn test_chunked_grid_map_for_sparse_large_worlds() {
        let mut render = RenderState::new();
        let mat = render.create_material([1.0, 1.0, 1.0, 1.0]);
        let sprite = render.create_sprite("rock".into(), 1.0, 1.0);
        let sprite_obj = render
            .create_object(SceneObjectKind::Sprite(sprite), Some(mat))
            .unwrap();

        let map = render.create_chunked_grid_map(100_000, 100_000, 64);
        render.set_chunked_grid_cell(map, 0, 0, sprite_obj).unwrap();
        render
            .set_chunked_grid_cell(map, 99_999, 99_999, sprite_obj)
            .unwrap();

        assert_eq!(render.render_chunked_grid_map(map).unwrap(), 2);
        assert_eq!(
            render
                .chunked_maps
                .get(&map)
                .expect("chunked map exists")
                .chunks
                .len(),
            2
        );
    }
}

pub fn render_state_new() -> RenderState {
    RenderState::new()
}

pub fn input_state_new() -> InputState {
    InputState::new()
}
