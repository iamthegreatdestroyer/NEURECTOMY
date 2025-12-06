/**
 * Force Calculation Compute Shader
 * 
 * WGSL compute shader for parallel force-directed graph layout.
 * Implements repulsion (charge), attraction (links), center gravity,
 * and collision forces on the GPU.
 * 
 * @module @neurectomy/3d-engine/webgpu/compute
 * @agents @CORE @VELOCITY @AXIOM
 */

// ============================================================================
// Structures
// ============================================================================

struct Node {
    position: vec3f,
    velocity: vec3f,
    force: vec3f,
    mass: f32,
    radius: f32,
    charge: f32,
    pinned: u32,  // 0 = free, 1 = pinned
    _padding: f32,
}

struct Edge {
    source_idx: u32,
    target_idx: u32,
    weight: f32,
    _padding: f32,
}

struct SimParams {
    node_count: u32,
    edge_count: u32,
    alpha: f32,
    alpha_decay: f32,
    velocity_decay: f32,
    charge_strength: f32,
    charge_distance_min: f32,
    charge_distance_max: f32,
    center_strength: f32,
    link_strength: f32,
    link_distance: f32,
    collision_radius_mult: f32,
    theta: f32,  // Barnes-Hut threshold
    is_3d: u32,
    _padding1: f32,
    _padding2: f32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> edges: array<Edge>;
@group(0) @binding(2) var<uniform> params: SimParams;

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const EPSILON: f32 = 0.0001;

// ============================================================================
// Helper Functions
// ============================================================================

fn safe_normalize(v: vec3f) -> vec3f {
    let len = length(v);
    if (len < EPSILON) {
        return vec3f(0.0);
    }
    return v / len;
}

fn distance_squared(a: vec3f, b: vec3f) -> f32 {
    let d = a - b;
    return dot(d, d);
}

// ============================================================================
// Clear Forces Kernel
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn clear_forces(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    if (idx >= params.node_count) {
        return;
    }
    
    nodes[idx].force = vec3f(0.0);
}

// ============================================================================
// Center Force Kernel
// ============================================================================

// First pass: compute weighted sum for center of mass
// Uses atomic operations or reduction pattern
@compute @workgroup_size(WORKGROUP_SIZE)
fn apply_center_force(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    if (idx >= params.node_count) {
        return;
    }
    
    if (nodes[idx].pinned == 1u) {
        return;
    }
    
    let strength = params.center_strength * params.alpha;
    let pos = nodes[idx].position;
    
    // Pull towards origin (0, 0, 0)
    let force = -pos * strength;
    
    nodes[idx].force += force;
}

// ============================================================================
// Charge (Repulsion) Force Kernel - O(n²) version for small graphs
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn apply_charge_force_naive(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    if (idx >= params.node_count) {
        return;
    }
    
    if (nodes[idx].pinned == 1u) {
        return;
    }
    
    let pos = nodes[idx].position;
    let charge = nodes[idx].charge;
    let dist_min_sq = params.charge_distance_min * params.charge_distance_min;
    let dist_max_sq = params.charge_distance_max * params.charge_distance_max;
    
    var total_force = vec3f(0.0);
    
    // Calculate repulsion from all other nodes
    for (var j = 0u; j < params.node_count; j++) {
        if (j == idx) {
            continue;
        }
        
        let other_pos = nodes[j].position;
        let other_charge = nodes[j].charge;
        
        let delta = pos - other_pos;
        var dist_sq = dot(delta, delta);
        
        // Skip if too far
        if (dist_sq > dist_max_sq) {
            continue;
        }
        
        // Clamp minimum distance
        dist_sq = max(dist_sq, dist_min_sq);
        
        let dist = sqrt(dist_sq);
        
        // Coulomb-like repulsion: F = k * q1 * q2 / r²
        // Using negative charge_strength for repulsion
        let magnitude = params.charge_strength * params.alpha * charge * other_charge / dist_sq;
        
        // Add force in direction away from other node
        let direction = safe_normalize(delta);
        total_force += direction * magnitude;
    }
    
    nodes[idx].force += total_force;
}

// ============================================================================
// Charge Force with Shared Memory Optimization
// ============================================================================

var<workgroup> shared_positions: array<vec3f, WORKGROUP_SIZE>;
var<workgroup> shared_charges: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn apply_charge_force_tiled(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    
    // Load this node's data
    var my_pos = vec3f(0.0);
    var my_charge = 1.0;
    var is_pinned = false;
    
    if (idx < params.node_count) {
        my_pos = nodes[idx].position;
        my_charge = nodes[idx].charge;
        is_pinned = nodes[idx].pinned == 1u;
    }
    
    var total_force = vec3f(0.0);
    let dist_min_sq = params.charge_distance_min * params.charge_distance_min;
    let dist_max_sq = params.charge_distance_max * params.charge_distance_max;
    
    // Process all nodes in tiles
    let num_tiles = (params.node_count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    
    for (var tile = 0u; tile < num_tiles; tile++) {
        // Load tile data into shared memory
        let tile_idx = tile * WORKGROUP_SIZE + local_idx;
        if (tile_idx < params.node_count) {
            shared_positions[local_idx] = nodes[tile_idx].position;
            shared_charges[local_idx] = nodes[tile_idx].charge;
        } else {
            shared_positions[local_idx] = vec3f(0.0);
            shared_charges[local_idx] = 0.0;
        }
        
        workgroupBarrier();
        
        // Calculate forces from all nodes in this tile
        if (idx < params.node_count && !is_pinned) {
            for (var j = 0u; j < WORKGROUP_SIZE; j++) {
                let other_idx = tile * WORKGROUP_SIZE + j;
                if (other_idx >= params.node_count || other_idx == idx) {
                    continue;
                }
                
                let other_pos = shared_positions[j];
                let other_charge = shared_charges[j];
                
                let delta = my_pos - other_pos;
                var dist_sq = dot(delta, delta);
                
                if (dist_sq > dist_max_sq || dist_sq < EPSILON) {
                    continue;
                }
                
                dist_sq = max(dist_sq, dist_min_sq);
                let dist = sqrt(dist_sq);
                
                let magnitude = params.charge_strength * params.alpha * my_charge * other_charge / dist_sq;
                let direction = delta / dist;
                total_force += direction * magnitude;
            }
        }
        
        workgroupBarrier();
    }
    
    // Write result
    if (idx < params.node_count && !is_pinned) {
        nodes[idx].force += total_force;
    }
}

// ============================================================================
// Link (Spring) Force Kernel
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn apply_link_force(@builtin(global_invocation_id) global_id: vec3u) {
    let edge_idx = global_id.x;
    if (edge_idx >= params.edge_count) {
        return;
    }
    
    let edge = edges[edge_idx];
    let source_idx = edge.source_idx;
    let target_idx = edge.target_idx;
    
    let source_pos = nodes[source_idx].position;
    let target_pos = nodes[target_idx].position;
    let source_pinned = nodes[source_idx].pinned == 1u;
    let target_pinned = nodes[target_idx].pinned == 1u;
    
    let delta = target_pos - source_pos;
    let dist = max(length(delta), EPSILON);
    let direction = delta / dist;
    
    // Spring force: F = k * (dist - ideal_dist)
    let displacement = dist - params.link_distance;
    let force_magnitude = displacement * params.link_strength * params.alpha * edge.weight;
    
    let force = direction * force_magnitude;
    
    // Apply forces (atomic operations for thread safety)
    // Note: WGSL doesn't have atomic floats, so we use a workaround
    // In practice, we may need to use a separate reduction pass
    if (!source_pinned) {
        // Approximation: direct write (may have race conditions for shared nodes)
        nodes[source_idx].force += force;
    }
    
    if (!target_pinned) {
        nodes[target_idx].force -= force;
    }
}

// ============================================================================
// Collision Force Kernel
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn apply_collision_force(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    if (idx >= params.node_count) {
        return;
    }
    
    if (nodes[idx].pinned == 1u || params.collision_radius_mult <= 0.0) {
        return;
    }
    
    let pos = nodes[idx].position;
    let radius = nodes[idx].radius * params.collision_radius_mult;
    
    var total_force = vec3f(0.0);
    
    for (var j = 0u; j < params.node_count; j++) {
        if (j == idx) {
            continue;
        }
        
        let other_pos = nodes[j].position;
        let other_radius = nodes[j].radius * params.collision_radius_mult;
        let min_dist = radius + other_radius;
        
        let delta = pos - other_pos;
        let dist = length(delta);
        
        if (dist < min_dist && dist > EPSILON) {
            let overlap = min_dist - dist;
            let force_magnitude = overlap * 0.5 * params.alpha;
            let direction = delta / dist;
            total_force += direction * force_magnitude;
        }
    }
    
    nodes[idx].force += total_force;
}

// ============================================================================
// Integration Kernel - Update velocities and positions
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE)
fn integrate(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    if (idx >= params.node_count) {
        return;
    }
    
    if (nodes[idx].pinned == 1u) {
        nodes[idx].velocity = vec3f(0.0);
        return;
    }
    
    // Update velocity: v = (v + F/m) * decay
    let mass = max(nodes[idx].mass, 0.1);
    var velocity = nodes[idx].velocity + nodes[idx].force / mass;
    
    // Apply velocity decay (friction)
    velocity *= params.velocity_decay;
    
    // Clamp velocity to prevent explosions
    let max_velocity = 10.0;
    let speed = length(velocity);
    if (speed > max_velocity) {
        velocity = velocity * (max_velocity / speed);
    }
    
    // Update position
    var position = nodes[idx].position + velocity;
    
    // For 2D mode, keep z = 0
    if (params.is_3d == 0u) {
        position.z = 0.0;
        velocity.z = 0.0;
    }
    
    nodes[idx].velocity = velocity;
    nodes[idx].position = position;
}

// ============================================================================
// Energy Calculation Kernel (for convergence detection)
// ============================================================================

@group(0) @binding(3) var<storage, read_write> energy_output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn calculate_energy(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    let idx = global_id.x;
    
    var kinetic_energy = 0.0;
    
    if (idx < params.node_count) {
        let velocity = nodes[idx].velocity;
        let mass = nodes[idx].mass;
        kinetic_energy = 0.5 * mass * dot(velocity, velocity);
    }
    
    // Store per-node energy for later reduction
    if (idx < params.node_count) {
        energy_output[idx] = kinetic_energy;
    }
}
