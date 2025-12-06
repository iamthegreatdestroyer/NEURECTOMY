/**
 * WGSL Shaders for Graph Rendering
 * 
 * Native WebGPU shaders for high-performance node and edge rendering
 * using instanced drawing for maximum efficiency.
 * 
 * @module @neurectomy/3d-engine/webgpu/graph/shaders
 * @agents @CORE @VELOCITY
 */

// =============================================================================
// Common Structures
// =============================================================================

struct Camera {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    viewProjection: mat4x4<f32>,
    position: vec3<f32>,
    near: f32,
    far: f32,
    fov: f32,
    aspect: f32,
    _padding: f32,
}

struct SceneUniforms {
    time: f32,
    deltaTime: f32,
    viewportSize: vec2<f32>,
    highlightedNodeId: u32,
    selectedNodeId: u32,
    hoveredNodeId: u32,
    _padding: u32,
}

// =============================================================================
// Node Rendering - Instanced Spheres / Billboards
// =============================================================================

struct NodeInstance {
    position: vec3<f32>,
    radius: f32,
    color: vec4<f32>,
    nodeId: u32,
    flags: u32,  // bit 0: selected, bit 1: highlighted, bit 2: pinned
    _padding: vec2<f32>,
}

struct NodeVertexInput {
    @location(0) vertexPosition: vec3<f32>,  // Unit sphere vertex
    @location(1) vertexNormal: vec3<f32>,    // Unit sphere normal
    @location(2) vertexUV: vec2<f32>,        // UV coords
}

struct NodeVertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) worldPosition: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) nodeId: u32,
    @location(5) @interpolate(flat) flags: u32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> scene: SceneUniforms;
@group(1) @binding(0) var<storage, read> nodeInstances: array<NodeInstance>;

@vertex
fn vs_node_main(
    vertex: NodeVertexInput,
    @builtin(instance_index) instanceIdx: u32
) -> NodeVertexOutput {
    let instance = nodeInstances[instanceIdx];
    
    // Scale and translate vertex position
    let worldPos = vertex.vertexPosition * instance.radius + instance.position;
    
    // Transform normal (no scaling for unit sphere)
    let worldNormal = vertex.vertexNormal;
    
    var output: NodeVertexOutput;
    output.clipPosition = camera.viewProjection * vec4<f32>(worldPos, 1.0);
    output.worldPosition = worldPos;
    output.worldNormal = worldNormal;
    output.color = instance.color;
    output.uv = vertex.vertexUV;
    output.nodeId = instance.nodeId;
    output.flags = instance.flags;
    
    return output;
}

@fragment
fn fs_node_main(input: NodeVertexOutput) -> @location(0) vec4<f32> {
    // Simple lighting calculation
    let lightDir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let viewDir = normalize(camera.position - input.worldPosition);
    let halfDir = normalize(lightDir + viewDir);
    
    // Lambertian diffuse
    let ndotl = max(dot(input.worldNormal, lightDir), 0.0);
    let diffuse = ndotl * 0.7 + 0.3; // Add ambient
    
    // Blinn-Phong specular
    let ndoth = max(dot(input.worldNormal, halfDir), 0.0);
    let specular = pow(ndoth, 32.0) * 0.3;
    
    // Base color
    var color = input.color.rgb * diffuse + vec3<f32>(1.0) * specular;
    
    // Selection highlight
    let isSelected = (input.flags & 1u) != 0u;
    let isHighlighted = (input.flags & 2u) != 0u;
    
    if (isSelected) {
        color = mix(color, vec3<f32>(0.2, 0.8, 1.0), 0.4);
    }
    
    if (isHighlighted) {
        // Pulsing highlight
        let pulse = sin(scene.time * 3.0) * 0.5 + 0.5;
        color = mix(color, vec3<f32>(1.0, 0.8, 0.2), 0.3 * pulse);
    }
    
    // Fresnel rim light
    let ndotv = max(dot(input.worldNormal, viewDir), 0.0);
    let fresnel = pow(1.0 - ndotv, 3.0) * 0.2;
    color += vec3<f32>(0.3, 0.5, 0.8) * fresnel;
    
    return vec4<f32>(color, input.color.a);
}


// =============================================================================
// Edge Rendering - Instanced Line Segments
// =============================================================================

struct EdgeInstance {
    sourcePosition: vec3<f32>,
    sourceRadius: f32,
    targetPosition: vec3<f32>,
    targetRadius: f32,
    color: vec4<f32>,
    edgeId: u32,
    flags: u32,  // bit 0: selected, bit 1: bidirectional, bit 2: animated
    width: f32,
    _padding: f32,
}

struct EdgeVertexInput {
    @builtin(vertex_index) vertexIdx: u32,
}

struct EdgeVertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) edgeId: u32,
    @location(3) @interpolate(flat) flags: u32,
}

@group(1) @binding(1) var<storage, read> edgeInstances: array<EdgeInstance>;

// Generate tube/capsule geometry from line segment
// Vertex indices 0-23 form a cylinder with caps
fn getEdgeVertex(vertexIdx: u32, edge: EdgeInstance) -> vec3<f32> {
    let segments = 8u;
    let segmentIdx = vertexIdx / 3u;
    let localIdx = vertexIdx % 3u;
    
    // Direction and perpendicular vectors
    let direction = normalize(edge.targetPosition - edge.sourcePosition);
    let length = distance(edge.sourcePosition, edge.targetPosition);
    
    // Create perpendicular vectors
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(direction, up)) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(direction, up));
    let realUp = cross(right, direction);
    
    // Cylinder parameters
    let angle = f32(segmentIdx % segments) / f32(segments) * 6.28318530718;
    let nextAngle = f32((segmentIdx + 1u) % segments) / f32(segments) * 6.28318530718;
    let isEndCap = segmentIdx >= segments * 2u;
    let isSourceEnd = segmentIdx < segments || (isEndCap && segmentIdx < segments * 3u);
    
    let radius = edge.width * 0.5;
    
    if (isEndCap) {
        // Cap vertices
        let basePos = select(edge.targetPosition, edge.sourcePosition, isSourceEnd);
        let capDir = select(direction, -direction, isSourceEnd);
        
        if (localIdx == 0u) {
            return basePos + capDir * radius;
        }
        let a = select(angle, nextAngle, localIdx == 2u);
        return basePos + (cos(a) * right + sin(a) * realUp) * radius;
    }
    
    // Cylinder body
    let t = select(1.0, 0.0, isSourceEnd);
    let pos = mix(edge.sourcePosition, edge.targetPosition, t);
    let a = select(angle, nextAngle, localIdx == 1u || localIdx == 2u);
    let t2 = select(0.0, 1.0, localIdx >= 2u);
    let pos2 = mix(edge.sourcePosition, edge.targetPosition, select(t, t + 1.0, isSourceEnd) * t2);
    
    return pos + (cos(a) * right + sin(a) * realUp) * radius;
}

@vertex
fn vs_edge_main(
    input: EdgeVertexInput,
    @builtin(instance_index) instanceIdx: u32
) -> EdgeVertexOutput {
    let edge = edgeInstances[instanceIdx];
    
    // Calculate vertex position on cylinder
    let worldPos = getEdgeVertex(input.vertexIdx, edge);
    
    var output: EdgeVertexOutput;
    output.clipPosition = camera.viewProjection * vec4<f32>(worldPos, 1.0);
    output.color = edge.color;
    output.uv = vec2<f32>(f32(input.vertexIdx % 2u), f32(input.vertexIdx / 2u));
    output.edgeId = edge.edgeId;
    output.flags = edge.flags;
    
    return output;
}

@fragment
fn fs_edge_main(input: EdgeVertexOutput) -> @location(0) vec4<f32> {
    var color = input.color;
    
    // Selection/animation effects
    let isSelected = (input.flags & 1u) != 0u;
    let isAnimated = (input.flags & 4u) != 0u;
    
    if (isSelected) {
        color = vec4<f32>(mix(color.rgb, vec3<f32>(0.2, 0.8, 1.0), 0.5), color.a);
    }
    
    if (isAnimated) {
        // Flow animation
        let flow = fract(scene.time * 0.5 + input.uv.x);
        let pulse = smoothstep(0.4, 0.5, flow) * smoothstep(0.6, 0.5, flow);
        color = vec4<f32>(color.rgb + vec3<f32>(0.3, 0.5, 0.8) * pulse, color.a);
    }
    
    return color;
}


// =============================================================================
// Edge Rendering - Simple Lines (Alternative, Lower Quality, Higher Performance)
// =============================================================================

struct LineVertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) progress: f32,
    @location(2) @interpolate(flat) edgeId: u32,
}

@vertex
fn vs_edge_line_main(
    @builtin(vertex_index) vertexIdx: u32,
    @builtin(instance_index) instanceIdx: u32
) -> LineVertexOutput {
    let edge = edgeInstances[instanceIdx];
    
    // Simple line: vertex 0 = source, vertex 1 = target
    let t = f32(vertexIdx);
    let worldPos = mix(edge.sourcePosition, edge.targetPosition, t);
    
    var output: LineVertexOutput;
    output.clipPosition = camera.viewProjection * vec4<f32>(worldPos, 1.0);
    output.color = edge.color;
    output.progress = t;
    output.edgeId = edge.edgeId;
    
    return output;
}

@fragment
fn fs_edge_line_main(input: LineVertexOutput) -> @location(0) vec4<f32> {
    var color = input.color;
    
    // Flow animation
    let flow = fract(scene.time * 0.5 - input.progress);
    let pulse = smoothstep(0.0, 0.1, flow) * smoothstep(0.2, 0.1, flow);
    color = vec4<f32>(color.rgb + vec3<f32>(0.2, 0.4, 0.6) * pulse * 0.5, color.a);
    
    return color;
}


// =============================================================================
// Billboard Nodes (Alternative, for large graphs)
// =============================================================================

struct BillboardVertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) nodeId: u32,
}

// Quad vertices for billboard: 4 vertices per instance
const BILLBOARD_OFFSETS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0)
);

@vertex
fn vs_billboard_main(
    @builtin(vertex_index) vertexIdx: u32,
    @builtin(instance_index) instanceIdx: u32
) -> BillboardVertexOutput {
    let instance = nodeInstances[instanceIdx];
    let quadIdx = vertexIdx % 4u;
    let offset = BILLBOARD_OFFSETS[quadIdx];
    
    // Billboard: always face camera
    let viewPos = camera.view * vec4<f32>(instance.position, 1.0);
    let billboardPos = viewPos + vec4<f32>(offset * instance.radius, 0.0, 0.0);
    
    var output: BillboardVertexOutput;
    output.clipPosition = camera.projection * billboardPos;
    output.color = instance.color;
    output.uv = offset * 0.5 + 0.5;
    output.nodeId = instance.nodeId;
    
    return output;
}

@fragment
fn fs_billboard_main(input: BillboardVertexOutput) -> @location(0) vec4<f32> {
    // Circular billboard with soft edge
    let dist = length(input.uv - 0.5) * 2.0;
    
    if (dist > 1.0) {
        discard;
    }
    
    // Soft edge
    let alpha = 1.0 - smoothstep(0.8, 1.0, dist);
    
    // Pseudo-3D shading
    let normal = vec3<f32>(input.uv.x - 0.5, input.uv.y - 0.5, sqrt(max(0.0, 1.0 - dist * dist)));
    let lightDir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let diffuse = max(dot(normal, lightDir), 0.0) * 0.6 + 0.4;
    
    return vec4<f32>(input.color.rgb * diffuse, input.color.a * alpha);
}


// =============================================================================
// Depth Pre-Pass (for large scenes)
// =============================================================================

struct DepthOnlyOutput {
    @builtin(position) clipPosition: vec4<f32>,
}

@vertex
fn vs_depth_node_main(
    vertex: NodeVertexInput,
    @builtin(instance_index) instanceIdx: u32
) -> DepthOnlyOutput {
    let instance = nodeInstances[instanceIdx];
    let worldPos = vertex.vertexPosition * instance.radius + instance.position;
    
    var output: DepthOnlyOutput;
    output.clipPosition = camera.viewProjection * vec4<f32>(worldPos, 1.0);
    return output;
}

@fragment
fn fs_depth_main() {
    // Depth-only pass, no color output
}
