/**
 * WGSL Compute Shader for GPU Frustum Culling
 * 
 * Performs visibility testing of nodes and edges against camera frustum planes.
 * Uses compute shaders for parallel culling of large datasets.
 * 
 * @module @neurectomy/3d-engine/webgpu/culling/frustum-cull.wgsl
 * @agents @VELOCITY @CORE
 */

// =============================================================================
// Structures
// =============================================================================

struct FrustumPlane {
    normal: vec3<f32>,
    distance: f32,
}

struct Frustum {
    planes: array<FrustumPlane, 6>,  // Left, Right, Bottom, Top, Near, Far
    cameraPosition: vec3<f32>,
    maxDistance: f32,  // Max render distance for LOD
}

struct CullParams {
    nodeCount: u32,
    edgeCount: u32,
    enableDistanceCull: u32,  // Boolean as uint
    _padding: u32,
}

struct NodeBounds {
    center: vec3<f32>,
    radius: f32,
}

struct EdgeBounds {
    sourceCenter: vec3<f32>,
    sourceRadius: f32,
    targetCenter: vec3<f32>,
    targetRadius: f32,
}

struct CullResult {
    visibleNodeCount: atomic<u32>,
    visibleEdgeCount: atomic<u32>,
    _padding: vec2<u32>,
}

// =============================================================================
// Bindings
// =============================================================================

@group(0) @binding(0) var<uniform> frustum: Frustum;
@group(0) @binding(1) var<uniform> params: CullParams;

// Input bounds
@group(1) @binding(0) var<storage, read> nodeBounds: array<NodeBounds>;
@group(1) @binding(1) var<storage, read> edgeBounds: array<EdgeBounds>;

// Output visibility flags (1 = visible, 0 = culled)
@group(1) @binding(2) var<storage, read_write> nodeVisibility: array<u32>;
@group(1) @binding(3) var<storage, read_write> edgeVisibility: array<u32>;

// Indirect draw arguments (for instanced rendering)
@group(1) @binding(4) var<storage, read_write> cullResult: CullResult;

// Visible index lists (for compaction)
@group(1) @binding(5) var<storage, read_write> visibleNodeIndices: array<u32>;
@group(1) @binding(6) var<storage, read_write> visibleEdgeIndices: array<u32>;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Test if a sphere is inside or intersects a plane
 * Returns: > 0 = fully in front, < 0 = fully behind, 0 = intersecting
 */
fn spherePlaneTest(center: vec3<f32>, radius: f32, plane: FrustumPlane) -> f32 {
    let distance = dot(plane.normal, center) + plane.distance;
    return distance + radius;  // Positive if sphere is at least partially in front
}

/**
 * Test if a sphere is inside or intersects the frustum
 * Returns true if visible (not culled)
 */
fn sphereFrustumTest(center: vec3<f32>, radius: f32) -> bool {
    // Test against all 6 frustum planes
    for (var i = 0u; i < 6u; i = i + 1u) {
        if (spherePlaneTest(center, radius, frustum.planes[i]) < 0.0) {
            return false;  // Sphere is completely behind this plane
        }
    }
    return true;
}

/**
 * Test if a sphere is within render distance
 */
fn distanceCullTest(center: vec3<f32>, radius: f32) -> bool {
    if (params.enableDistanceCull == 0u) {
        return true;  // Distance culling disabled
    }
    
    let distSq = dot(
        center - frustum.cameraPosition,
        center - frustum.cameraPosition
    );
    let maxDistWithRadius = frustum.maxDistance + radius;
    return distSq <= maxDistWithRadius * maxDistWithRadius;
}

/**
 * Combined visibility test for a sphere
 */
fn isNodeVisible(bounds: NodeBounds) -> bool {
    // First do cheap distance test
    if (!distanceCullTest(bounds.center, bounds.radius)) {
        return false;
    }
    
    // Then do frustum test
    return sphereFrustumTest(bounds.center, bounds.radius);
}

/**
 * Test if an edge (line segment with radius) is visible
 * Uses a conservative AABB test on the edge's bounding capsule
 */
fn isEdgeVisible(bounds: EdgeBounds) -> bool {
    // Compute bounding sphere of the edge capsule
    let midpoint = (bounds.sourceCenter + bounds.targetCenter) * 0.5;
    let halfLength = length(bounds.targetCenter - bounds.sourceCenter) * 0.5;
    let maxRadius = max(bounds.sourceRadius, bounds.targetRadius);
    let boundingRadius = halfLength + maxRadius;
    
    // Test bounding sphere
    if (!distanceCullTest(midpoint, boundingRadius)) {
        return false;
    }
    
    return sphereFrustumTest(midpoint, boundingRadius);
}

// =============================================================================
// Node Culling Kernel
// =============================================================================

@compute @workgroup_size(256)
fn cullNodes(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let nodeIdx = globalId.x;
    
    if (nodeIdx >= params.nodeCount) {
        return;
    }
    
    let bounds = nodeBounds[nodeIdx];
    let visible = isNodeVisible(bounds);
    
    // Store visibility flag
    nodeVisibility[nodeIdx] = select(0u, 1u, visible);
    
    // If visible, add to visible list
    if (visible) {
        let visibleIdx = atomicAdd(&cullResult.visibleNodeCount, 1u);
        visibleNodeIndices[visibleIdx] = nodeIdx;
    }
}

// =============================================================================
// Edge Culling Kernel
// =============================================================================

@compute @workgroup_size(256)
fn cullEdges(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let edgeIdx = globalId.x;
    
    if (edgeIdx >= params.edgeCount) {
        return;
    }
    
    let bounds = edgeBounds[edgeIdx];
    let visible = isEdgeVisible(bounds);
    
    // Store visibility flag
    edgeVisibility[edgeIdx] = select(0u, 1u, visible);
    
    // If visible, add to visible list
    if (visible) {
        let visibleIdx = atomicAdd(&cullResult.visibleEdgeCount, 1u);
        visibleEdgeIndices[visibleIdx] = edgeIdx;
    }
}

// =============================================================================
// Reset Counters Kernel
// =============================================================================

@compute @workgroup_size(1)
fn resetCounters() {
    atomicStore(&cullResult.visibleNodeCount, 0u);
    atomicStore(&cullResult.visibleEdgeCount, 0u);
}

// =============================================================================
// Frustum Extraction from ViewProjection Matrix
// =============================================================================

/**
 * This kernel extracts frustum planes from a view-projection matrix.
 * Run with workgroup_size(1) - single thread operation.
 * 
 * ViewProjection matrix layout (column-major):
 * [m00 m01 m02 m03]
 * [m10 m11 m12 m13]
 * [m20 m21 m22 m23]
 * [m30 m31 m32 m33]
 */

struct ViewProjectionMatrix {
    m: array<f32, 16>,
}

@group(2) @binding(0) var<uniform> viewProj: ViewProjectionMatrix;
@group(2) @binding(1) var<storage, read_write> extractedFrustum: Frustum;

fn getMatrixElement(row: u32, col: u32) -> f32 {
    return viewProj.m[col * 4u + row];
}

@compute @workgroup_size(1)
fn extractFrustum() {
    // Left plane: row3 + row0
    extractedFrustum.planes[0].normal = vec3<f32>(
        getMatrixElement(3u, 0u) + getMatrixElement(0u, 0u),
        getMatrixElement(3u, 1u) + getMatrixElement(0u, 1u),
        getMatrixElement(3u, 2u) + getMatrixElement(0u, 2u)
    );
    extractedFrustum.planes[0].distance = getMatrixElement(3u, 3u) + getMatrixElement(0u, 3u);
    
    // Right plane: row3 - row0
    extractedFrustum.planes[1].normal = vec3<f32>(
        getMatrixElement(3u, 0u) - getMatrixElement(0u, 0u),
        getMatrixElement(3u, 1u) - getMatrixElement(0u, 1u),
        getMatrixElement(3u, 2u) - getMatrixElement(0u, 2u)
    );
    extractedFrustum.planes[1].distance = getMatrixElement(3u, 3u) - getMatrixElement(0u, 3u);
    
    // Bottom plane: row3 + row1
    extractedFrustum.planes[2].normal = vec3<f32>(
        getMatrixElement(3u, 0u) + getMatrixElement(1u, 0u),
        getMatrixElement(3u, 1u) + getMatrixElement(1u, 1u),
        getMatrixElement(3u, 2u) + getMatrixElement(1u, 2u)
    );
    extractedFrustum.planes[2].distance = getMatrixElement(3u, 3u) + getMatrixElement(1u, 3u);
    
    // Top plane: row3 - row1
    extractedFrustum.planes[3].normal = vec3<f32>(
        getMatrixElement(3u, 0u) - getMatrixElement(1u, 0u),
        getMatrixElement(3u, 1u) - getMatrixElement(1u, 1u),
        getMatrixElement(3u, 2u) - getMatrixElement(1u, 2u)
    );
    extractedFrustum.planes[3].distance = getMatrixElement(3u, 3u) - getMatrixElement(1u, 3u);
    
    // Near plane: row3 + row2 (for reversed-Z: row2 only)
    extractedFrustum.planes[4].normal = vec3<f32>(
        getMatrixElement(3u, 0u) + getMatrixElement(2u, 0u),
        getMatrixElement(3u, 1u) + getMatrixElement(2u, 1u),
        getMatrixElement(3u, 2u) + getMatrixElement(2u, 2u)
    );
    extractedFrustum.planes[4].distance = getMatrixElement(3u, 3u) + getMatrixElement(2u, 3u);
    
    // Far plane: row3 - row2
    extractedFrustum.planes[5].normal = vec3<f32>(
        getMatrixElement(3u, 0u) - getMatrixElement(2u, 0u),
        getMatrixElement(3u, 1u) - getMatrixElement(2u, 1u),
        getMatrixElement(3u, 2u) - getMatrixElement(2u, 2u)
    );
    extractedFrustum.planes[5].distance = getMatrixElement(3u, 3u) - getMatrixElement(2u, 3u);
    
    // Normalize all planes
    for (var i = 0u; i < 6u; i = i + 1u) {
        let len = length(extractedFrustum.planes[i].normal);
        if (len > 0.0001) {
            extractedFrustum.planes[i].normal = extractedFrustum.planes[i].normal / len;
            extractedFrustum.planes[i].distance = extractedFrustum.planes[i].distance / len;
        }
    }
}

// =============================================================================
// Hierarchical Culling (for BVH/Octree)
// =============================================================================

struct BVHNode {
    minBounds: vec3<f32>,
    leftChild: u32,  // If MSB set, this is a leaf with index in lower bits
    maxBounds: vec3<f32>,
    rightChild: u32,
}

@group(3) @binding(0) var<storage, read> bvhNodes: array<BVHNode>;
@group(3) @binding(1) var<storage, read_write> nodeStack: array<u32>;
@group(3) @binding(2) var<storage, read_write> stackPointer: atomic<u32>;

/**
 * Test AABB against frustum
 */
fn aabbFrustumTest(minBounds: vec3<f32>, maxBounds: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = frustum.planes[i];
        
        // Find the corner most in the direction of the normal
        let testPoint = vec3<f32>(
            select(minBounds.x, maxBounds.x, plane.normal.x >= 0.0),
            select(minBounds.y, maxBounds.y, plane.normal.y >= 0.0),
            select(minBounds.z, maxBounds.z, plane.normal.z >= 0.0)
        );
        
        if (dot(plane.normal, testPoint) + plane.distance < 0.0) {
            return false;
        }
    }
    return true;
}

/**
 * Hierarchical BVH traversal for culling
 * Uses a stack-based approach for iterative traversal
 */
@compute @workgroup_size(64)
fn cullBVH(@builtin(local_invocation_id) localId: vec3<u32>) {
    // Only thread 0 processes the BVH
    if (localId.x != 0u) {
        return;
    }
    
    // Initialize stack with root
    atomicStore(&stackPointer, 1u);
    nodeStack[0] = 0u;  // Root node index
    
    while (atomicLoad(&stackPointer) > 0u) {
        // Pop from stack
        let stackIdx = atomicSub(&stackPointer, 1u) - 1u;
        let nodeIdx = nodeStack[stackIdx];
        
        let node = bvhNodes[nodeIdx];
        
        // Test against frustum
        if (!aabbFrustumTest(node.minBounds, node.maxBounds)) {
            continue;  // Entire subtree is culled
        }
        
        // Check if leaf node (MSB set)
        let isLeftLeaf = (node.leftChild & 0x80000000u) != 0u;
        let isRightLeaf = (node.rightChild & 0x80000000u) != 0u;
        
        if (isLeftLeaf) {
            let leafIdx = node.leftChild & 0x7FFFFFFFu;
            nodeVisibility[leafIdx] = 1u;
            let visibleIdx = atomicAdd(&cullResult.visibleNodeCount, 1u);
            visibleNodeIndices[visibleIdx] = leafIdx;
        } else if (node.leftChild != 0xFFFFFFFFu) {
            // Push left child to stack
            let pushIdx = atomicAdd(&stackPointer, 1u);
            nodeStack[pushIdx] = node.leftChild;
        }
        
        if (isRightLeaf) {
            let leafIdx = node.rightChild & 0x7FFFFFFFu;
            nodeVisibility[leafIdx] = 1u;
            let visibleIdx = atomicAdd(&cullResult.visibleNodeCount, 1u);
            visibleNodeIndices[visibleIdx] = leafIdx;
        } else if (node.rightChild != 0xFFFFFFFFu) {
            // Push right child to stack
            let pushIdx = atomicAdd(&stackPointer, 1u);
            nodeStack[pushIdx] = node.rightChild;
        }
    }
}
