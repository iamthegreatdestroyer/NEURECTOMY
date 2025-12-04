/**
 * Scene Graph Management
 * 
 * Hierarchical scene graph with optimized traversal, frustum culling,
 * and spatial indexing for large-scale agent visualization.
 * 
 * @module @neurectomy/3d-engine/three/scene-graph
 * @agents @APEX @ARCHITECT
 */

import * as THREE from 'three';
import type { SceneNode, TransformData } from '../core/types';

// =============================================================================
// Types
// =============================================================================

export interface SceneGraphConfig {
  maxDepth?: number;
  enableSpatialIndex?: boolean;
  frustumCulling?: boolean;
  lodEnabled?: boolean;
  shadowsEnabled?: boolean;
}

export interface SceneLayer {
  id: string;
  name: string;
  visible: boolean;
  order: number;
  objects: THREE.Object3D[];
}

export interface SceneStatistics {
  totalNodes: number;
  visibleNodes: number;
  triangleCount: number;
  drawCalls: number;
  textureCount: number;
  lightCount: number;
}

export interface OctreeNode {
  bounds: THREE.Box3;
  objects: THREE.Object3D[];
  children: OctreeNode[];
  depth: number;
}

export interface LODConfig {
  distances: number[];
  bias?: number;
}

// =============================================================================
// SceneGraph Class
// =============================================================================

/**
 * SceneGraph - Optimized scene management for agent visualization
 * 
 * Features:
 * - Hierarchical scene structure with Three.js Group nodes
 * - Octree spatial indexing for efficient queries
 * - Automatic frustum culling
 * - Layer-based rendering order
 * - LOD (Level of Detail) support
 * - Shadow map management
 */
export class SceneGraph {
  public readonly scene: THREE.Scene;
  private config: Required<SceneGraphConfig>;
  private layers = new Map<string, SceneLayer>();
  private nodeMap = new Map<string, THREE.Object3D>();
  private octree?: OctreeNode;
  private boundingBox = new THREE.Box3();
  private statistics: SceneStatistics = {
    totalNodes: 0,
    visibleNodes: 0,
    triangleCount: 0,
    drawCalls: 0,
    textureCount: 0,
    lightCount: 0,
  };

  // Pre-allocated objects for culling
  private frustum = new THREE.Frustum();
  private projScreenMatrix = new THREE.Matrix4();
  private tempBox = new THREE.Box3();
  private tempSphere = new THREE.Sphere();

  constructor(config: SceneGraphConfig = {}) {
    this.config = {
      maxDepth: config.maxDepth ?? 8,
      enableSpatialIndex: config.enableSpatialIndex ?? true,
      frustumCulling: config.frustumCulling ?? true,
      lodEnabled: config.lodEnabled ?? true,
      shadowsEnabled: config.shadowsEnabled ?? true,
    };

    this.scene = new THREE.Scene();
    this.scene.name = 'neurectomy-scene';

    // Create default layers
    this.createLayer('background', -100);
    this.createLayer('environment', -50);
    this.createLayer('agents', 0);
    this.createLayer('connections', 10);
    this.createLayer('ui', 100);
    this.createLayer('effects', 150);

    // Default environment
    this.setupDefaultEnvironment();
  }

  /**
   * Setup default scene environment
   */
  private setupDefaultEnvironment(): void {
    // Ambient light
    const ambient = new THREE.AmbientLight(0x404050, 0.4);
    ambient.name = 'ambient-light';
    this.add('ambient-light', ambient, 'environment');

    // Directional light (sun)
    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.name = 'sun-light';
    directional.position.set(50, 100, 50);
    directional.castShadow = this.config.shadowsEnabled;
    
    if (directional.castShadow) {
      directional.shadow.mapSize.width = 2048;
      directional.shadow.mapSize.height = 2048;
      directional.shadow.camera.near = 1;
      directional.shadow.camera.far = 500;
      directional.shadow.camera.left = -100;
      directional.shadow.camera.right = 100;
      directional.shadow.camera.top = 100;
      directional.shadow.camera.bottom = -100;
      directional.shadow.bias = -0.0001;
    }
    
    this.add('sun-light', directional, 'environment');

    // Hemisphere light for sky/ground
    const hemisphere = new THREE.HemisphereLight(0x87ceeb, 0x362e2e, 0.3);
    hemisphere.name = 'hemisphere-light';
    this.add('hemisphere-light', hemisphere, 'environment');
  }

  /**
   * Create a rendering layer
   */
  createLayer(id: string, order: number = 0): SceneLayer {
    const layer: SceneLayer = {
      id,
      name: id,
      visible: true,
      order,
      objects: [],
    };

    const group = new THREE.Group();
    group.name = `layer-${id}`;
    group.renderOrder = order;
    this.scene.add(group);

    this.layers.set(id, layer);
    return layer;
  }

  /**
   * Add an object to the scene
   */
  add(id: string, object: THREE.Object3D, layerId: string = 'agents'): void {
    const layer = this.layers.get(layerId);
    if (!layer) {
      console.warn(`[SceneGraph] Layer '${layerId}' not found, using 'agents'`);
      this.add(id, object, 'agents');
      return;
    }

    // Remove existing object with same ID
    if (this.nodeMap.has(id)) {
      this.remove(id);
    }

    // Find the layer group in the scene
    const layerGroup = this.scene.getObjectByName(`layer-${layerId}`) as THREE.Group;
    if (layerGroup) {
      layerGroup.add(object);
    } else {
      this.scene.add(object);
    }

    object.userData.layerId = layerId;
    object.userData.nodeId = id;

    layer.objects.push(object);
    this.nodeMap.set(id, object);

    // Update spatial index
    if (this.config.enableSpatialIndex) {
      this.markOctreeDirty();
    }

    this.updateStatistics();
  }

  /**
   * Remove an object from the scene
   */
  remove(id: string): void {
    const object = this.nodeMap.get(id);
    if (!object) return;

    const layerId = object.userData.layerId as string;
    const layer = this.layers.get(layerId);

    if (layer) {
      const index = layer.objects.indexOf(object);
      if (index !== -1) {
        layer.objects.splice(index, 1);
      }
    }

    object.removeFromParent();
    this.nodeMap.delete(id);

    // Dispose resources
    this.disposeObject(object);

    if (this.config.enableSpatialIndex) {
      this.markOctreeDirty();
    }

    this.updateStatistics();
  }

  /**
   * Get an object by ID
   */
  get(id: string): THREE.Object3D | undefined {
    return this.nodeMap.get(id);
  }

  /**
   * Check if an object exists
   */
  has(id: string): boolean {
    return this.nodeMap.has(id);
  }

  /**
   * Update object transform
   */
  setTransform(id: string, transform: Partial<TransformData>): void {
    const object = this.nodeMap.get(id);
    if (!object) return;

    if (transform.position) {
      object.position.set(
        transform.position[0],
        transform.position[1],
        transform.position[2]
      );
    }

    if (transform.rotation) {
      object.quaternion.set(
        transform.rotation[0],
        transform.rotation[1],
        transform.rotation[2],
        transform.rotation[3]
      );
    }

    if (transform.scale) {
      object.scale.set(
        transform.scale[0],
        transform.scale[1],
        transform.scale[2]
      );
    }

    object.updateMatrix();
    object.updateMatrixWorld();

    if (this.config.enableSpatialIndex) {
      this.markOctreeDirty();
    }
  }

  /**
   * Set layer visibility
   */
  setLayerVisible(layerId: string, visible: boolean): void {
    const layer = this.layers.get(layerId);
    if (layer) {
      layer.visible = visible;
      const group = this.scene.getObjectByName(`layer-${layerId}`);
      if (group) {
        group.visible = visible;
      }
    }
  }

  /**
   * Perform frustum culling
   */
  performFrustumCulling(camera: THREE.Camera): number {
    if (!this.config.frustumCulling) {
      return this.statistics.totalNodes;
    }

    camera.updateMatrixWorld();
    this.projScreenMatrix.multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse
    );
    this.frustum.setFromProjectionMatrix(this.projScreenMatrix);

    let visibleCount = 0;

    this.scene.traverse((object) => {
      if (object.userData.skipCulling) {
        object.visible = true;
        return;
      }

      // Check bounding sphere first (faster)
      if (object instanceof THREE.Mesh && object.geometry.boundingSphere) {
        this.tempSphere.copy(object.geometry.boundingSphere);
        this.tempSphere.applyMatrix4(object.matrixWorld);

        object.visible = this.frustum.intersectsSphere(this.tempSphere);
      } else if (object instanceof THREE.Mesh && object.geometry.boundingBox) {
        this.tempBox.copy(object.geometry.boundingBox);
        this.tempBox.applyMatrix4(object.matrixWorld);

        object.visible = this.frustum.intersectsBox(this.tempBox);
      } else {
        object.visible = true;
      }

      if (object.visible) {
        visibleCount++;
      }
    });

    this.statistics.visibleNodes = visibleCount;
    return visibleCount;
  }

  /**
   * Query objects within a bounding box
   */
  queryBox(box: THREE.Box3): THREE.Object3D[] {
    if (this.octree) {
      return this.queryOctree(this.octree, box);
    }

    // Fallback to linear search
    const results: THREE.Object3D[] = [];
    this.scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        this.tempBox.setFromObject(object);
        if (box.intersectsBox(this.tempBox)) {
          results.push(object);
        }
      }
    });
    return results;
  }

  /**
   * Query objects within a sphere
   */
  querySphere(center: THREE.Vector3, radius: number): THREE.Object3D[] {
    const results: THREE.Object3D[] = [];
    const radiusSq = radius * radius;

    this.scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        const distance = object.position.distanceToSquared(center);
        if (distance <= radiusSq) {
          results.push(object);
        }
      }
    });

    return results;
  }

  /**
   * Raycast against the scene
   */
  raycast(
    raycaster: THREE.Raycaster,
    layerIds?: string[]
  ): THREE.Intersection[] {
    const objects: THREE.Object3D[] = [];

    if (layerIds) {
      for (const layerId of layerIds) {
        const layer = this.layers.get(layerId);
        if (layer) {
          objects.push(...layer.objects);
        }
      }
    } else {
      // All visible layers
      for (const layer of this.layers.values()) {
        if (layer.visible) {
          objects.push(...layer.objects);
        }
      }
    }

    return raycaster.intersectObjects(objects, true);
  }

  /**
   * Update scene bounding box
   */
  updateBounds(): THREE.Box3 {
    this.boundingBox.makeEmpty();

    this.scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        this.tempBox.setFromObject(object);
        this.boundingBox.union(this.tempBox);
      }
    });

    return this.boundingBox;
  }

  /**
   * Build octree spatial index
   */
  buildOctree(): void {
    this.updateBounds();

    if (this.boundingBox.isEmpty()) {
      this.octree = undefined;
      return;
    }

    // Expand bounds slightly
    this.boundingBox.expandByScalar(1);

    this.octree = this.createOctreeNode(this.boundingBox, 0);
  }

  private createOctreeNode(bounds: THREE.Box3, depth: number): OctreeNode {
    const node: OctreeNode = {
      bounds: bounds.clone(),
      objects: [],
      children: [],
      depth,
    };

    if (depth >= this.config.maxDepth) {
      return node;
    }

    // Collect objects in this bounds
    this.scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        this.tempBox.setFromObject(object);
        if (bounds.intersectsBox(this.tempBox)) {
          node.objects.push(object);
        }
      }
    });

    // If few objects, don't subdivide
    if (node.objects.length <= 8) {
      return node;
    }

    // Subdivide into 8 children
    const center = bounds.getCenter(new THREE.Vector3());
    const min = bounds.min;
    const max = bounds.max;

    const childBounds = [
      new THREE.Box3(new THREE.Vector3(min.x, min.y, min.z), center),
      new THREE.Box3(new THREE.Vector3(center.x, min.y, min.z), new THREE.Vector3(max.x, center.y, center.z)),
      new THREE.Box3(new THREE.Vector3(min.x, center.y, min.z), new THREE.Vector3(center.x, max.y, center.z)),
      new THREE.Box3(new THREE.Vector3(center.x, center.y, min.z), new THREE.Vector3(max.x, max.y, center.z)),
      new THREE.Box3(new THREE.Vector3(min.x, min.y, center.z), new THREE.Vector3(center.x, center.y, max.z)),
      new THREE.Box3(new THREE.Vector3(center.x, min.y, center.z), new THREE.Vector3(max.x, center.y, max.z)),
      new THREE.Box3(new THREE.Vector3(min.x, center.y, center.z), new THREE.Vector3(center.x, max.y, max.z)),
      new THREE.Box3(center, max),
    ];

    node.children = childBounds.map(b => this.createOctreeNode(b, depth + 1));
    node.objects = []; // Objects are now in children

    return node;
  }

  private queryOctree(node: OctreeNode, box: THREE.Box3): THREE.Object3D[] {
    if (!node.bounds.intersectsBox(box)) {
      return [];
    }

    const results: THREE.Object3D[] = [];

    // Add objects at this node
    for (const object of node.objects) {
      this.tempBox.setFromObject(object);
      if (box.intersectsBox(this.tempBox)) {
        results.push(object);
      }
    }

    // Query children
    for (const child of node.children) {
      results.push(...this.queryOctree(child, box));
    }

    return results;
  }

  private octreeDirty = false;

  private markOctreeDirty(): void {
    this.octreeDirty = true;
  }

  /**
   * Update octree if dirty
   */
  updateOctreeIfNeeded(): void {
    if (this.octreeDirty && this.config.enableSpatialIndex) {
      this.buildOctree();
      this.octreeDirty = false;
    }
  }

  /**
   * Update scene statistics
   */
  private updateStatistics(): void {
    let totalNodes = 0;
    let triangleCount = 0;
    let textureCount = 0;
    let lightCount = 0;

    const textures = new Set<THREE.Texture>();

    this.scene.traverse((object) => {
      totalNodes++;

      if (object instanceof THREE.Mesh) {
        if (object.geometry) {
          const position = object.geometry.getAttribute('position');
          if (position) {
            triangleCount += position.count / 3;
          }
        }

        const material = object.material;
        if (Array.isArray(material)) {
          for (const mat of material) {
            this.collectTextures(mat, textures);
          }
        } else if (material) {
          this.collectTextures(material, textures);
        }
      }

      if (object instanceof THREE.Light) {
        lightCount++;
      }
    });

    this.statistics = {
      totalNodes,
      visibleNodes: totalNodes,
      triangleCount: Math.floor(triangleCount),
      drawCalls: totalNodes, // Approximate
      textureCount: textures.size,
      lightCount,
    };
  }

  private collectTextures(material: THREE.Material, textures: Set<THREE.Texture>): void {
    if (material instanceof THREE.MeshStandardMaterial) {
      if (material.map) textures.add(material.map);
      if (material.normalMap) textures.add(material.normalMap);
      if (material.roughnessMap) textures.add(material.roughnessMap);
      if (material.metalnessMap) textures.add(material.metalnessMap);
      if (material.aoMap) textures.add(material.aoMap);
      if (material.emissiveMap) textures.add(material.emissiveMap);
      if (material.envMap) textures.add(material.envMap);
    }
  }

  /**
   * Dispose of an object and its resources
   */
  private disposeObject(object: THREE.Object3D): void {
    object.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.geometry?.dispose();

        const material = child.material;
        if (Array.isArray(material)) {
          material.forEach(m => m.dispose());
        } else {
          material?.dispose();
        }
      }
    });
  }

  /**
   * Get scene statistics
   */
  getStatistics(): SceneStatistics {
    return { ...this.statistics };
  }

  /**
   * Get all layers
   */
  getLayers(): SceneLayer[] {
    return Array.from(this.layers.values()).sort((a, b) => a.order - b.order);
  }

  /**
   * Clear all objects from a layer
   */
  clearLayer(layerId: string): void {
    const layer = this.layers.get(layerId);
    if (!layer) return;

    for (const object of [...layer.objects]) {
      const nodeId = object.userData.nodeId as string;
      if (nodeId) {
        this.remove(nodeId);
      }
    }
  }

  /**
   * Clear the entire scene
   */
  clear(): void {
    for (const layer of this.layers.values()) {
      this.clearLayer(layer.id);
    }

    this.updateStatistics();
  }

  /**
   * Dispose of the scene graph
   */
  dispose(): void {
    this.clear();
    this.scene.clear();
    this.layers.clear();
    this.nodeMap.clear();
    this.octree = undefined;
    console.log('[SceneGraph] Disposed');
  }
}
