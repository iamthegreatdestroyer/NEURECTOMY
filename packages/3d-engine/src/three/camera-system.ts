/**
 * Camera System
 * 
 * Advanced camera controls with smooth transitions, constraints,
 * and multiple camera modes for agent visualization.
 * 
 * @module @neurectomy/3d-engine/three/camera-system
 * @agents @CANVAS @APEX
 */

import * as THREE from 'three';

// =============================================================================
// Types
// =============================================================================

export type CameraMode = 
  | 'orbit'      // Standard orbit around target
  | 'fly'        // Free-fly first-person
  | 'follow'     // Follow a target object
  | 'path'       // Animate along a path
  | 'orthographic' // 2D-style orthographic view
  | 'overview';  // Bird's eye view

export interface CameraConfig {
  mode?: CameraMode;
  fov?: number;
  near?: number;
  far?: number;
  position?: [number, number, number];
  target?: [number, number, number];
  up?: [number, number, number];
  minDistance?: number;
  maxDistance?: number;
  minPolarAngle?: number;
  maxPolarAngle?: number;
  dampingFactor?: number;
  enableDamping?: boolean;
  autoRotate?: boolean;
  autoRotateSpeed?: number;
  enableZoom?: boolean;
  enablePan?: boolean;
  enableRotate?: boolean;
  panSpeed?: number;
  rotateSpeed?: number;
  zoomSpeed?: number;
}

export interface CameraState {
  position: THREE.Vector3;
  target: THREE.Vector3;
  fov: number;
  zoom: number;
  quaternion: THREE.Quaternion;
}

export interface CameraTransition {
  from: CameraState;
  to: CameraState;
  duration: number;
  elapsed: number;
  easing: EasingFunction;
  onComplete?: () => void;
}

export interface CameraPath {
  points: THREE.Vector3[];
  targets?: THREE.Vector3[];
  duration: number;
  loop?: boolean;
}

export type EasingFunction = (t: number) => number;

export interface CameraConstraints {
  minPosition?: THREE.Vector3;
  maxPosition?: THREE.Vector3;
  boundingSphere?: THREE.Sphere;
  boundingBox?: THREE.Box3;
}

// =============================================================================
// Easing Functions
// =============================================================================

export const Easing = {
  linear: (t: number) => t,
  easeInQuad: (t: number) => t * t,
  easeOutQuad: (t: number) => t * (2 - t),
  easeInOutQuad: (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  easeInCubic: (t: number) => t * t * t,
  easeOutCubic: (t: number) => (--t) * t * t + 1,
  easeInOutCubic: (t: number) => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
  easeInExpo: (t: number) => t === 0 ? 0 : Math.pow(2, 10 * (t - 1)),
  easeOutExpo: (t: number) => t === 1 ? 1 : 1 - Math.pow(2, -10 * t),
  easeInOutExpo: (t: number) => {
    if (t === 0) return 0;
    if (t === 1) return 1;
    if (t < 0.5) return Math.pow(2, 20 * t - 10) / 2;
    return (2 - Math.pow(2, -20 * t + 10)) / 2;
  },
  easeOutElastic: (t: number) => {
    const c4 = (2 * Math.PI) / 3;
    return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
  },
} as const;

// =============================================================================
// Default Configuration
// =============================================================================

const DEFAULT_CONFIG: Required<CameraConfig> = {
  mode: 'orbit',
  fov: 60,
  near: 0.1,
  far: 10000,
  position: [10, 10, 10],
  target: [0, 0, 0],
  up: [0, 1, 0],
  minDistance: 1,
  maxDistance: 1000,
  minPolarAngle: 0.1,
  maxPolarAngle: Math.PI - 0.1,
  dampingFactor: 0.05,
  enableDamping: true,
  autoRotate: false,
  autoRotateSpeed: 1,
  enableZoom: true,
  enablePan: true,
  enableRotate: true,
  panSpeed: 1,
  rotateSpeed: 1,
  zoomSpeed: 1,
};

// =============================================================================
// CameraSystem Class
// =============================================================================

/**
 * CameraSystem - Advanced camera management for 3D visualization
 * 
 * Features:
 * - Multiple camera modes (orbit, fly, follow, path, orthographic, overview)
 * - Smooth transitions between states with easing
 * - Keyboard and mouse input handling
 * - Constraint system (bounds, distance limits)
 * - Camera shake effects
 * - Save/restore camera states
 */
export class CameraSystem {
  public camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
  private perspectiveCamera: THREE.PerspectiveCamera;
  private orthographicCamera: THREE.OrthographicCamera;

  private config: Required<CameraConfig>;
  private mode: CameraMode;
  private target = new THREE.Vector3();
  private spherical = new THREE.Spherical();
  private sphericalDelta = new THREE.Spherical();
  private panOffset = new THREE.Vector3();
  private scale = 1;

  // Damping state
  private dampedSpherical = new THREE.Spherical();
  private dampedTarget = new THREE.Vector3();

  // Transition state
  private transition?: CameraTransition;

  // Follow mode state
  private followTarget?: THREE.Object3D;
  private followOffset = new THREE.Vector3(0, 5, 10);
  private followLookAhead = 0;

  // Path mode state
  private currentPath?: CameraPath;
  private pathProgress = 0;

  // Shake state
  private shakeIntensity = 0;
  private shakeDecay = 0.9;
  private shakeOffset = new THREE.Vector3();

  // Constraints
  private constraints?: CameraConstraints;

  // Saved states
  private savedStates = new Map<string, CameraState>();

  // Input state
  private isPointerDown = false;
  private pointerStart = new THREE.Vector2();
  private pointerCurrent = new THREE.Vector2();
  private pointerButton = 0;

  // Temp vectors
  private tempVec3 = new THREE.Vector3();
  private tempQuat = new THREE.Quaternion();

  constructor(config: CameraConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.mode = this.config.mode;

    // Create perspective camera
    this.perspectiveCamera = new THREE.PerspectiveCamera(
      this.config.fov,
      1, // Will be updated by resize
      this.config.near,
      this.config.far
    );
    this.perspectiveCamera.position.set(...this.config.position);
    this.perspectiveCamera.up.set(...this.config.up);

    // Create orthographic camera
    this.orthographicCamera = new THREE.OrthographicCamera(
      -10, 10, 10, -10,
      this.config.near,
      this.config.far
    );
    this.orthographicCamera.position.set(...this.config.position);
    this.orthographicCamera.up.set(...this.config.up);

    // Set active camera
    this.camera = this.mode === 'orthographic' 
      ? this.orthographicCamera 
      : this.perspectiveCamera;

    // Initialize target
    this.target.set(...this.config.target);
    this.dampedTarget.copy(this.target);

    // Initialize spherical coordinates
    this.updateSphericalFromPosition();
    this.dampedSpherical.copy(this.spherical);

    // Look at target
    this.camera.lookAt(this.target);
  }

  /**
   * Update spherical coordinates from camera position
   */
  private updateSphericalFromPosition(): void {
    this.tempVec3.copy(this.camera.position).sub(this.target);
    this.spherical.setFromVector3(this.tempVec3);
    this.spherical.makeSafe();
  }

  /**
   * Update camera position from spherical coordinates
   */
  private updatePositionFromSpherical(): void {
    this.tempVec3.setFromSpherical(this.dampedSpherical);
    this.camera.position.copy(this.dampedTarget).add(this.tempVec3);
    this.camera.lookAt(this.dampedTarget);
  }

  /**
   * Set camera mode
   */
  setMode(mode: CameraMode): void {
    const prevMode = this.mode;
    this.mode = mode;

    if (mode === 'orthographic' && prevMode !== 'orthographic') {
      // Switch to orthographic
      const aspect = this.perspectiveCamera.aspect;
      const distance = this.spherical.radius;
      const height = 2 * distance * Math.tan((this.perspectiveCamera.fov * Math.PI) / 360);
      const width = height * aspect;

      this.orthographicCamera.left = -width / 2;
      this.orthographicCamera.right = width / 2;
      this.orthographicCamera.top = height / 2;
      this.orthographicCamera.bottom = -height / 2;
      this.orthographicCamera.position.copy(this.perspectiveCamera.position);
      this.orthographicCamera.quaternion.copy(this.perspectiveCamera.quaternion);
      this.orthographicCamera.updateProjectionMatrix();

      this.camera = this.orthographicCamera;
    } else if (mode !== 'orthographic' && prevMode === 'orthographic') {
      // Switch back to perspective
      this.perspectiveCamera.position.copy(this.orthographicCamera.position);
      this.perspectiveCamera.quaternion.copy(this.orthographicCamera.quaternion);
      this.camera = this.perspectiveCamera;
    }

    if (mode === 'overview') {
      this.transitionTo({
        position: new THREE.Vector3(0, 100, 0),
        target: new THREE.Vector3(0, 0, 0),
        fov: 45,
        zoom: 1,
        quaternion: new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0)),
      }, 1.5);
    }
  }

  /**
   * Get current mode
   */
  getMode(): CameraMode {
    return this.mode;
  }

  /**
   * Set follow target
   */
  setFollowTarget(target: THREE.Object3D, offset?: THREE.Vector3): void {
    this.followTarget = target;
    if (offset) {
      this.followOffset.copy(offset);
    }
    this.setMode('follow');
  }

  /**
   * Clear follow target
   */
  clearFollowTarget(): void {
    this.followTarget = undefined;
    this.setMode('orbit');
  }

  /**
   * Start path animation
   */
  startPath(path: CameraPath): void {
    this.currentPath = path;
    this.pathProgress = 0;
    this.setMode('path');
  }

  /**
   * Stop path animation
   */
  stopPath(): void {
    this.currentPath = undefined;
    this.pathProgress = 0;
    this.setMode('orbit');
  }

  /**
   * Transition to a new camera state
   */
  transitionTo(
    state: Partial<CameraState>,
    duration: number = 1,
    easing: EasingFunction = Easing.easeInOutCubic,
    onComplete?: () => void
  ): void {
    const currentState = this.getState();

    this.transition = {
      from: currentState,
      to: {
        position: state.position ?? currentState.position.clone(),
        target: state.target ?? currentState.target.clone(),
        fov: state.fov ?? currentState.fov,
        zoom: state.zoom ?? currentState.zoom,
        quaternion: state.quaternion ?? currentState.quaternion.clone(),
      },
      duration,
      elapsed: 0,
      easing,
      onComplete,
    };
  }

  /**
   * Look at a point
   */
  lookAt(point: THREE.Vector3, animate: boolean = true): void {
    if (animate) {
      this.transitionTo({ target: point }, 0.5);
    } else {
      this.target.copy(point);
      this.dampedTarget.copy(point);
      this.camera.lookAt(point);
    }
  }

  /**
   * Focus on an object
   */
  focusOn(object: THREE.Object3D, padding: number = 1.5, animate: boolean = true): void {
    const box = new THREE.Box3().setFromObject(object);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    // Calculate distance to fit object in view
    const fov = this.perspectiveCamera.fov * (Math.PI / 180);
    const distance = (maxDim * padding) / (2 * Math.tan(fov / 2));

    // Calculate new position
    const direction = this.tempVec3.copy(this.camera.position).sub(center).normalize();
    const newPosition = center.clone().add(direction.multiplyScalar(distance));

    if (animate) {
      this.transitionTo({
        position: newPosition,
        target: center,
      }, 1);
    } else {
      this.camera.position.copy(newPosition);
      this.target.copy(center);
      this.dampedTarget.copy(center);
      this.camera.lookAt(center);
      this.updateSphericalFromPosition();
      this.dampedSpherical.copy(this.spherical);
    }
  }

  /**
   * Apply camera shake
   */
  shake(intensity: number = 1): void {
    this.shakeIntensity = intensity;
  }

  /**
   * Set constraints
   */
  setConstraints(constraints: CameraConstraints): void {
    this.constraints = constraints;
  }

  /**
   * Clear constraints
   */
  clearConstraints(): void {
    this.constraints = undefined;
  }

  /**
   * Get current camera state
   */
  getState(): CameraState {
    return {
      position: this.camera.position.clone(),
      target: this.target.clone(),
      fov: this.perspectiveCamera.fov,
      zoom: this.camera instanceof THREE.OrthographicCamera ? this.camera.zoom : 1,
      quaternion: this.camera.quaternion.clone(),
    };
  }

  /**
   * Save current state
   */
  saveState(name: string): void {
    this.savedStates.set(name, this.getState());
  }

  /**
   * Restore saved state
   */
  restoreState(name: string, animate: boolean = true): boolean {
    const state = this.savedStates.get(name);
    if (!state) return false;

    if (animate) {
      this.transitionTo(state, 1);
    } else {
      this.camera.position.copy(state.position);
      this.target.copy(state.target);
      this.dampedTarget.copy(state.target);
      this.perspectiveCamera.fov = state.fov;
      this.perspectiveCamera.updateProjectionMatrix();
      this.camera.quaternion.copy(state.quaternion);
      this.updateSphericalFromPosition();
      this.dampedSpherical.copy(this.spherical);
    }

    return true;
  }

  /**
   * Handle pointer down
   */
  onPointerDown(event: PointerEvent): void {
    this.isPointerDown = true;
    this.pointerButton = event.button;
    this.pointerStart.set(event.clientX, event.clientY);
    this.pointerCurrent.copy(this.pointerStart);
  }

  /**
   * Handle pointer move
   */
  onPointerMove(event: PointerEvent): void {
    if (!this.isPointerDown) return;

    const dx = event.clientX - this.pointerCurrent.x;
    const dy = event.clientY - this.pointerCurrent.y;
    this.pointerCurrent.set(event.clientX, event.clientY);

    if (this.pointerButton === 0 && this.config.enableRotate) {
      // Left button - rotate
      this.sphericalDelta.theta -= dx * 0.005 * this.config.rotateSpeed;
      this.sphericalDelta.phi -= dy * 0.005 * this.config.rotateSpeed;
    } else if (this.pointerButton === 2 && this.config.enablePan) {
      // Right button - pan
      this.pan(dx, dy);
    } else if (this.pointerButton === 1 && this.config.enableZoom) {
      // Middle button - dolly
      this.dolly(dy > 0 ? 0.95 : 1.05);
    }
  }

  /**
   * Handle pointer up
   */
  onPointerUp(): void {
    this.isPointerDown = false;
  }

  /**
   * Handle wheel
   */
  onWheel(event: WheelEvent): void {
    if (!this.config.enableZoom) return;
    
    const delta = event.deltaY > 0 ? 0.9 : 1.1;
    this.dolly(delta);
  }

  /**
   * Pan the camera
   */
  private pan(dx: number, dy: number): void {
    const element = document.body;
    const distance = this.spherical.radius;
    const fov = this.perspectiveCamera.fov * (Math.PI / 180);
    const targetDistance = distance * Math.tan(fov / 2);

    // Pan horizontally
    this.tempVec3.setFromMatrixColumn(this.camera.matrix, 0);
    this.tempVec3.multiplyScalar(-2 * dx * targetDistance / element.clientHeight);
    this.panOffset.add(this.tempVec3);

    // Pan vertically
    this.tempVec3.setFromMatrixColumn(this.camera.matrix, 1);
    this.tempVec3.multiplyScalar(2 * dy * targetDistance / element.clientHeight);
    this.panOffset.add(this.tempVec3);
  }

  /**
   * Dolly the camera (zoom)
   */
  private dolly(scale: number): void {
    this.scale *= scale;
  }

  /**
   * Resize camera
   */
  resize(width: number, height: number): void {
    const aspect = width / height;

    this.perspectiveCamera.aspect = aspect;
    this.perspectiveCamera.updateProjectionMatrix();

    // Update orthographic camera
    const frustumSize = 20;
    this.orthographicCamera.left = (-frustumSize * aspect) / 2;
    this.orthographicCamera.right = (frustumSize * aspect) / 2;
    this.orthographicCamera.top = frustumSize / 2;
    this.orthographicCamera.bottom = -frustumSize / 2;
    this.orthographicCamera.updateProjectionMatrix();
  }

  /**
   * Update camera (call every frame)
   */
  update(deltaTime: number): void {
    // Update transition
    if (this.transition) {
      this.updateTransition(deltaTime);
      return;
    }

    // Mode-specific update
    switch (this.mode) {
      case 'follow':
        this.updateFollowMode(deltaTime);
        break;
      case 'path':
        this.updatePathMode(deltaTime);
        break;
      case 'fly':
        // Fly mode is handled by input
        break;
      default:
        this.updateOrbitMode(deltaTime);
        break;
    }

    // Apply shake
    if (this.shakeIntensity > 0.001) {
      this.shakeOffset.set(
        (Math.random() - 0.5) * this.shakeIntensity,
        (Math.random() - 0.5) * this.shakeIntensity,
        (Math.random() - 0.5) * this.shakeIntensity
      );
      this.camera.position.add(this.shakeOffset);
      this.shakeIntensity *= this.shakeDecay;
    }

    // Apply constraints
    this.applyConstraints();
  }

  /**
   * Update orbit mode
   */
  private updateOrbitMode(deltaTime: number): void {
    // Apply rotation delta
    this.spherical.theta += this.sphericalDelta.theta;
    this.spherical.phi += this.sphericalDelta.phi;

    // Clamp phi
    this.spherical.phi = Math.max(
      this.config.minPolarAngle,
      Math.min(this.config.maxPolarAngle, this.spherical.phi)
    );
    this.spherical.makeSafe();

    // Apply scale (zoom)
    this.spherical.radius *= this.scale;
    this.spherical.radius = Math.max(
      this.config.minDistance,
      Math.min(this.config.maxDistance, this.spherical.radius)
    );

    // Apply pan
    this.target.add(this.panOffset);

    // Auto rotate
    if (this.config.autoRotate && !this.isPointerDown) {
      this.spherical.theta += this.config.autoRotateSpeed * deltaTime;
    }

    // Damping
    if (this.config.enableDamping) {
      this.dampedSpherical.theta += (this.spherical.theta - this.dampedSpherical.theta) * this.config.dampingFactor;
      this.dampedSpherical.phi += (this.spherical.phi - this.dampedSpherical.phi) * this.config.dampingFactor;
      this.dampedSpherical.radius += (this.spherical.radius - this.dampedSpherical.radius) * this.config.dampingFactor;
      this.dampedTarget.lerp(this.target, this.config.dampingFactor);
    } else {
      this.dampedSpherical.copy(this.spherical);
      this.dampedTarget.copy(this.target);
    }

    // Update camera position
    this.updatePositionFromSpherical();

    // Reset deltas
    this.sphericalDelta.set(0, 0, 0);
    this.panOffset.set(0, 0, 0);
    this.scale = 1;
  }

  /**
   * Update follow mode
   */
  private updateFollowMode(deltaTime: number): void {
    if (!this.followTarget) return;

    // Get target position
    const targetPos = this.followTarget.position;

    // Calculate desired camera position
    const desiredPosition = targetPos.clone().add(this.followOffset);

    // Smooth follow
    this.camera.position.lerp(desiredPosition, deltaTime * 5);

    // Look at target
    const lookTarget = targetPos.clone();
    if (this.followLookAhead > 0 && this.followTarget instanceof THREE.Object3D) {
      // Look ahead based on velocity (if available)
      const velocity = this.followTarget.userData.velocity as THREE.Vector3 | undefined;
      if (velocity) {
        lookTarget.add(velocity.clone().multiplyScalar(this.followLookAhead));
      }
    }

    this.camera.lookAt(lookTarget);
    this.target.copy(lookTarget);
  }

  /**
   * Update path mode
   */
  private updatePathMode(deltaTime: number): void {
    if (!this.currentPath) return;

    this.pathProgress += deltaTime / this.currentPath.duration;

    if (this.pathProgress >= 1) {
      if (this.currentPath.loop) {
        this.pathProgress = 0;
      } else {
        this.stopPath();
        return;
      }
    }

    // Interpolate position along path
    const points = this.currentPath.points;
    const t = this.pathProgress * (points.length - 1);
    const i = Math.floor(t);
    const f = t - i;

    if (i < points.length - 1) {
      this.camera.position.lerpVectors(points[i]!, points[i + 1]!, f);
    }

    // Interpolate look target
    if (this.currentPath.targets && this.currentPath.targets.length > 0) {
      const targets = this.currentPath.targets;
      const ti = Math.floor(this.pathProgress * (targets.length - 1));
      const tf = (this.pathProgress * (targets.length - 1)) - ti;

      if (ti < targets.length - 1) {
        this.target.lerpVectors(targets[ti]!, targets[ti + 1]!, tf);
      }
    }

    this.camera.lookAt(this.target);
  }

  /**
   * Update transition
   */
  private updateTransition(deltaTime: number): void {
    if (!this.transition) return;

    this.transition.elapsed += deltaTime;
    let t = Math.min(this.transition.elapsed / this.transition.duration, 1);
    t = this.transition.easing(t);

    // Interpolate position
    this.camera.position.lerpVectors(
      this.transition.from.position,
      this.transition.to.position,
      t
    );

    // Interpolate target
    this.target.lerpVectors(
      this.transition.from.target,
      this.transition.to.target,
      t
    );
    this.dampedTarget.copy(this.target);

    // Interpolate FOV
    const fov = THREE.MathUtils.lerp(
      this.transition.from.fov,
      this.transition.to.fov,
      t
    );
    if (this.perspectiveCamera.fov !== fov) {
      this.perspectiveCamera.fov = fov;
      this.perspectiveCamera.updateProjectionMatrix();
    }

    // Look at target
    this.camera.lookAt(this.target);

    // Check completion
    if (t >= 1) {
      this.transition.onComplete?.();
      this.transition = undefined;
      this.updateSphericalFromPosition();
      this.dampedSpherical.copy(this.spherical);
    }
  }

  /**
   * Apply constraints
   */
  private applyConstraints(): void {
    if (!this.constraints) return;

    // Box constraint
    if (this.constraints.boundingBox) {
      this.camera.position.clamp(
        this.constraints.boundingBox.min,
        this.constraints.boundingBox.max
      );
    }

    // Sphere constraint
    if (this.constraints.boundingSphere) {
      const center = this.constraints.boundingSphere.center;
      const radius = this.constraints.boundingSphere.radius;
      const dist = this.camera.position.distanceTo(center);
      
      if (dist > radius) {
        this.camera.position.sub(center).normalize().multiplyScalar(radius).add(center);
      }
    }

    // Min/max position
    if (this.constraints.minPosition) {
      this.camera.position.max(this.constraints.minPosition);
    }
    if (this.constraints.maxPosition) {
      this.camera.position.min(this.constraints.maxPosition);
    }
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.savedStates.clear();
    console.log('[CameraSystem] Disposed');
  }
}
