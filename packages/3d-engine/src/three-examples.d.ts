/**
 * Type declarations for Three.js examples modules
 *
 * These modules are part of three/examples/jsm but don't have built-in
 * type declarations in all versions, so we provide them here.
 *
 * @module @neurectomy/3d-engine/three-examples
 */

declare module "three/examples/jsm/renderers/CSS2DRenderer" {
  import {
    Object3D,
    Camera,
    Scene,
    WebGLRenderer,
    BufferGeometry,
    Material,
    Group,
    Object3DEventMap,
  } from "three";

  export class CSS2DObject extends Object3D<Object3DEventMap> {
    constructor(element: HTMLElement);
    element: HTMLElement;
    center: { x: number; y: number };
    isCSS2DObject: true;
  }

  export class CSS2DRenderer {
    constructor(parameters?: { element?: HTMLElement });
    domElement: HTMLElement;
    getSize(): { width: number; height: number };
    setSize(width: number, height: number): void;
    render(scene: Scene, camera: Camera): void;
  }
}

declare module "three/examples/jsm/controls/TransformControls" {
  import {
    Object3D,
    Camera,
    Raycaster,
    Intersection,
    EventDispatcher,
    Event,
    Matrix4,
    Quaternion,
    Vector3,
  } from "three";

  export interface TransformControlsEventMap {
    change: Event;
    mouseDown: Event;
    mouseUp: Event;
    objectChange: Event;
    "dragging-changed": { value: boolean };
  }

  export class TransformControls extends Object3D {
    constructor(camera: Camera, domElement?: HTMLElement);

    domElement: HTMLElement;
    camera: Camera;
    object: Object3D | undefined;
    enabled: boolean;
    axis: "X" | "Y" | "Z" | "XY" | "YZ" | "XZ" | "XYZ" | null;
    mode: "translate" | "rotate" | "scale";
    translationSnap: number | null;
    rotationSnap: number | null;
    scaleSnap: number | null;
    space: "world" | "local";
    size: number;
    dragging: boolean;
    showX: boolean;
    showY: boolean;
    showZ: boolean;

    attach(object: Object3D): this;
    detach(): this;
    getMode(): "translate" | "rotate" | "scale";
    setMode(mode: "translate" | "rotate" | "scale"): void;
    setTranslationSnap(translationSnap: number | null): void;
    setRotationSnap(rotationSnap: number | null): void;
    setScaleSnap(scaleSnap: number | null): void;
    setSize(size: number): void;
    setSpace(space: "world" | "local"): void;
    dispose(): void;
    getRaycaster(): Raycaster;
    updateMatrixWorld(): void;

    addEventListener<T extends keyof TransformControlsEventMap>(
      type: T,
      listener: (event: TransformControlsEventMap[T]) => void
    ): void;
    removeEventListener<T extends keyof TransformControlsEventMap>(
      type: T,
      listener: (event: TransformControlsEventMap[T]) => void
    ): void;
  }
}

declare module "three/examples/jsm/controls/OrbitControls" {
  import { Camera, EventDispatcher, Vector3 } from "three";

  export interface OrbitControlsEventMap {
    change: { type: "change" };
    start: { type: "start" };
    end: { type: "end" };
  }

  export class OrbitControls {
    constructor(camera: Camera, domElement?: HTMLElement);

    domElement: HTMLElement;
    object: Camera;
    enabled: boolean;
    target: Vector3;
    minDistance: number;
    maxDistance: number;
    minZoom: number;
    maxZoom: number;
    minPolarAngle: number;
    maxPolarAngle: number;
    minAzimuthAngle: number;
    maxAzimuthAngle: number;
    enableDamping: boolean;
    dampingFactor: number;
    enableZoom: boolean;
    zoomSpeed: number;
    enableRotate: boolean;
    rotateSpeed: number;
    enablePan: boolean;
    panSpeed: number;
    screenSpacePanning: boolean;
    keyPanSpeed: number;
    autoRotate: boolean;
    autoRotateSpeed: number;
    keys: { LEFT: string; UP: string; RIGHT: string; BOTTOM: string };
    mouseButtons: { LEFT: number; MIDDLE: number; RIGHT: number };
    touches: { ONE: number; TWO: number };

    update(): boolean;
    saveState(): void;
    reset(): void;
    dispose(): void;
    getPolarAngle(): number;
    getAzimuthalAngle(): number;
    getDistance(): number;
    listenToKeyEvents(domElement: HTMLElement): void;
    stopListenToKeyEvents(): void;

    addEventListener<T extends keyof OrbitControlsEventMap>(
      type: T,
      listener: (event: OrbitControlsEventMap[T]) => void
    ): void;
    removeEventListener<T extends keyof OrbitControlsEventMap>(
      type: T,
      listener: (event: OrbitControlsEventMap[T]) => void
    ): void;
  }
}

declare module "three/examples/jsm/controls/FlyControls" {
  import { Camera, EventDispatcher, Vector3 } from "three";

  export class FlyControls {
    constructor(camera: Camera, domElement?: HTMLElement);

    domElement: HTMLElement;
    object: Camera;
    movementSpeed: number;
    rollSpeed: number;
    dragToLook: boolean;
    autoForward: boolean;

    update(delta: number): void;
    dispose(): void;
  }
}

declare module "three/examples/jsm/controls/TrackballControls" {
  import { Camera, EventDispatcher, Vector3 } from "three";

  export class TrackballControls {
    constructor(camera: Camera, domElement?: HTMLElement);

    domElement: HTMLElement;
    object: Camera;
    enabled: boolean;
    target: Vector3;
    rotateSpeed: number;
    zoomSpeed: number;
    panSpeed: number;
    noRotate: boolean;
    noZoom: boolean;
    noPan: boolean;
    noRoll: boolean;
    staticMoving: boolean;
    dynamicDampingFactor: number;
    minDistance: number;
    maxDistance: number;
    keys: number[];

    update(): void;
    reset(): void;
    dispose(): void;
    checkDistances(): void;
    handleResize(): void;
  }
}

declare module "three/examples/jsm/postprocessing/EffectComposer" {
  import { WebGLRenderer, WebGLRenderTarget } from "three";

  export class EffectComposer {
    constructor(renderer: WebGLRenderer, renderTarget?: WebGLRenderTarget);

    renderer: WebGLRenderer;
    renderTarget1: WebGLRenderTarget;
    renderTarget2: WebGLRenderTarget;
    writeBuffer: WebGLRenderTarget;
    readBuffer: WebGLRenderTarget;
    passes: Pass[];
    copyPass: ShaderPass;
    clock: Clock;

    addPass(pass: Pass): void;
    insertPass(pass: Pass, index: number): void;
    removePass(pass: Pass): void;
    isLastEnabledPass(passIndex: number): boolean;
    render(deltaTime?: number): void;
    reset(renderTarget?: WebGLRenderTarget): void;
    setSize(width: number, height: number): void;
    setPixelRatio(pixelRatio: number): void;
    dispose(): void;
  }

  export class Pass {
    enabled: boolean;
    needsSwap: boolean;
    clear: boolean;
    renderToScreen: boolean;

    setSize(width: number, height: number): void;
    render(
      renderer: WebGLRenderer,
      writeBuffer: WebGLRenderTarget,
      readBuffer: WebGLRenderTarget,
      deltaTime: number,
      maskActive: boolean
    ): void;
    dispose(): void;
  }

  export class ShaderPass extends Pass {
    constructor(shader: object, textureID?: string);
    textureID: string;
    uniforms: { [uniform: string]: { value: unknown } };
    material: ShaderMaterial;
    fsQuad: FullScreenQuad;
  }

  class Clock {
    constructor(autoStart?: boolean);
    autoStart: boolean;
    startTime: number;
    oldTime: number;
    elapsedTime: number;
    running: boolean;
    start(): void;
    stop(): void;
    getElapsedTime(): number;
    getDelta(): number;
  }

  class FullScreenQuad {
    constructor(material?: Material);
    render(renderer: WebGLRenderer): void;
    dispose(): void;
    material: Material;
  }

  import { Material, ShaderMaterial } from "three";
}

declare module "three/examples/jsm/postprocessing/RenderPass" {
  import { Scene, Camera, Color, Material, WebGLRenderTarget } from "three";
  import { Pass } from "three/examples/jsm/postprocessing/EffectComposer";

  export class RenderPass extends Pass {
    constructor(
      scene: Scene,
      camera: Camera,
      overrideMaterial?: Material,
      clearColor?: Color,
      clearAlpha?: number
    );

    scene: Scene;
    camera: Camera;
    overrideMaterial: Material | null;
    clearColor: Color | null;
    clearAlpha: number | null;
    clearDepth: boolean;
  }
}

declare module "three/examples/jsm/postprocessing/UnrealBloomPass" {
  import { Vector2, Color, WebGLRenderTarget } from "three";
  import { Pass } from "three/examples/jsm/postprocessing/EffectComposer";

  export class UnrealBloomPass extends Pass {
    constructor(
      resolution: Vector2,
      strength: number,
      radius: number,
      threshold: number
    );

    resolution: Vector2;
    strength: number;
    radius: number;
    threshold: number;
    clearColor: Color;
    renderTargetsHorizontal: WebGLRenderTarget[];
    renderTargetsVertical: WebGLRenderTarget[];
    nMips: number;
    renderTargetBright: WebGLRenderTarget;
  }
}

declare module "three/examples/jsm/postprocessing/SSAOPass" {
  import { Scene, Camera, Vector2, Color } from "three";
  import { Pass } from "three/examples/jsm/postprocessing/EffectComposer";

  export class SSAOPass extends Pass {
    constructor(scene: Scene, camera: Camera, width?: number, height?: number);

    scene: Scene;
    camera: Camera;
    width: number;
    height: number;
    kernelRadius: number;
    kernelSize: number;
    kernel: Vector3[];
    noiseTexture: Texture;
    output: number;
    minDistance: number;
    maxDistance: number;

    static OUTPUT: {
      Default: number;
      SSAO: number;
      Blur: number;
      Beauty: number;
      Depth: number;
      Normal: number;
    };
  }

  import { Vector3, Texture } from "three";
}

declare module "three/examples/jsm/loaders/GLTFLoader" {
  import {
    Loader,
    LoadingManager,
    Group,
    AnimationClip,
    Camera,
    Scene,
  } from "three";

  export interface GLTF {
    scene: Group;
    scenes: Group[];
    cameras: Camera[];
    animations: AnimationClip[];
    asset: {
      copyright?: string;
      generator?: string;
      version: string;
      minVersion?: string;
      extensions?: object;
      extras?: unknown;
    };
    parser: GLTFParser;
    userData: object;
  }

  export class GLTFLoader extends Loader {
    constructor(manager?: LoadingManager);

    load(
      url: string,
      onLoad: (gltf: GLTF) => void,
      onProgress?: (event: ProgressEvent) => void,
      onError?: (event: ErrorEvent) => void
    ): void;

    loadAsync(
      url: string,
      onProgress?: (event: ProgressEvent) => void
    ): Promise<GLTF>;

    setDRACOLoader(dracoLoader: object): GLTFLoader;
    setKTX2Loader(ktx2Loader: object): GLTFLoader;
    setMeshoptDecoder(decoder: typeof MeshoptDecoder): GLTFLoader;

    register(callback: (parser: GLTFParser) => object): GLTFLoader;
    unregister(callback: (parser: GLTFParser) => object): GLTFLoader;

    parse(
      data: ArrayBuffer | string,
      path: string,
      onLoad: (gltf: GLTF) => void,
      onError?: (event: ErrorEvent) => void
    ): void;

    parseAsync(data: ArrayBuffer | string, path: string): Promise<GLTF>;
  }

  export class GLTFParser {
    json: object;
    extensions: { [key: string]: object };
    plugins: { [key: string]: object };
    options: object;
    cache: GLTFRegistry;
    associations: Map<object, { type: string; index: number }>;
    primitiveCache: { [key: string]: Promise<Group> };
    meshCache: {
      refs: { [key: number]: number };
      uses: { [key: number]: number };
    };
    cameraCache: {
      refs: { [key: number]: number };
      uses: { [key: number]: number };
    };
    lightCache: {
      refs: { [key: number]: number };
      uses: { [key: number]: number };
    };
    textureCache: { [key: string]: Promise<Texture> };
    sourceCache: { [key: string]: Promise<Source> };
    nodeNamesUsed: { [key: string]: number };

    setExtensions(extensions: { [key: string]: object }): void;
    setPlugins(plugins: { [key: string]: object }): void;
    parse(
      onLoad: (gltf: GLTF) => void,
      onError?: (error: ErrorEvent) => void
    ): void;
  }

  export class GLTFRegistry {
    objects: { [key: string]: unknown };
    add(key: string, value: unknown): void;
    get(key: string): unknown;
    remove(key: string): void;
    removeAll(): void;
  }

  class MeshoptDecoder {
    static ready: Promise<void>;
  }

  import { Texture } from "three";
  class Source {}
}
