/**
 * @file React Three Fiber Global Type Declarations
 * @description Extends JSX.IntrinsicElements with Three.js elements for React Three Fiber
 */

import type { Object3DNode, MaterialNode } from "@react-three/fiber";
import type { ThreeElements } from "@react-three/fiber";
import type * as THREE from "three";

declare module "react" {
  namespace JSX {
    // eslint-disable-next-line @typescript-eslint/no-empty-interface
    interface IntrinsicElements extends ThreeElements {}
  }
}

export {};
