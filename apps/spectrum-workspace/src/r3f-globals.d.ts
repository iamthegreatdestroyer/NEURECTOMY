/**
 * @file React Three Fiber Global Type Declarations
 * @description Extends JSX.IntrinsicElements with Three.js elements for React Three Fiber
 */

import type { ThreeElements } from "@react-three/fiber";

declare module "react" {
  namespace JSX {
    // eslint-disable-next-line @typescript-eslint/no-empty-interface
    interface IntrinsicElements extends ThreeElements {}
  }
}

export {};
