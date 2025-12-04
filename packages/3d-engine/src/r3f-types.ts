/**
 * @file React Three Fiber Type Augmentation
 * @description Imports ThreeElements to augment JSX.IntrinsicElements
 * @module @neurectomy/3d-engine
 */

// Import the ThreeElements interface that extends JSX.IntrinsicElements
import type { ThreeElements } from '@react-three/fiber';

// Re-declare global JSX namespace to include Three.js elements
declare global {
  namespace JSX {
    // Extend IntrinsicElements with all Three.js element types
    // This allows using <mesh>, <group>, <boxGeometry>, etc. in JSX
    interface IntrinsicElements extends ThreeElements {}
  }
}

export {};
