import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "webgpu/index": "src/webgpu/index.ts",
    "physics/index": "src/physics/index.ts",
  },
  format: ["esm"],
  dts: true,
  sourcemap: true,
  clean: true,
  external: [
    "react",
    "@react-three/fiber",
    "@react-three/drei",
    "@react-three/postprocessing",
    "three",
    "zustand",
    "immer",
    "@dimforge/rapier3d-compat",
  ],
  treeshake: true,
  minify: false,
  splitting: true,
});
