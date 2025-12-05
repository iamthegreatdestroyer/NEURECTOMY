import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  splitting: false,
  minify: false,
  external: [
    "react",
    "react-dom",
    "three",
    "@react-three/fiber",
    "@react-three/drei",
  ],
  esbuildOptions(options) {
    options.banner = {
      js: "/* @neurectomy/ui - React Component Library */",
    };
  },
});
