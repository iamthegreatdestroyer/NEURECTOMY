import { defineConfig } from "tsup";

export default defineConfig({
  entry: [
    "src/index.ts",
    "src/docker/index.ts",
    "src/kubernetes/index.ts",
    "src/firecracker/index.ts",
    "src/wasm/index.ts",
  ],
  format: ["esm", "cjs"],
  dts: true,
  splitting: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  external: ["dockerode", "@kubernetes/client-node"],
});
