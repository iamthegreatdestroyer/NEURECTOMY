import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "profiler/index": "src/profiler/index.ts",
    "optimizer/index": "src/optimizer/index.ts",
    "memory/index": "src/memory/index.ts",
    "cache/index": "src/cache/index.ts",
  },
  format: ["esm"],
  dts: true,
  splitting: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  minify: false,
  external: ["@neurectomy/core", "@neurectomy/types"],
});
