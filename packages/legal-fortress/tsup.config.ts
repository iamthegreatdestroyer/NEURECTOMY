import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "blockchain/index": "src/blockchain/index.ts",
    "license/index": "src/license/index.ts",
    "plagiarism/index": "src/plagiarism/index.ts",
  },
  format: ["esm"],
  dts: true,
  clean: true,
  sourcemap: true,
  splitting: false,
  treeshake: true,
  minify: false,
  external: ["@neurectomy/core", "@neurectomy/types"],
});
