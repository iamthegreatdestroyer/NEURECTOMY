import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "blockchain/index": "src/blockchain/index.ts",
    "license/index": "src/license/index.ts",
    "plagiarism/index": "src/plagiarism/index.ts",
  },
  format: ["esm"],
  // TODO: Re-enable DTS after fixing TypeScript errors in plagiarism/ast-comparator.ts
  // The file has many "Object is possibly undefined" errors that need proper null checks
  dts: false,
  clean: true,
  sourcemap: true,
  splitting: false,
  treeshake: true,
  minify: false,
  external: ["@neurectomy/core", "@neurectomy/types"],
});
