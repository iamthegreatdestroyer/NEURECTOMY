import { defineConfig } from "tsup";

export default defineConfig({
  entry: [
    "src/index.ts",
    "src/strategies/index.ts",
    "src/kubernetes/index.ts",
    "src/gitops/index.ts",
    "src/approval/index.ts",
    "src/rollback/index.ts",
    "src/health/index.ts",
  ],
  format: ["esm"],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  treeshake: true,
  external: ["@kubernetes/client-node", "@neurectomy/github-universe"],
});
