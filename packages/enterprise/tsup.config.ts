import { defineConfig } from "tsup";

export default defineConfig({
  entry: [
    "src/index.ts",
    "src/auth/index.ts",
    "src/tenancy/index.ts",
    "src/compliance/index.ts",
    "src/audit/index.ts",
  ],
  format: ["esm"],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  treeshake: true,
  external: [
    "@neurectomy/core",
    "@neurectomy/types",
    "@neurectomy/legal-fortress",
  ],
});
