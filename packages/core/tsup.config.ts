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
  external: ["@neurectomy/types"],
  esbuildOptions(options) {
    options.banner = {
      js: "/* @neurectomy/core - Core Utilities and Business Logic */",
    };
  },
});
