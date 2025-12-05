import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "repository/index": "src/repository/index.ts",
    "branch/index": "src/branch/index.ts",
    "pr/index": "src/pr/index.ts",
    "issues/index": "src/issues/index.ts",
    "actions/index": "src/actions/index.ts",
    "webhooks/index": "src/webhooks/index.ts",
    "agents/index": "src/agents/index.ts",
  },
  format: ["cjs", "esm"],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  treeshake: true,
  external: ["@neurectomy/core", "@neurectomy/types"],
});
