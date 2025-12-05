import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    graphql: "src/graphql/index.ts",
    rest: "src/rest/index.ts",
  },
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  splitting: false,
  minify: false,
  external: [
    "@neurectomy/types",
    "@tanstack/react-query",
    "react",
    "graphql",
    "graphql-request",
    "graphql-ws",
  ],
  esbuildOptions(options) {
    options.banner = {
      js: "/* @neurectomy/api-client - GraphQL & REST Client Library */",
    };
  },
});
