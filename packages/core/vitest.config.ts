import { defineConfig } from "vitest/config";
import { baseConfig } from "@neurectomy/test-config";

export default defineConfig({
  ...baseConfig,
  test: {
    ...baseConfig.test,
    include: ["src/**/*.test.ts"],
    coverage: {
      ...baseConfig.test?.coverage,
      include: ["src/**/*.ts"],
      exclude: ["src/**/*.test.ts", "src/**/index.ts"],
    },
  },
});
