import { defineConfig } from "vitest/config";
import { reactConfig } from "@neurectomy/test-config";

export default defineConfig({
  ...reactConfig,
  test: {
    ...reactConfig.test,
    include: ["src/**/*.test.{ts,tsx}"],
    coverage: {
      ...reactConfig.test?.coverage,
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/**/*.test.{ts,tsx}", "src/**/index.ts"],
    },
  },
});
