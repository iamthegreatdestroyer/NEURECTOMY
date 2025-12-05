import { defineConfig } from "vitest/config";
import { integrationConfig } from "@neurectomy/test-config";

export default defineConfig({
  ...integrationConfig,
  test: {
    ...integrationConfig.test,
    include: ["src/**/*.test.ts"],
  },
});
