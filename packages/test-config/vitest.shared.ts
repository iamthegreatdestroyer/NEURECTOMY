/**
 * @neurectomy/test-config - Shared Vitest Configuration
 *
 * This module provides base configurations for vitest that can be
 * extended by individual packages in the monorepo.
 *
 * @example
 * // In your package's vitest.config.ts:
 * import { defineConfig, mergeConfig } from 'vitest/config';
 * import { baseConfig, reactConfig } from '@neurectomy/test-config/vitest';
 *
 * export default mergeConfig(reactConfig, defineConfig({
 *   test: {
 *     // package-specific overrides
 *   }
 * }));
 */

import { defineConfig } from "vitest/config";
import type { UserConfig } from "vitest/config";

/**
 * Base test configuration for Node.js packages
 * Use for non-React packages like core, types, api-client (server-side)
 */
export const baseConfig: UserConfig = defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: [
      "src/**/*.test.ts",
      "src/**/*.spec.ts",
      "tests/**/*.test.ts",
      "tests/**/*.spec.ts",
    ],
    exclude: ["node_modules", "dist", ".turbo"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html", "lcov"],
      reportsDirectory: "./coverage",
      exclude: [
        "node_modules/**",
        "dist/**",
        "**/*.d.ts",
        "**/*.test.ts",
        "**/*.spec.ts",
        "**/index.ts",
        "**/types.ts",
        "coverage/**",
        "vitest.config.ts",
      ],
      thresholds: {
        global: {
          branches: 70,
          functions: 70,
          lines: 70,
          statements: 70,
        },
      },
    },
    reporters: ["default", "hanging-process"],
    testTimeout: 10000,
    hookTimeout: 10000,
    teardownTimeout: 5000,
    retry: 0,
    pool: "forks",
    poolOptions: {
      forks: {
        singleFork: false,
      },
    },
    sequence: {
      shuffle: true,
    },
    typecheck: {
      enabled: false,
      checker: "tsc",
      include: ["**/*.{test,spec}-d.ts"],
    },
  },
  esbuild: {
    target: "node20",
  },
});

/**
 * React/DOM test configuration
 * Use for React component packages like ui, 3d-engine
 */
export const reactConfig: UserConfig = defineConfig({
  test: {
    ...baseConfig.test,
    environment: "jsdom",
    include: [
      "src/**/*.test.ts",
      "src/**/*.test.tsx",
      "src/**/*.spec.ts",
      "src/**/*.spec.tsx",
      "tests/**/*.test.ts",
      "tests/**/*.test.tsx",
    ],
    setupFiles: ["@neurectomy/test-config/setup-dom"],
    deps: {
      optimizer: {
        web: {
          include: ["@testing-library/react", "@testing-library/user-event"],
        },
      },
    },
    css: {
      modules: {
        classNameStrategy: "non-scoped",
      },
    },
    coverage: {
      ...baseConfig.test?.coverage,
      exclude: [
        ...((baseConfig.test?.coverage?.exclude as string[]) || []),
        "**/*.stories.tsx",
        "**/*.stories.ts",
      ],
    },
  },
});

/**
 * Integration test configuration
 * Use for cross-package and API integration tests
 */
export const integrationConfig: UserConfig = defineConfig({
  test: {
    ...baseConfig.test,
    include: ["tests/integration/**/*.test.ts", "tests/e2e/**/*.test.ts"],
    testTimeout: 30000,
    hookTimeout: 30000,
    maxConcurrency: 1,
    sequence: {
      shuffle: false,
    },
    retry: 1,
    coverage: {
      ...baseConfig.test?.coverage,
      thresholds: {
        global: {
          branches: 50,
          functions: 50,
          lines: 50,
          statements: 50,
        },
      },
    },
  },
});

/**
 * Benchmark configuration
 * Use for performance testing
 */
export const benchmarkConfig: UserConfig = defineConfig({
  test: {
    ...baseConfig.test,
    include: ["src/**/*.bench.ts", "tests/bench/**/*.ts"],
    benchmark: {
      include: ["**/*.bench.ts"],
      reporters: ["default", "json"],
      outputFile: "./benchmarks/results.json",
    },
  },
});

// Re-export for convenience
export { defineConfig, mergeConfig } from "vitest/config";
