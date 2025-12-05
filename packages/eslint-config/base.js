/** @type {import('eslint').Linter.Config} */
module.exports = {
  extends: ["eslint:recommended", "prettier"],
  env: {
    es2022: true,
    node: true,
  },
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
  },
  rules: {
    "no-console": ["warn", { allow: ["warn", "error"] }],
    "no-unused-vars": "off", // Handled by TypeScript
    "prefer-const": "error",
    "no-var": "error",
    "object-shorthand": "error",
    "prefer-arrow-callback": "error",
    "prefer-template": "error",
    "no-nested-ternary": "warn",
    "no-unneeded-ternary": "error",
    eqeqeq: ["error", "always", { null: "ignore" }],
    curly: ["error", "all"],
  },
  ignorePatterns: [
    "node_modules/",
    "dist/",
    "build/",
    "coverage/",
    "*.min.js",
    "*.d.ts",
  ],
};
