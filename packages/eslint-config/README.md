# @neurectomy/eslint-config

> Shared ESLint configurations for the NEURECTOMY monorepo

## Installation

This package is internal to the NEURECTOMY monorepo and is automatically available to all packages.

```bash
pnpm add -D @neurectomy/eslint-config eslint
```

## Available Configurations

### Default / TypeScript (`./` or `./typescript`)

For TypeScript packages with strict type-aware linting.

```js
// .eslintrc.js
module.exports = {
  extends: ["@neurectomy/eslint-config"],
  parserOptions: {
    project: "./tsconfig.json",
  },
};
```

### Base (`./base`)

Foundation configuration without TypeScript-specific rules.

```js
module.exports = {
  extends: ["@neurectomy/eslint-config/base"],
};
```

### React (`./react`)

For React component libraries with JSX, hooks, and accessibility rules.

```js
module.exports = {
  extends: ["@neurectomy/eslint-config/react"],
  parserOptions: {
    project: "./tsconfig.json",
  },
};
```

## Features

All configurations include:

- **Prettier integration** - Disables formatting rules handled by Prettier
- **Import ordering** - Automatic grouping and alphabetization
- **Strict TypeScript** - Type-aware linting rules
- **No console** - Warns on console.log (allows warn/error)

### TypeScript Features

- Consistent type imports (`import type`)
- Interface preference over type aliases
- No floating promises
- Optional chaining preference
- Nullish coalescing preference

### React Features

- Hooks rules enforcement
- Accessibility (a11y) rules
- JSX best practices
- Self-closing components

## Usage

### Basic TypeScript Package

```js
// packages/my-package/.eslintrc.js
module.exports = {
  root: true,
  extends: ["@neurectomy/eslint-config"],
  parserOptions: {
    project: "./tsconfig.json",
    tsconfigRootDir: __dirname,
  },
};
```

### React Component Package

```js
// packages/ui/.eslintrc.js
module.exports = {
  root: true,
  extends: ["@neurectomy/eslint-config/react"],
  parserOptions: {
    project: "./tsconfig.json",
    tsconfigRootDir: __dirname,
  },
};
```

## Configuration Comparison

| Feature              | base | typescript | react |
| -------------------- | ---- | ---------- | ----- |
| ES2022 support       | ✅   | ✅         | ✅    |
| Prettier integration | ✅   | ✅         | ✅    |
| Import ordering      | ❌   | ✅         | ✅    |
| Type-aware rules     | ❌   | ✅         | ✅    |
| React rules          | ❌   | ❌         | ✅    |
| Hooks rules          | ❌   | ❌         | ✅    |
| Accessibility rules  | ❌   | ❌         | ✅    |

## Overriding Rules

You can override any rule in your local `.eslintrc.js`:

```js
module.exports = {
  extends: ["@neurectomy/eslint-config"],
  rules: {
    "@typescript-eslint/no-explicit-any": "off", // Allow any in this package
  },
};
```

## License

Proprietary - NEURECTOMY Project
