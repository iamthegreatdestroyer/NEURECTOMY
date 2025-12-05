# @neurectomy/typescript-config

> Shared TypeScript configurations for the NEURECTOMY monorepo

## Installation

This package is internal to the NEURECTOMY monorepo and is automatically available to all packages.

## Available Configurations

### Base (`./base`)

Foundation configuration with strict settings for all packages.

```json
{
  "extends": "@neurectomy/typescript-config/base"
}
```

### Library (`./library`)

For publishable library packages that output to `dist/`.

```json
{
  "extends": "@neurectomy/typescript-config/library"
}
```

### React Library (`./react-library`)

For React component libraries with JSX support.

```json
{
  "extends": "@neurectomy/typescript-config/react-library"
}
```

### Node (`./node`)

For Node.js services and CLI tools.

```json
{
  "extends": "@neurectomy/typescript-config/node"
}
```

## Features

All configurations include:

- **ES2022** target for modern JavaScript features
- **Strict mode** for maximum type safety
- **Bundler module resolution** for modern build tools
- **Source maps** and **declaration maps** for debugging
- **Verbatim module syntax** for explicit imports

## Usage

In your package's `tsconfig.json`:

```json
{
  "extends": "@neurectomy/typescript-config/library",
  "compilerOptions": {
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

## Configuration Details

| Config        | Target | Module   | JSX       | Use Case         |
| ------------- | ------ | -------- | --------- | ---------------- |
| base          | ES2022 | ESNext   | -         | Foundation       |
| library       | ES2022 | ESNext   | -         | TS libraries     |
| react-library | ES2022 | ESNext   | react-jsx | React components |
| node          | ES2022 | NodeNext | -         | Node.js services |

## License

Proprietary - NEURECTOMY Project
