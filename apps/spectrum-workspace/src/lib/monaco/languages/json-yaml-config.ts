/**
 * JSON and YAML Language Configuration for Monaco Editor
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import type * as Monaco from "monaco-editor";

/**
 * Package.json schema for validation
 */
const PACKAGE_JSON_SCHEMA = {
  uri: "https://json.schemastore.org/package.json",
  fileMatch: ["**/package.json"],
  schema: {
    type: "object",
    properties: {
      name: { type: "string", description: "Package name" },
      version: { type: "string", description: "Package version (semver)" },
      description: { type: "string", description: "Package description" },
      main: { type: "string", description: "Entry point for CommonJS" },
      module: { type: "string", description: "Entry point for ES modules" },
      types: { type: "string", description: "TypeScript type definitions" },
      scripts: { type: "object", description: "NPM scripts" },
      dependencies: { type: "object", description: "Production dependencies" },
      devDependencies: {
        type: "object",
        description: "Development dependencies",
      },
      peerDependencies: { type: "object", description: "Peer dependencies" },
    },
  },
};

/**
 * TSConfig schema for validation
 */
const TSCONFIG_SCHEMA = {
  uri: "https://json.schemastore.org/tsconfig.json",
  fileMatch: ["**/tsconfig.json", "**/tsconfig.*.json"],
  schema: {
    type: "object",
    properties: {
      compilerOptions: {
        type: "object",
        description: "TypeScript compiler options",
        properties: {
          target: {
            type: "string",
            enum: [
              "ES5",
              "ES6",
              "ES2015",
              "ES2016",
              "ES2017",
              "ES2018",
              "ES2019",
              "ES2020",
              "ES2021",
              "ES2022",
              "ESNext",
            ],
          },
          module: {
            type: "string",
            enum: [
              "CommonJS",
              "AMD",
              "UMD",
              "System",
              "ES6",
              "ES2015",
              "ES2020",
              "ESNext",
              "Node16",
              "NodeNext",
            ],
          },
          strict: { type: "boolean" },
          esModuleInterop: { type: "boolean" },
          skipLibCheck: { type: "boolean" },
        },
      },
      include: { type: "array", items: { type: "string" } },
      exclude: { type: "array", items: { type: "string" } },
      extends: { type: "string" },
    },
  },
};

/**
 * NEURECTOMY agent configuration schema
 */
const AGENT_CONFIG_SCHEMA = {
  uri: "internal://neurectomy/agent-config.json",
  fileMatch: [
    "**/agent.json",
    "**/agent.config.json",
    "**/.neurectomy/agents/*.json",
  ],
  schema: {
    type: "object",
    required: ["name", "codename", "tier"],
    properties: {
      name: { type: "string", description: "Agent display name" },
      codename: { type: "string", description: "Agent codename (uppercase)" },
      tier: {
        type: "string",
        enum: [
          "foundational",
          "specialist",
          "innovator",
          "meta",
          "domain",
          "emerging",
          "human-centric",
          "enterprise",
        ],
        description: "Agent tier classification",
      },
      description: { type: "string", description: "Agent description" },
      philosophy: { type: "string", description: "Agent philosophy quote" },
      capabilities: {
        type: "array",
        items: { type: "string" },
        description: "List of agent capabilities",
      },
      version: { type: "string", description: "Agent version (semver)" },
    },
  },
};

/**
 * Configure JSON language settings
 */
export function configureJson(monaco: typeof Monaco): void {
  monaco.languages.json.jsonDefaults.setDiagnosticsOptions({
    validate: true,
    schemaValidation: "error",
    schemaRequest: "warning",
    trailingCommas: "warning",
    comments: "warning",
    allowComments: true,
    schemas: [PACKAGE_JSON_SCHEMA, TSCONFIG_SCHEMA, AGENT_CONFIG_SCHEMA],
  });
}

/**
 * YAML language configuration
 * Note: Monaco doesn't have built-in YAML support,
 * so we register basic tokenization
 */
export function configureYaml(monaco: typeof Monaco): Monaco.IDisposable {
  // Register YAML language if not already registered
  const languages = monaco.languages.getLanguages();
  const hasYaml = languages.some((lang) => lang.id === "yaml");

  if (!hasYaml) {
    monaco.languages.register({
      id: "yaml",
      extensions: [".yaml", ".yml"],
      aliases: ["YAML", "yaml"],
      mimetypes: ["application/x-yaml", "text/yaml"],
    });
  }

  // Set YAML tokenization rules
  return monaco.languages.setMonarchTokensProvider("yaml", {
    tokenizer: {
      root: [
        // Comments
        [/#.*$/, "comment"],

        // Keys
        [/^[\w][\w-]*(?=\s*:)/, "type.identifier"],
        [/^\s+[\w][\w-]*(?=\s*:)/, "type.identifier"],

        // Strings
        [/"([^"\\]|\\.)*$/, "string.invalid"],
        [/'([^'\\]|\\.)*$/, "string.invalid"],
        [/"/, "string", "@string_double"],
        [/'/, "string", "@string_single"],

        // Numbers
        [/-?\d+\.?\d*(e[+-]?\d+)?/, "number"],

        // Booleans
        [/\b(true|false|yes|no|on|off)\b/, "keyword"],

        // Null
        [/\bnull\b/, "keyword"],

        // Special keys
        [/:/, "delimiter"],
        [/-\s/, "delimiter"],
        [/\|/, "delimiter"],
        [/>/, "delimiter"],

        // Anchors and aliases
        [/&\w+/, "tag"],
        [/\*\w+/, "tag"],

        // Tags
        [/!\w+/, "tag"],
      ],
      string_double: [
        [/[^\\"]+/, "string"],
        [/\\./, "string.escape"],
        [/"/, "string", "@pop"],
      ],
      string_single: [
        [/[^\\']+/, "string"],
        [/\\./, "string.escape"],
        [/'/, "string", "@pop"],
      ],
    },
  });
}

/**
 * Configure all JSON/YAML languages
 */
export function configureJsonYaml(monaco: typeof Monaco): Monaco.IDisposable {
  configureJson(monaco);
  return configureYaml(monaco);
}
