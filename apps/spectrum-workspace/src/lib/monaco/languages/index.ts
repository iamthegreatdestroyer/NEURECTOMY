/**
 * Language Configuration Index for Monaco Editor
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import type * as Monaco from "monaco-editor";
import {
  configureTypeScript,
  registerTypeScriptCodeActions,
} from "./typescript-config";
import { configureJsonYaml } from "./json-yaml-config";

/**
 * Language to file extension mapping
 */
export const LANGUAGE_EXTENSIONS: Record<string, string[]> = {
  typescript: [".ts", ".tsx", ".mts", ".cts"],
  javascript: [".js", ".jsx", ".mjs", ".cjs"],
  python: [".py", ".pyw", ".pyi"],
  rust: [".rs"],
  go: [".go"],
  json: [".json", ".jsonc"],
  yaml: [".yaml", ".yml"],
  markdown: [".md", ".mdx"],
  html: [".html", ".htm"],
  css: [".css"],
  scss: [".scss", ".sass"],
  less: [".less"],
  sql: [".sql"],
  graphql: [".graphql", ".gql"],
  dockerfile: ["Dockerfile", ".dockerfile"],
  shell: [".sh", ".bash", ".zsh"],
  powershell: [".ps1", ".psm1", ".psd1"],
  toml: [".toml"],
  xml: [".xml", ".xsd", ".xsl"],
  plaintext: [".txt", ".text"],
};

/**
 * Language icons mapping
 */
export const LANGUAGE_ICONS: Record<string, string> = {
  typescript: "typescript",
  javascript: "javascript",
  typescriptreact: "react",
  javascriptreact: "react",
  python: "python",
  rust: "rust",
  go: "go",
  json: "json",
  yaml: "yaml",
  markdown: "markdown",
  html: "html",
  css: "css",
  scss: "scss",
  less: "less",
  sql: "database",
  graphql: "graphql",
  dockerfile: "docker",
  shell: "terminal",
  powershell: "terminal",
  toml: "settings",
  xml: "xml",
  plaintext: "file",
};

/**
 * Language display names
 */
export const LANGUAGE_NAMES: Record<string, string> = {
  typescript: "TypeScript",
  javascript: "JavaScript",
  typescriptreact: "TypeScript React",
  javascriptreact: "JavaScript React",
  python: "Python",
  rust: "Rust",
  go: "Go",
  json: "JSON",
  yaml: "YAML",
  markdown: "Markdown",
  html: "HTML",
  css: "CSS",
  scss: "SCSS",
  less: "Less",
  sql: "SQL",
  graphql: "GraphQL",
  dockerfile: "Dockerfile",
  shell: "Shell",
  powershell: "PowerShell",
  toml: "TOML",
  xml: "XML",
  plaintext: "Plain Text",
};

/**
 * Detect language from file path
 */
export function getLanguageFromPath(filePath: string): string {
  const fileName =
    filePath.split("/").pop() || filePath.split("\\").pop() || "";
  const ext = fileName.includes(".")
    ? `.${fileName.split(".").pop()?.toLowerCase()}`
    : "";

  // Check for exact filename matches first
  if (fileName === "Dockerfile" || fileName.startsWith("Dockerfile.")) {
    return "dockerfile";
  }
  if (fileName === "Makefile") {
    return "makefile";
  }

  // Check React extensions
  if (ext === ".tsx") return "typescriptreact";
  if (ext === ".jsx") return "javascriptreact";

  // Check by extension
  for (const [language, extensions] of Object.entries(LANGUAGE_EXTENSIONS)) {
    if (extensions.includes(ext) || extensions.includes(fileName)) {
      return language;
    }
  }

  return "plaintext";
}

/**
 * Get icon for language
 */
export function getLanguageIcon(language: string): string {
  return LANGUAGE_ICONS[language] || "file";
}

/**
 * Get display name for language
 */
export function getLanguageName(language: string): string {
  return LANGUAGE_NAMES[language] || language;
}

/**
 * Configure all languages
 */
export function configureAllLanguages(
  monaco: typeof Monaco
): Monaco.IDisposable[] {
  const disposables: Monaco.IDisposable[] = [];

  // Configure TypeScript/JavaScript
  configureTypeScript(monaco);
  disposables.push(registerTypeScriptCodeActions(monaco));

  // Configure JSON/YAML
  disposables.push(configureJsonYaml(monaco));

  return disposables;
}

// Re-export individual configurations
export {
  configureTypeScript,
  registerTypeScriptCodeActions,
} from "./typescript-config";
export {
  configureJson,
  configureYaml,
  configureJsonYaml,
} from "./json-yaml-config";
