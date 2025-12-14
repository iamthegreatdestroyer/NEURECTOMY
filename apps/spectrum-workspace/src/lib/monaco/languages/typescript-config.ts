/**
 * TypeScript/JavaScript Language Configuration for Monaco Editor
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import type * as Monaco from "monaco-editor";

/**
 * TypeScript compiler options for NEURECTOMY projects
 */
export const TYPESCRIPT_COMPILER_OPTIONS: Monaco.languages.typescript.CompilerOptions =
  {
    target: 99, // ESNext
    module: 99, // ESNext
    moduleResolution: 2, // Node
    lib: ["esnext", "dom", "dom.iterable"],
    jsx: 4, // react-jsx
    strict: true,
    noUnusedLocals: true,
    noUnusedParameters: true,
    noFallthroughCasesInSwitch: true,
    noUncheckedIndexedAccess: true,
    esModuleInterop: true,
    skipLibCheck: true,
    forceConsistentCasingInFileNames: true,
    allowSyntheticDefaultImports: true,
    resolveJsonModule: true,
    isolatedModules: true,
    declaration: true,
    declarationMap: true,
    sourceMap: true,
    experimentalDecorators: true,
    emitDecoratorMetadata: true,
    allowJs: true,
    checkJs: false,
  };

/**
 * TypeScript diagnostic options
 */
export const TYPESCRIPT_DIAGNOSTICS_OPTIONS: Monaco.languages.typescript.DiagnosticsOptions =
  {
    noSemanticValidation: false,
    noSyntaxValidation: false,
    noSuggestionDiagnostics: false,
    diagnosticCodesToIgnore: [
      // Allow unused variables during development
      // 6133, // Variable is declared but never used
      // 6196, // Function is declared but never used
    ],
  };

/**
 * NEURECTOMY type definitions for enhanced IntelliSense
 */
const NEURECTOMY_TYPE_DEFINITIONS = `
// NEURECTOMY Agent Types
declare namespace Neurectomy {
  interface Agent {
    id: string;
    name: string;
    codename: string;
    tier: AgentTier;
    status: AgentStatus;
    description: string;
    philosophy: string;
    capabilities: string[];
    version: string;
    createdAt: Date;
    updatedAt: Date;
    metadata: Record<string, unknown>;
  }

  type AgentTier = 
    | 'foundational'
    | 'specialist'
    | 'innovator'
    | 'meta'
    | 'domain'
    | 'emerging'
    | 'human-centric'
    | 'enterprise';

  type AgentStatus = 
    | 'idle'
    | 'running'
    | 'paused'
    | 'error'
    | 'completed'
    | 'active';

  interface Workflow {
    id: string;
    name: string;
    description: string;
    nodes: WorkflowNode[];
    edges: WorkflowEdge[];
    status: WorkflowStatus;
  }

  type WorkflowStatus = 'draft' | 'active' | 'paused' | 'completed' | 'failed';

  interface WorkflowNode {
    id: string;
    type: 'agent' | 'condition' | 'action' | 'input' | 'output';
    position: { x: number; y: number };
    data: Record<string, unknown>;
  }

  interface WorkflowEdge {
    id: string;
    source: string;
    target: string;
    label?: string;
  }
}

// Global NEURECTOMY namespace
declare global {
  const neurectomy: {
    agents: Neurectomy.Agent[];
    workflows: Neurectomy.Workflow[];
    version: string;
  };
}
`;

/**
 * Configure TypeScript language settings
 */
export function configureTypeScript(monaco: typeof Monaco): void {
  const typescript = monaco.languages.typescript;

  // TypeScript configuration
  typescript.typescriptDefaults.setCompilerOptions(TYPESCRIPT_COMPILER_OPTIONS);
  typescript.typescriptDefaults.setDiagnosticsOptions(
    TYPESCRIPT_DIAGNOSTICS_OPTIONS
  );

  // Enable extra libraries
  typescript.typescriptDefaults.setEagerModelSync(true);

  // Add NEURECTOMY type definitions
  typescript.typescriptDefaults.addExtraLib(
    NEURECTOMY_TYPE_DEFINITIONS,
    "ts:neurectomy.d.ts"
  );

  // JavaScript configuration (inherits from TypeScript)
  typescript.javascriptDefaults.setCompilerOptions({
    ...TYPESCRIPT_COMPILER_OPTIONS,
    allowJs: true,
    checkJs: true,
    jsx: 2, // react
  });

  typescript.javascriptDefaults.setDiagnosticsOptions({
    noSemanticValidation: false,
    noSyntaxValidation: false,
  });

  typescript.javascriptDefaults.addExtraLib(
    NEURECTOMY_TYPE_DEFINITIONS,
    "js:neurectomy.d.ts"
  );

  // Add common library types
  const reactTypes = `
    declare module 'react' {
      export const useState: <T>(initial: T) => [T, (value: T) => void];
      export const useEffect: (effect: () => void | (() => void), deps?: any[]) => void;
      export const useCallback: <T extends (...args: any[]) => any>(callback: T, deps: any[]) => T;
      export const useMemo: <T>(factory: () => T, deps: any[]) => T;
      export const useRef: <T>(initial: T) => { current: T };
      export const useContext: <T>(context: React.Context<T>) => T;
      export const createContext: <T>(defaultValue: T) => React.Context<T>;
      export const forwardRef: <T, P>(component: (props: P, ref: React.Ref<T>) => React.ReactElement | null) => React.ForwardRefExoticComponent<React.PropsWithoutRef<P> & React.RefAttributes<T>>;
      export const memo: <T extends React.ComponentType<any>>(component: T) => T;
      export default React;
    }
  `;

  typescript.typescriptDefaults.addExtraLib(reactTypes, "ts:react.d.ts");
  typescript.javascriptDefaults.addExtraLib(reactTypes, "js:react.d.ts");
}

/**
 * Register TypeScript/JavaScript code actions
 */
export function registerTypeScriptCodeActions(
  monaco: typeof Monaco
): Monaco.IDisposable {
  return monaco.languages.registerCodeActionProvider(
    ["typescript", "javascript", "typescriptreact", "javascriptreact"],
    {
      provideCodeActions(model, range, context) {
        const actions: Monaco.languages.CodeAction[] = [];

        // Add quick fixes for common issues
        for (const marker of context.markers) {
          if (marker.code === "6133") {
            // Unused variable - offer to prefix with underscore
            const word = model.getWordAtPosition({
              lineNumber: range.startLineNumber,
              column: marker.startColumn,
            });
            if (word && !word.word.startsWith("_")) {
              actions.push({
                title: `Prefix with underscore: _${word.word}`,
                kind: "quickfix",
                edit: {
                  edits: [
                    {
                      resource: model.uri,
                      textEdit: {
                        range: {
                          startLineNumber: range.startLineNumber,
                          startColumn: marker.startColumn,
                          endLineNumber: range.startLineNumber,
                          endColumn: marker.startColumn + word.word.length,
                        },
                        text: `_${word.word}`,
                      },
                      versionId: undefined,
                    },
                  ],
                },
                isPreferred: true,
              });
            }
          }
        }

        return { actions, dispose: () => {} };
      },
    }
  );
}
