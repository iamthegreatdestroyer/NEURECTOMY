/**
 * Monaco Editor Mock for Vitest
 *
 * Provides mock implementations of monaco-editor APIs for testing.
 */

export const languages = {
  register: () => {},
  setLanguageConfiguration: () => {},
  setMonarchTokensProvider: () => {},
  registerCompletionItemProvider: () => {},
  CompletionItemKind: {
    Function: 1,
    Class: 5,
    Snippet: 27,
    Keyword: 14,
    Type: 25,
    Operator: 11,
    Text: 0,
    Method: 2,
    Property: 10,
    Variable: 6,
  },
  CompletionItemInsertTextRule: {
    InsertAsSnippet: 4,
    None: 0,
  },
};

export const editor = {
  create: () => ({}),
  defineTheme: () => {},
  setTheme: () => {},
  createModel: () => ({}),
};

export default {
  languages,
  editor,
};
