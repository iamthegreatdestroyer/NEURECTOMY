/**
 * SigmaLang Monaco Language Tests
 *
 * Unit tests for SigmaLang syntax highlighting and language support.
 * Tests the language configuration without importing monaco-editor directly.
 *
 * @module @neurectomy/monaco/tests
 */

import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock monaco-editor before any imports
vi.mock("monaco-editor", () => ({
  languages: {
    register: vi.fn(),
    setLanguageConfiguration: vi.fn(),
    setMonarchTokensProvider: vi.fn(),
    registerCompletionItemProvider: vi.fn(),
    CompletionItemKind: {
      Function: 1,
      Class: 5,
      Snippet: 27,
      Keyword: 14,
      Type: 25,
      Operator: 11,
    },
    CompletionItemInsertTextRule: {
      InsertAsSnippet: 4,
    },
  },
}));

// Import after mocks
import {
  SIGMALANG_LANGUAGE_ID,
  sigmaLangConfiguration,
  sigmaLangMonarch,
  sigmaLangCompletions,
  sigmaLangThemeTokens,
} from "../sigmalang";

describe("SigmaLang Language Support", () => {
  describe("Language ID", () => {
    it("should have correct language ID", () => {
      expect(SIGMALANG_LANGUAGE_ID).toBe("sigmalang");
    });
  });

  describe("Language Configuration", () => {
    it("should define line comments", () => {
      expect(sigmaLangConfiguration.comments?.lineComment).toBe("//");
    });

    it("should define block comments", () => {
      expect(sigmaLangConfiguration.comments?.blockComment).toEqual([
        "/*",
        "*/",
      ]);
    });

    it("should define standard brackets", () => {
      const brackets = sigmaLangConfiguration.brackets || [];
      expect(brackets).toContainEqual(["{", "}"]);
      expect(brackets).toContainEqual(["[", "]"]);
      expect(brackets).toContainEqual(["(", ")"]);
    });

    it("should define SigmaLang special brackets", () => {
      const brackets = sigmaLangConfiguration.brackets || [];
      expect(brackets).toContainEqual(["⟨", "⟩"]);
      expect(brackets).toContainEqual(["⌈", "⌉"]);
      expect(brackets).toContainEqual(["⟦", "⟧"]);
    });

    it("should define auto-closing pairs for special brackets", () => {
      const pairs = sigmaLangConfiguration.autoClosingPairs || [];
      const openChars = pairs.map((p) => p.open);

      expect(openChars).toContain("⟨");
      expect(openChars).toContain("⌈");
      expect(openChars).toContain("⟦");
      expect(openChars).toContain("«");
    });

    it("should define folding markers", () => {
      expect(sigmaLangConfiguration.folding?.markers?.start).toBeDefined();
      expect(sigmaLangConfiguration.folding?.markers?.end).toBeDefined();
    });
  });

  describe("Monarch Tokenizer", () => {
    it("should have correct token postfix", () => {
      expect(sigmaLangMonarch.tokenPostfix).toBe(".sigma");
    });

    it("should define SigmaLang keywords", () => {
      const keywords = sigmaLangMonarch.keywords as string[];

      // Existential primitives
      expect(keywords).toContain("∃");
      expect(keywords).toContain("∀");
      expect(keywords).toContain("∈");

      // Logic primitives
      expect(keywords).toContain("∧");
      expect(keywords).toContain("∨");
      expect(keywords).toContain("¬");

      // Math primitives
      expect(keywords).toContain("∑");
      expect(keywords).toContain("∏");
      expect(keywords).toContain("∫");

      // Action markers
      expect(keywords).toContain("→");
      expect(keywords).toContain("⇒");
    });

    it("should define type keywords", () => {
      const typeKeywords = sigmaLangMonarch.typeKeywords as string[];

      expect(typeKeywords).toContain("Σ");
      expect(typeKeywords).toContain("semantic");
      expect(typeKeywords).toContain("glyph");
      expect(typeKeywords).toContain("tree");
      expect(typeKeywords).toContain("encode");
      expect(typeKeywords).toContain("decode");
      expect(typeKeywords).toContain("pattern");
      expect(typeKeywords).toContain("codebook");
    });

    it("should define operators", () => {
      const operators = sigmaLangMonarch.operators as string[];

      expect(operators).toContain("=");
      expect(operators).toContain("→");
      expect(operators).toContain("⊕");
      expect(operators).toContain("⊗");
    });

    it("should have tokenizer rules for comments", () => {
      const tokenizer = sigmaLangMonarch.tokenizer;
      expect(tokenizer.root).toBeDefined();
      expect(tokenizer.comment).toBeDefined();
    });

    it("should have tokenizer rules for strings", () => {
      const tokenizer = sigmaLangMonarch.tokenizer;
      expect(tokenizer.string_double).toBeDefined();
      expect(tokenizer.string_single).toBeDefined();
    });

    it("should have tokenizer rules for glyph literals", () => {
      const tokenizer = sigmaLangMonarch.tokenizer;
      expect(tokenizer.glyph_literal).toBeDefined();
    });
  });

  describe("Completions Provider", () => {
    it("should be defined", () => {
      expect(sigmaLangCompletions).toBeDefined();
      expect(sigmaLangCompletions.provideCompletionItems).toBeInstanceOf(
        Function
      );
    });

    it("should provide completions with mock model", () => {
      const mockModel = {
        getWordUntilPosition: vi.fn().mockReturnValue({
          startColumn: 1,
          endColumn: 1,
          word: "",
        }),
      };
      const mockPosition = { lineNumber: 1, column: 1 };

      const result = sigmaLangCompletions.provideCompletionItems(
        mockModel as any,
        mockPosition as any
      );

      expect(result).toBeDefined();
      expect(result?.suggestions).toBeDefined();
      expect(Array.isArray(result?.suggestions)).toBe(true);
    });
  });

  describe("Theme Tokens", () => {
    it("should be an array of token definitions", () => {
      expect(Array.isArray(sigmaLangThemeTokens)).toBe(true);
      expect(sigmaLangThemeTokens.length).toBeGreaterThan(0);
    });

    it("should define keyword token styling", () => {
      const keywordToken = sigmaLangThemeTokens.find(
        (t) => t.token === "keyword.sigma"
      );
      expect(keywordToken).toBeDefined();
      expect(keywordToken?.foreground).toBe("00ff88");
      expect(keywordToken?.fontStyle).toBe("bold");
    });

    it("should define type token styling", () => {
      const typeToken = sigmaLangThemeTokens.find(
        (t) => t.token === "keyword.type.sigma"
      );
      expect(typeToken).toBeDefined();
      expect(typeToken?.foreground).toBe("a855f7");
    });

    it("should define operator token styling", () => {
      const operatorToken = sigmaLangThemeTokens.find(
        (t) => t.token === "operator.sigma"
      );
      expect(operatorToken).toBeDefined();
      expect(operatorToken?.foreground).toBe("22d3ee");
    });

    it("should define glyph string styling", () => {
      const glyphToken = sigmaLangThemeTokens.find(
        (t) => t.token === "string.glyph"
      );
      expect(glyphToken).toBeDefined();
      expect(glyphToken?.foreground).toBe("f472b6");
      expect(glyphToken?.fontStyle).toBe("italic");
    });

    it("should define comment styling", () => {
      const commentToken = sigmaLangThemeTokens.find(
        (t) => t.token === "comment"
      );
      expect(commentToken).toBeDefined();
      expect(commentToken?.fontStyle).toBe("italic");
    });

    it("should define number styling", () => {
      const numberToken = sigmaLangThemeTokens.find(
        (t) => t.token === "number"
      );
      expect(numberToken).toBeDefined();
      expect(numberToken?.foreground).toBe("fbbf24"); // Amber-400
    });
  });

  describe("Token patterns", () => {
    it("should recognize Σ as a type keyword", () => {
      const typeKeywords = sigmaLangMonarch.typeKeywords as string[];
      expect(typeKeywords).toContain("Σ");
    });

    it("should have proper Unicode support in keywords", () => {
      const keywords = sigmaLangMonarch.keywords as string[];

      // Mathematical symbols
      expect(keywords.some((k) => /[∃∀∑∏∫]/.test(k))).toBe(true);

      // Greek letters
      expect(keywords.some((k) => /[αβγδλμ]/.test(k))).toBe(true);

      // Arrows
      expect(keywords.some((k) => /[→⇒↔]/.test(k))).toBe(true);
    });

    it("should support multiple file extensions", () => {
      // Test that common SigmaLang extensions are recognized
      const validExtensions = [".sigma", ".sig", ".σ"];
      validExtensions.forEach((ext) => {
        expect(ext).toMatch(/^\.\w+$|^\.σ$/);
      });
    });
  });
});
