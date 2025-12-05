/**
 * @fileoverview AST-Based Code Comparison Engine
 * @module @neurectomy/legal-fortress/plagiarism/ast-comparator
 *
 * Agent Assignment: @PHANTOM (Reverse Engineering) + @CORE (Low-Level Parsing)
 *
 * Implements deep Abstract Syntax Tree analysis for structural code comparison,
 * detecting plagiarism even when surface-level changes (variable renaming,
 * comment modification, whitespace changes) have been applied.
 *
 * @author NEURECTOMY Phase 5 - Legal Fortress
 * @version 1.0.0
 */

import { EventEmitter } from "events";

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Supported programming languages for AST analysis
 */
export type SupportedLanguage =
  | "typescript"
  | "javascript"
  | "python"
  | "java"
  | "csharp"
  | "cpp"
  | "go"
  | "rust"
  | "ruby"
  | "php";

/**
 * AST node types for cross-language normalization
 */
export enum NormalizedNodeType {
  // Declarations
  FUNCTION_DECLARATION = "function_declaration",
  CLASS_DECLARATION = "class_declaration",
  VARIABLE_DECLARATION = "variable_declaration",
  INTERFACE_DECLARATION = "interface_declaration",
  TYPE_DECLARATION = "type_declaration",
  ENUM_DECLARATION = "enum_declaration",

  // Statements
  IF_STATEMENT = "if_statement",
  FOR_STATEMENT = "for_statement",
  WHILE_STATEMENT = "while_statement",
  DO_WHILE_STATEMENT = "do_while_statement",
  SWITCH_STATEMENT = "switch_statement",
  TRY_STATEMENT = "try_statement",
  RETURN_STATEMENT = "return_statement",
  THROW_STATEMENT = "throw_statement",
  BREAK_STATEMENT = "break_statement",
  CONTINUE_STATEMENT = "continue_statement",

  // Expressions
  FUNCTION_CALL = "function_call",
  METHOD_CALL = "method_call",
  BINARY_EXPRESSION = "binary_expression",
  UNARY_EXPRESSION = "unary_expression",
  ASSIGNMENT_EXPRESSION = "assignment_expression",
  CONDITIONAL_EXPRESSION = "conditional_expression",
  MEMBER_ACCESS = "member_access",
  ARRAY_ACCESS = "array_access",
  NEW_EXPRESSION = "new_expression",
  AWAIT_EXPRESSION = "await_expression",

  // Literals
  STRING_LITERAL = "string_literal",
  NUMBER_LITERAL = "number_literal",
  BOOLEAN_LITERAL = "boolean_literal",
  NULL_LITERAL = "null_literal",
  ARRAY_LITERAL = "array_literal",
  OBJECT_LITERAL = "object_literal",

  // Other
  IDENTIFIER = "identifier",
  PARAMETER = "parameter",
  BLOCK = "block",
  COMMENT = "comment",
  IMPORT = "import",
  EXPORT = "export",
  UNKNOWN = "unknown",
}

/**
 * Normalized AST node for cross-language comparison
 */
export interface NormalizedASTNode {
  /** Unique node identifier */
  id: string;
  /** Normalized node type */
  type: NormalizedNodeType;
  /** Child nodes */
  children: NormalizedASTNode[];
  /** Node metadata */
  metadata: NodeMetadata;
  /** Structural hash for quick comparison */
  structuralHash: string;
  /** Source location (for reporting) */
  location?: SourceLocation;
}

/**
 * Node metadata for detailed analysis
 */
export interface NodeMetadata {
  /** Original node type from parser */
  originalType: string;
  /** Normalized operator (for expressions) */
  operator?: string;
  /** Arity (number of parameters for functions) */
  arity?: number;
  /** Depth in AST */
  depth: number;
  /** Subtree size (total descendants) */
  subtreeSize: number;
  /** Is this a leaf node */
  isLeaf: boolean;
  /** Additional language-specific metadata */
  extra?: Record<string, unknown>;
}

/**
 * Source code location
 */
export interface SourceLocation {
  startLine: number;
  startColumn: number;
  endLine: number;
  endColumn: number;
  filePath?: string;
}

/**
 * AST comparison result
 */
export interface ASTComparisonResult {
  /** Overall structural similarity (0-1) */
  similarity: number;
  /** Matched subtree pairs */
  matchedSubtrees: SubtreeMatch[];
  /** Structural differences */
  differences: StructuralDifference[];
  /** Summary statistics */
  statistics: ComparisonStatistics;
  /** Detailed analysis */
  analysis: DetailedAnalysis;
}

/**
 * Matched subtree pair
 */
export interface SubtreeMatch {
  /** Node from source AST */
  sourceNode: NormalizedASTNode;
  /** Node from target AST */
  targetNode: NormalizedASTNode;
  /** Match similarity score */
  similarity: number;
  /** Match type */
  matchType: MatchType;
  /** Transformation applied (if any) */
  transformation?: TransformationType;
}

/**
 * Match type classification
 */
export enum MatchType {
  /** Exact structural match */
  EXACT = "exact",
  /** Match after normalization */
  NORMALIZED = "normalized",
  /** Partial subtree match */
  PARTIAL = "partial",
  /** Semantic equivalence */
  SEMANTIC = "semantic",
  /** Potential with modifications */
  MODIFIED = "modified",
}

/**
 * Common code transformation types
 */
export enum TransformationType {
  RENAME_VARIABLE = "rename_variable",
  RENAME_FUNCTION = "rename_function",
  REORDER_STATEMENTS = "reorder_statements",
  LOOP_TRANSFORM = "loop_transform",
  EXTRACT_VARIABLE = "extract_variable",
  INLINE_VARIABLE = "inline_variable",
  EXTRACT_METHOD = "extract_method",
  INLINE_METHOD = "inline_method",
  CHANGE_LITERALS = "change_literals",
  ADD_DEAD_CODE = "add_dead_code",
  SPLIT_EXPRESSION = "split_expression",
  MERGE_EXPRESSIONS = "merge_expressions",
}

/**
 * Structural difference between ASTs
 */
export interface StructuralDifference {
  type: "addition" | "deletion" | "modification";
  location: "source" | "target";
  node: NormalizedASTNode;
  description: string;
}

/**
 * Comparison statistics
 */
export interface ComparisonStatistics {
  sourceNodeCount: number;
  targetNodeCount: number;
  matchedNodeCount: number;
  exactMatches: number;
  normalizedMatches: number;
  partialMatches: number;
  uniqueToSource: number;
  uniqueToTarget: number;
  maxMatchedDepth: number;
  averageMatchSimilarity: number;
}

/**
 * Detailed analysis results
 */
export interface DetailedAnalysis {
  /** Function-level similarities */
  functionSimilarities: Array<{
    sourceName: string;
    targetName: string;
    similarity: number;
  }>;
  /** Control flow similarity */
  controlFlowSimilarity: number;
  /** Data flow similarity */
  dataFlowSimilarity: number;
  /** API usage similarity */
  apiUsageSimilarity: number;
  /** Detected obfuscation techniques */
  detectedObfuscations: TransformationType[];
  /** Confidence in plagiarism detection */
  plagiarismConfidence: number;
}

/**
 * Configuration for AST comparator
 */
export interface ASTComparatorConfig {
  /** Minimum subtree size to consider for matching */
  minSubtreeSize: number;
  /** Similarity threshold for match detection */
  similarityThreshold: number;
  /** Enable semantic matching */
  enableSemanticMatching: boolean;
  /** Weight for structural similarity */
  structuralWeight: number;
  /** Weight for control flow similarity */
  controlFlowWeight: number;
  /** Weight for identifier similarity */
  identifierWeight: number;
  /** Maximum tree edit distance for matching */
  maxEditDistance: number;
  /** Enable obfuscation detection */
  detectObfuscation: boolean;
}

// ============================================================================
// Language Parsers
// ============================================================================

/**
 * Abstract base class for language-specific AST parsers
 */
abstract class LanguageParser {
  abstract parse(code: string): NormalizedASTNode;
  abstract getLanguage(): SupportedLanguage;

  protected generateNodeId(): string {
    return `node_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }

  protected computeStructuralHash(node: NormalizedASTNode): string {
    const childHashes = node.children
      .map((c) => c.structuralHash)
      .sort()
      .join("|");
    const content = `${node.type}:${node.metadata.operator || ""}:${node.metadata.arity || 0}:${childHashes}`;
    return this.hashString(content);
  }

  protected hashString(str: string): string {
    // Simple hash function - in production use crypto
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(36);
  }

  protected calculateSubtreeSize(node: NormalizedASTNode): number {
    return (
      1 +
      node.children.reduce(
        (sum, child) => sum + this.calculateSubtreeSize(child),
        0
      )
    );
  }
}

/**
 * TypeScript/JavaScript AST parser
 */
class TypeScriptParser extends LanguageParser {
  getLanguage(): SupportedLanguage {
    return "typescript";
  }

  parse(code: string): NormalizedASTNode {
    // Simplified parsing - in production, use ts-morph or @typescript-eslint/parser
    const ast = this.parseToRawAST(code);
    return this.normalizeAST(ast, 0);
  }

  private parseToRawAST(code: string): any {
    // Placeholder - would use actual TypeScript parser
    // For demonstration, create a mock AST structure
    return {
      type: "Program",
      body: this.tokenize(code),
    };
  }

  private tokenize(code: string): any[] {
    const nodes: any[] = [];
    const lines = code.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      if (
        line.startsWith("function ") ||
        line.includes("function(") ||
        line.includes("=>")
      ) {
        nodes.push({
          type: "FunctionDeclaration",
          line: i + 1,
          content: line,
        });
      } else if (line.startsWith("class ")) {
        nodes.push({
          type: "ClassDeclaration",
          line: i + 1,
          content: line,
        });
      } else if (line.startsWith("if ") || line.startsWith("if(")) {
        nodes.push({
          type: "IfStatement",
          line: i + 1,
          content: line,
        });
      } else if (line.startsWith("for ") || line.startsWith("for(")) {
        nodes.push({
          type: "ForStatement",
          line: i + 1,
          content: line,
        });
      } else if (line.startsWith("while ") || line.startsWith("while(")) {
        nodes.push({
          type: "WhileStatement",
          line: i + 1,
          content: line,
        });
      } else if (line.startsWith("return ")) {
        nodes.push({
          type: "ReturnStatement",
          line: i + 1,
          content: line,
        });
      } else if (
        line.includes("const ") ||
        line.includes("let ") ||
        line.includes("var ")
      ) {
        nodes.push({
          type: "VariableDeclaration",
          line: i + 1,
          content: line,
        });
      }
    }

    return nodes;
  }

  private normalizeAST(rawNode: any, depth: number): NormalizedASTNode {
    const nodeType = this.mapNodeType(rawNode.type);
    const children = (rawNode.body || []).map((child: any) =>
      this.normalizeAST(child, depth + 1)
    );

    const node: NormalizedASTNode = {
      id: this.generateNodeId(),
      type: nodeType,
      children,
      metadata: {
        originalType: rawNode.type,
        depth,
        subtreeSize: 0,
        isLeaf: children.length === 0,
      },
      structuralHash: "",
    };

    node.metadata.subtreeSize = this.calculateSubtreeSize(node);
    node.structuralHash = this.computeStructuralHash(node);

    return node;
  }

  private mapNodeType(originalType: string): NormalizedNodeType {
    const mapping: Record<string, NormalizedNodeType> = {
      Program: NormalizedNodeType.BLOCK,
      FunctionDeclaration: NormalizedNodeType.FUNCTION_DECLARATION,
      ClassDeclaration: NormalizedNodeType.CLASS_DECLARATION,
      VariableDeclaration: NormalizedNodeType.VARIABLE_DECLARATION,
      IfStatement: NormalizedNodeType.IF_STATEMENT,
      ForStatement: NormalizedNodeType.FOR_STATEMENT,
      WhileStatement: NormalizedNodeType.WHILE_STATEMENT,
      ReturnStatement: NormalizedNodeType.RETURN_STATEMENT,
      CallExpression: NormalizedNodeType.FUNCTION_CALL,
      BinaryExpression: NormalizedNodeType.BINARY_EXPRESSION,
      Identifier: NormalizedNodeType.IDENTIFIER,
    };

    return mapping[originalType] || NormalizedNodeType.UNKNOWN;
  }
}

/**
 * Python AST parser
 */
class PythonParser extends LanguageParser {
  getLanguage(): SupportedLanguage {
    return "python";
  }

  parse(code: string): NormalizedASTNode {
    const ast = this.parseToRawAST(code);
    return this.normalizeAST(ast, 0);
  }

  private parseToRawAST(code: string): any {
    return {
      type: "Module",
      body: this.tokenize(code),
    };
  }

  private tokenize(code: string): any[] {
    const nodes: any[] = [];
    const lines = code.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      if (line.startsWith("def ")) {
        nodes.push({ type: "FunctionDef", line: i + 1, content: line });
      } else if (line.startsWith("class ")) {
        nodes.push({ type: "ClassDef", line: i + 1, content: line });
      } else if (line.startsWith("if ")) {
        nodes.push({ type: "If", line: i + 1, content: line });
      } else if (line.startsWith("for ")) {
        nodes.push({ type: "For", line: i + 1, content: line });
      } else if (line.startsWith("while ")) {
        nodes.push({ type: "While", line: i + 1, content: line });
      } else if (line.startsWith("return ")) {
        nodes.push({ type: "Return", line: i + 1, content: line });
      } else if (line.includes(" = ")) {
        nodes.push({ type: "Assign", line: i + 1, content: line });
      }
    }

    return nodes;
  }

  private normalizeAST(rawNode: any, depth: number): NormalizedASTNode {
    const nodeType = this.mapNodeType(rawNode.type);
    const children = (rawNode.body || []).map((child: any) =>
      this.normalizeAST(child, depth + 1)
    );

    const node: NormalizedASTNode = {
      id: this.generateNodeId(),
      type: nodeType,
      children,
      metadata: {
        originalType: rawNode.type,
        depth,
        subtreeSize: 0,
        isLeaf: children.length === 0,
      },
      structuralHash: "",
    };

    node.metadata.subtreeSize = this.calculateSubtreeSize(node);
    node.structuralHash = this.computeStructuralHash(node);

    return node;
  }

  private mapNodeType(originalType: string): NormalizedNodeType {
    const mapping: Record<string, NormalizedNodeType> = {
      Module: NormalizedNodeType.BLOCK,
      FunctionDef: NormalizedNodeType.FUNCTION_DECLARATION,
      AsyncFunctionDef: NormalizedNodeType.FUNCTION_DECLARATION,
      ClassDef: NormalizedNodeType.CLASS_DECLARATION,
      If: NormalizedNodeType.IF_STATEMENT,
      For: NormalizedNodeType.FOR_STATEMENT,
      While: NormalizedNodeType.WHILE_STATEMENT,
      Return: NormalizedNodeType.RETURN_STATEMENT,
      Assign: NormalizedNodeType.ASSIGNMENT_EXPRESSION,
      Call: NormalizedNodeType.FUNCTION_CALL,
      BinOp: NormalizedNodeType.BINARY_EXPRESSION,
      Name: NormalizedNodeType.IDENTIFIER,
    };

    return mapping[originalType] || NormalizedNodeType.UNKNOWN;
  }
}

// ============================================================================
// Tree Matching Algorithms
// ============================================================================

/**
 * Zhang-Shasha Tree Edit Distance calculator
 */
class TreeEditDistance {
  private insertCost = 1;
  private deleteCost = 1;
  private renameCost = 1;

  /**
   * Calculate the tree edit distance between two ASTs
   */
  calculate(tree1: NormalizedASTNode, tree2: NormalizedASTNode): number {
    const nodes1 = this.postorderTraversal(tree1);
    const nodes2 = this.postorderTraversal(tree2);

    const m = nodes1.length;
    const n = nodes2.length;

    // Initialize distance matrix
    const dp: number[][] = Array(m + 1)
      .fill(null)
      .map(() => Array(n + 1).fill(0));

    // Base cases
    for (let i = 1; i <= m; i++) {
      dp[i][0] = dp[i - 1][0] + this.deleteCost;
    }
    for (let j = 1; j <= n; j++) {
      dp[0][j] = dp[0][j - 1] + this.insertCost;
    }

    // Fill the matrix
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const node1 = nodes1[i - 1];
        const node2 = nodes2[j - 1];

        const costRename = node1.type === node2.type ? 0 : this.renameCost;

        dp[i][j] = Math.min(
          dp[i - 1][j] + this.deleteCost,
          dp[i][j - 1] + this.insertCost,
          dp[i - 1][j - 1] + costRename
        );
      }
    }

    return dp[m][n];
  }

  /**
   * Convert edit distance to similarity score (0-1)
   */
  toSimilarity(distance: number, size1: number, size2: number): number {
    const maxSize = Math.max(size1, size2);
    if (maxSize === 0) return 1;
    return Math.max(0, 1 - distance / maxSize);
  }

  private postorderTraversal(node: NormalizedASTNode): NormalizedASTNode[] {
    const result: NormalizedASTNode[] = [];
    for (const child of node.children) {
      result.push(...this.postorderTraversal(child));
    }
    result.push(node);
    return result;
  }
}

/**
 * GumTree-inspired AST matching algorithm
 */
class GumTreeMatcher {
  private minHeight = 2;
  private maxSize = 100;

  /**
   * Find optimal mapping between two ASTs
   */
  match(
    source: NormalizedASTNode,
    target: NormalizedASTNode
  ): Map<string, string> {
    const mappings = new Map<string, string>();

    // Phase 1: Top-down matching based on structural hash
    this.topDownMatch(source, target, mappings);

    // Phase 2: Bottom-up matching for remaining nodes
    this.bottomUpMatch(source, target, mappings);

    return mappings;
  }

  private topDownMatch(
    source: NormalizedASTNode,
    target: NormalizedASTNode,
    mappings: Map<string, string>
  ): void {
    // Build hash index for target tree
    const targetIndex = this.buildHashIndex(target);

    // Match nodes with identical structural hashes
    const queue = [source];
    while (queue.length > 0) {
      const node = queue.shift()!;

      if (node.metadata.subtreeSize >= this.minHeight) {
        const candidates = targetIndex.get(node.structuralHash) || [];

        for (const candidate of candidates) {
          if (
            !this.isAncestorMapped(node, mappings) &&
            !this.isDescendantMapped(candidate, mappings)
          ) {
            mappings.set(node.id, candidate.id);
            this.mapDescendants(node, candidate, mappings);
            break;
          }
        }
      }

      queue.push(...node.children);
    }
  }

  private bottomUpMatch(
    source: NormalizedASTNode,
    target: NormalizedASTNode,
    mappings: Map<string, string>
  ): void {
    // Get unmapped source nodes
    const unmappedSource = this.getUnmappedNodes(source, mappings, "source");
    const unmappedTarget = this.getUnmappedNodes(target, mappings, "target");

    // Try to match based on similarity
    for (const sNode of unmappedSource) {
      let bestMatch: NormalizedASTNode | null = null;
      let bestScore = 0;

      for (const tNode of unmappedTarget) {
        if (sNode.type === tNode.type) {
          const score = this.calculateNodeSimilarity(sNode, tNode);
          if (score > bestScore && score > 0.5) {
            bestScore = score;
            bestMatch = tNode;
          }
        }
      }

      if (bestMatch) {
        mappings.set(sNode.id, bestMatch.id);
        const idx = unmappedTarget.indexOf(bestMatch);
        if (idx > -1) unmappedTarget.splice(idx, 1);
      }
    }
  }

  private buildHashIndex(
    node: NormalizedASTNode
  ): Map<string, NormalizedASTNode[]> {
    const index = new Map<string, NormalizedASTNode[]>();

    const traverse = (n: NormalizedASTNode) => {
      if (!index.has(n.structuralHash)) {
        index.set(n.structuralHash, []);
      }
      index.get(n.structuralHash)!.push(n);
      n.children.forEach(traverse);
    };

    traverse(node);
    return index;
  }

  private isAncestorMapped(
    node: NormalizedASTNode,
    mappings: Map<string, string>
  ): boolean {
    // Simplified - would need parent references
    return false;
  }

  private isDescendantMapped(
    node: NormalizedASTNode,
    mappings: Map<string, string>
  ): boolean {
    const check = (n: NormalizedASTNode): boolean => {
      if (Array.from(mappings.values()).includes(n.id)) return true;
      return n.children.some(check);
    };
    return node.children.some(check);
  }

  private mapDescendants(
    source: NormalizedASTNode,
    target: NormalizedASTNode,
    mappings: Map<string, string>
  ): void {
    for (
      let i = 0;
      i < Math.min(source.children.length, target.children.length);
      i++
    ) {
      mappings.set(source.children[i].id, target.children[i].id);
      this.mapDescendants(source.children[i], target.children[i], mappings);
    }
  }

  private getUnmappedNodes(
    root: NormalizedASTNode,
    mappings: Map<string, string>,
    side: "source" | "target"
  ): NormalizedASTNode[] {
    const unmapped: NormalizedASTNode[] = [];
    const mappedIds =
      side === "source" ? new Set(mappings.keys()) : new Set(mappings.values());

    const traverse = (node: NormalizedASTNode) => {
      if (!mappedIds.has(node.id)) {
        unmapped.push(node);
      }
      node.children.forEach(traverse);
    };

    traverse(root);
    return unmapped;
  }

  private calculateNodeSimilarity(
    node1: NormalizedASTNode,
    node2: NormalizedASTNode
  ): number {
    if (node1.type !== node2.type) return 0;

    let score = 0.5; // Base score for matching type

    if (node1.metadata.arity === node2.metadata.arity) {
      score += 0.2;
    }

    if (node1.metadata.operator === node2.metadata.operator) {
      score += 0.3;
    }

    return score;
  }
}

// ============================================================================
// Main AST Comparator
// ============================================================================

/**
 * Primary AST comparison engine
 * Combines multiple algorithms for comprehensive structural analysis
 */
export class ASTComparator extends EventEmitter {
  private config: ASTComparatorConfig;
  private parsers: Map<SupportedLanguage, LanguageParser>;
  private treeEditDistance: TreeEditDistance;
  private gumTreeMatcher: GumTreeMatcher;

  constructor(config?: Partial<ASTComparatorConfig>) {
    super();

    this.config = {
      minSubtreeSize: 5,
      similarityThreshold: 0.7,
      enableSemanticMatching: true,
      structuralWeight: 0.6,
      controlFlowWeight: 0.25,
      identifierWeight: 0.15,
      maxEditDistance: 50,
      detectObfuscation: true,
      ...config,
    };

    this.parsers = new Map();
    this.parsers.set("typescript", new TypeScriptParser());
    this.parsers.set("javascript", new TypeScriptParser());
    this.parsers.set("python", new PythonParser());

    this.treeEditDistance = new TreeEditDistance();
    this.gumTreeMatcher = new GumTreeMatcher();
  }

  /**
   * Compare two code snippets
   */
  async compare(
    sourceCode: string,
    targetCode: string,
    language: SupportedLanguage
  ): Promise<ASTComparisonResult> {
    this.emit("comparison:start", { language });

    const parser = this.parsers.get(language);
    if (!parser) {
      throw new Error(`Unsupported language: ${language}`);
    }

    // Parse both code snippets
    const sourceAST = parser.parse(sourceCode);
    const targetAST = parser.parse(targetCode);

    this.emit("parsing:complete", {
      sourceNodes: sourceAST.metadata.subtreeSize,
      targetNodes: targetAST.metadata.subtreeSize,
    });

    // Find node mappings
    const mappings = this.gumTreeMatcher.match(sourceAST, targetAST);

    // Calculate edit distance
    const editDistance = this.treeEditDistance.calculate(sourceAST, targetAST);
    const structuralSimilarity = this.treeEditDistance.toSimilarity(
      editDistance,
      sourceAST.metadata.subtreeSize,
      targetAST.metadata.subtreeSize
    );

    // Find matched subtrees
    const matchedSubtrees = this.findMatchedSubtrees(
      sourceAST,
      targetAST,
      mappings
    );

    // Calculate detailed similarities
    const controlFlowSimilarity = this.calculateControlFlowSimilarity(
      sourceAST,
      targetAST
    );
    const dataFlowSimilarity = this.calculateDataFlowSimilarity(
      sourceAST,
      targetAST
    );
    const apiUsageSimilarity = this.calculateAPIUsageSimilarity(
      sourceAST,
      targetAST
    );

    // Detect obfuscation patterns
    const detectedObfuscations = this.config.detectObfuscation
      ? this.detectObfuscationPatterns(sourceAST, targetAST, matchedSubtrees)
      : [];

    // Calculate overall similarity
    const overallSimilarity = this.calculateOverallSimilarity(
      structuralSimilarity,
      controlFlowSimilarity,
      matchedSubtrees
    );

    // Find structural differences
    const differences = this.findStructuralDifferences(
      sourceAST,
      targetAST,
      mappings
    );

    // Function-level analysis
    const functionSimilarities = this.analyzeFunctionSimilarities(
      sourceAST,
      targetAST
    );

    // Calculate statistics
    const statistics = this.calculateStatistics(
      sourceAST,
      targetAST,
      mappings,
      matchedSubtrees
    );

    const result: ASTComparisonResult = {
      similarity: overallSimilarity,
      matchedSubtrees,
      differences,
      statistics,
      analysis: {
        functionSimilarities,
        controlFlowSimilarity,
        dataFlowSimilarity,
        apiUsageSimilarity,
        detectedObfuscations,
        plagiarismConfidence: this.calculatePlagiarismConfidence(
          overallSimilarity,
          detectedObfuscations,
          statistics
        ),
      },
    };

    this.emit("comparison:complete", result);
    return result;
  }

  /**
   * Batch compare multiple code pairs
   */
  async batchCompare(
    pairs: Array<{
      sourceCode: string;
      targetCode: string;
      language: SupportedLanguage;
      id: string;
    }>
  ): Promise<Map<string, ASTComparisonResult>> {
    const results = new Map<string, ASTComparisonResult>();

    for (const pair of pairs) {
      try {
        const result = await this.compare(
          pair.sourceCode,
          pair.targetCode,
          pair.language
        );
        results.set(pair.id, result);
      } catch (error) {
        this.emit("comparison:error", { id: pair.id, error });
      }
    }

    return results;
  }

  private findMatchedSubtrees(
    source: NormalizedASTNode,
    target: NormalizedASTNode,
    mappings: Map<string, string>
  ): SubtreeMatch[] {
    const matches: SubtreeMatch[] = [];
    const targetIndex = this.buildNodeIndex(target);

    const traverse = (node: NormalizedASTNode) => {
      const mappedId = mappings.get(node.id);
      if (mappedId && targetIndex.has(mappedId)) {
        const targetNode = targetIndex.get(mappedId)!;

        const similarity =
          node.structuralHash === targetNode.structuralHash
            ? 1.0
            : this.treeEditDistance.toSimilarity(
                this.treeEditDistance.calculate(node, targetNode),
                node.metadata.subtreeSize,
                targetNode.metadata.subtreeSize
              );

        const matchType = this.classifyMatch(node, targetNode, similarity);

        matches.push({
          sourceNode: node,
          targetNode,
          similarity,
          matchType,
          transformation: this.detectTransformation(node, targetNode),
        });
      }

      node.children.forEach(traverse);
    };

    traverse(source);
    return matches;
  }

  private buildNodeIndex(
    root: NormalizedASTNode
  ): Map<string, NormalizedASTNode> {
    const index = new Map<string, NormalizedASTNode>();

    const traverse = (node: NormalizedASTNode) => {
      index.set(node.id, node);
      node.children.forEach(traverse);
    };

    traverse(root);
    return index;
  }

  private classifyMatch(
    source: NormalizedASTNode,
    target: NormalizedASTNode,
    similarity: number
  ): MatchType {
    if (source.structuralHash === target.structuralHash) {
      return MatchType.EXACT;
    }
    if (similarity >= 0.95) {
      return MatchType.NORMALIZED;
    }
    if (similarity >= 0.8) {
      return MatchType.PARTIAL;
    }
    if (similarity >= 0.6) {
      return MatchType.MODIFIED;
    }
    return MatchType.SEMANTIC;
  }

  private detectTransformation(
    source: NormalizedASTNode,
    target: NormalizedASTNode
  ): TransformationType | undefined {
    // Detect common transformations
    if (source.type === target.type) {
      if (source.children.length !== target.children.length) {
        if (source.children.length < target.children.length) {
          return TransformationType.ADD_DEAD_CODE;
        }
        return TransformationType.INLINE_VARIABLE;
      }
    }

    // Check for loop transformations
    if (
      (source.type === NormalizedNodeType.FOR_STATEMENT &&
        target.type === NormalizedNodeType.WHILE_STATEMENT) ||
      (source.type === NormalizedNodeType.WHILE_STATEMENT &&
        target.type === NormalizedNodeType.FOR_STATEMENT)
    ) {
      return TransformationType.LOOP_TRANSFORM;
    }

    return undefined;
  }

  private calculateControlFlowSimilarity(
    source: NormalizedASTNode,
    target: NormalizedASTNode
  ): number {
    const controlFlowTypes = new Set([
      NormalizedNodeType.IF_STATEMENT,
      NormalizedNodeType.FOR_STATEMENT,
      NormalizedNodeType.WHILE_STATEMENT,
      NormalizedNodeType.DO_WHILE_STATEMENT,
      NormalizedNodeType.SWITCH_STATEMENT,
      NormalizedNodeType.TRY_STATEMENT,
    ]);

    const getControlFlowSequence = (
      node: NormalizedASTNode
    ): NormalizedNodeType[] => {
      const sequence: NormalizedNodeType[] = [];
      const traverse = (n: NormalizedASTNode) => {
        if (controlFlowTypes.has(n.type)) {
          sequence.push(n.type);
        }
        n.children.forEach(traverse);
      };
      traverse(node);
      return sequence;
    };

    const sourceSeq = getControlFlowSequence(source);
    const targetSeq = getControlFlowSequence(target);

    return this.calculateSequenceSimilarity(sourceSeq, targetSeq);
  }

  private calculateDataFlowSimilarity(
    source: NormalizedASTNode,
    target: NormalizedASTNode
  ): number {
    const dataFlowTypes = new Set([
      NormalizedNodeType.VARIABLE_DECLARATION,
      NormalizedNodeType.ASSIGNMENT_EXPRESSION,
      NormalizedNodeType.RETURN_STATEMENT,
    ]);

    const countDataFlowNodes = (node: NormalizedASTNode): number => {
      let count = dataFlowTypes.has(node.type) ? 1 : 0;
      node.children.forEach((child) => {
        count += countDataFlowNodes(child);
      });
      return count;
    };

    const sourceCount = countDataFlowNodes(source);
    const targetCount = countDataFlowNodes(target);

    if (sourceCount === 0 && targetCount === 0) return 1;
    const maxCount = Math.max(sourceCount, targetCount);
    const diff = Math.abs(sourceCount - targetCount);

    return 1 - diff / maxCount;
  }

  private calculateAPIUsageSimilarity(
    source: NormalizedASTNode,
    target: NormalizedASTNode
  ): number {
    const getAPICalls = (node: NormalizedASTNode): Set<string> => {
      const calls = new Set<string>();
      const traverse = (n: NormalizedASTNode) => {
        if (
          n.type === NormalizedNodeType.FUNCTION_CALL ||
          n.type === NormalizedNodeType.METHOD_CALL
        ) {
          calls.add(n.structuralHash);
        }
        n.children.forEach(traverse);
      };
      traverse(node);
      return calls;
    };

    const sourceCalls = getAPICalls(source);
    const targetCalls = getAPICalls(target);

    if (sourceCalls.size === 0 && targetCalls.size === 0) return 1;

    const intersection = new Set(
      [...sourceCalls].filter((x) => targetCalls.has(x))
    );
    const union = new Set([...sourceCalls, ...targetCalls]);

    return intersection.size / union.size;
  }

  private detectObfuscationPatterns(
    source: NormalizedASTNode,
    target: NormalizedASTNode,
    matches: SubtreeMatch[]
  ): TransformationType[] {
    const detected: Set<TransformationType> = new Set();

    for (const match of matches) {
      if (match.transformation) {
        detected.add(match.transformation);
      }
    }

    // Check for systematic renaming
    if (this.detectSystematicRenaming(matches)) {
      detected.add(TransformationType.RENAME_VARIABLE);
      detected.add(TransformationType.RENAME_FUNCTION);
    }

    // Check for dead code insertion
    if (target.metadata.subtreeSize > source.metadata.subtreeSize * 1.3) {
      detected.add(TransformationType.ADD_DEAD_CODE);
    }

    return Array.from(detected);
  }

  private detectSystematicRenaming(matches: SubtreeMatch[]): boolean {
    const identifierMatches = matches.filter(
      (m) => m.sourceNode.type === NormalizedNodeType.IDENTIFIER
    );

    // If most identifiers match but have different hashes, systematic renaming detected
    const renamedCount = identifierMatches.filter(
      (m) =>
        m.matchType === MatchType.NORMALIZED ||
        m.matchType === MatchType.MODIFIED
    ).length;

    return (
      identifierMatches.length > 0 &&
      renamedCount / identifierMatches.length > 0.5
    );
  }

  private calculateOverallSimilarity(
    structuralSimilarity: number,
    controlFlowSimilarity: number,
    matches: SubtreeMatch[]
  ): number {
    const avgMatchSimilarity =
      matches.length > 0
        ? matches.reduce((sum, m) => sum + m.similarity, 0) / matches.length
        : 0;

    return (
      structuralSimilarity * this.config.structuralWeight +
      controlFlowSimilarity * this.config.controlFlowWeight +
      avgMatchSimilarity * this.config.identifierWeight
    );
  }

  private findStructuralDifferences(
    source: NormalizedASTNode,
    target: NormalizedASTNode,
    mappings: Map<string, string>
  ): StructuralDifference[] {
    const differences: StructuralDifference[] = [];
    const mappedSourceIds = new Set(mappings.keys());
    const mappedTargetIds = new Set(mappings.values());

    // Find unique to source
    const findUnmapped = (
      node: NormalizedASTNode,
      side: "source" | "target",
      mapped: Set<string>
    ) => {
      if (
        !mapped.has(node.id) &&
        node.metadata.subtreeSize >= this.config.minSubtreeSize
      ) {
        differences.push({
          type: side === "source" ? "deletion" : "addition",
          location: side,
          node,
          description: `${node.type} not found in ${side === "source" ? "target" : "source"}`,
        });
      }
      node.children.forEach((child) => findUnmapped(child, side, mapped));
    };

    findUnmapped(source, "source", mappedSourceIds);
    findUnmapped(target, "target", mappedTargetIds);

    return differences;
  }

  private analyzeFunctionSimilarities(
    source: NormalizedASTNode,
    target: NormalizedASTNode
  ): Array<{ sourceName: string; targetName: string; similarity: number }> {
    const getFunctions = (node: NormalizedASTNode): NormalizedASTNode[] => {
      const functions: NormalizedASTNode[] = [];
      const traverse = (n: NormalizedASTNode) => {
        if (n.type === NormalizedNodeType.FUNCTION_DECLARATION) {
          functions.push(n);
        }
        n.children.forEach(traverse);
      };
      traverse(node);
      return functions;
    };

    const sourceFunctions = getFunctions(source);
    const targetFunctions = getFunctions(target);
    const similarities: Array<{
      sourceName: string;
      targetName: string;
      similarity: number;
    }> = [];

    for (const sf of sourceFunctions) {
      for (const tf of targetFunctions) {
        const similarity = this.treeEditDistance.toSimilarity(
          this.treeEditDistance.calculate(sf, tf),
          sf.metadata.subtreeSize,
          tf.metadata.subtreeSize
        );

        if (similarity > this.config.similarityThreshold) {
          similarities.push({
            sourceName: sf.id,
            targetName: tf.id,
            similarity,
          });
        }
      }
    }

    return similarities.sort((a, b) => b.similarity - a.similarity);
  }

  private calculateStatistics(
    source: NormalizedASTNode,
    target: NormalizedASTNode,
    mappings: Map<string, string>,
    matches: SubtreeMatch[]
  ): ComparisonStatistics {
    const exactMatches = matches.filter(
      (m) => m.matchType === MatchType.EXACT
    ).length;
    const normalizedMatches = matches.filter(
      (m) => m.matchType === MatchType.NORMALIZED
    ).length;
    const partialMatches = matches.filter(
      (m) => m.matchType === MatchType.PARTIAL
    ).length;

    return {
      sourceNodeCount: source.metadata.subtreeSize,
      targetNodeCount: target.metadata.subtreeSize,
      matchedNodeCount: mappings.size,
      exactMatches,
      normalizedMatches,
      partialMatches,
      uniqueToSource: source.metadata.subtreeSize - mappings.size,
      uniqueToTarget: target.metadata.subtreeSize - mappings.size,
      maxMatchedDepth: Math.max(
        ...matches.map((m) => m.sourceNode.metadata.depth),
        0
      ),
      averageMatchSimilarity:
        matches.length > 0
          ? matches.reduce((sum, m) => sum + m.similarity, 0) / matches.length
          : 0,
    };
  }

  private calculatePlagiarismConfidence(
    similarity: number,
    obfuscations: TransformationType[],
    statistics: ComparisonStatistics
  ): number {
    let confidence = similarity;

    // Boost confidence if obfuscation detected
    if (obfuscations.length > 0) {
      confidence += 0.1 * Math.min(obfuscations.length, 3);
    }

    // Boost if many exact matches
    if (statistics.exactMatches > 10) {
      confidence += 0.1;
    }

    // Cap at 1.0
    return Math.min(confidence, 1.0);
  }

  private calculateSequenceSimilarity(seq1: any[], seq2: any[]): number {
    if (seq1.length === 0 && seq2.length === 0) return 1;
    if (seq1.length === 0 || seq2.length === 0) return 0;

    // LCS-based similarity
    const m = seq1.length;
    const n = seq2.length;
    const dp: number[][] = Array(m + 1)
      .fill(null)
      .map(() => Array(n + 1).fill(0));

    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (seq1[i - 1] === seq2[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
    }

    const lcsLength = dp[m][n];
    return (2 * lcsLength) / (m + n);
  }
}

// ============================================================================
// Factory and Exports
// ============================================================================

/**
 * Create a configured AST comparator
 */
export function createASTComparator(
  config?: Partial<ASTComparatorConfig>
): ASTComparator {
  return new ASTComparator(config);
}

/**
 * Default export
 */
export default ASTComparator;
