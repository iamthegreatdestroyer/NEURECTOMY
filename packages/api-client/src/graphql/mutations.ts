/**
 * @fileoverview Type-Safe Mutation Helpers for Neurectomy GraphQL Client
 * @module @neurectomy/api-client/mutations
 *
 * Provides type-safe mutation execution with:
 * - Unified error handling
 * - Optimistic updates support
 * - Retry logic for transient failures
 * - Input validation
 * - Comprehensive result types
 */

// ============================================================================
// Types & Interfaces
// ============================================================================

/**
 * User-facing error from mutations
 */
export interface UserError {
  /** Error message for display */
  message: string;
  /** Field path that caused the error */
  field?: string[];
  /** Error code for programmatic handling */
  code?: string;
}

/**
 * Generic mutation result wrapper
 */
export interface MutationResult<T> {
  /** Success indicator */
  success: boolean;
  /** Result data on success */
  data?: T;
  /** User-facing errors */
  errors: UserError[];
  /** Unique request ID for debugging */
  requestId?: string;
}

/**
 * Mutation execution options
 */
export interface MutationOptions<TVariables = Record<string, unknown>> {
  /** Mutation variables */
  variables?: TVariables;
  /** Operation name for tracing */
  operationName?: string;
  /** Custom headers for this request */
  headers?: Record<string, string>;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Number of retry attempts for transient errors */
  retryCount?: number;
  /** Delay between retries in milliseconds */
  retryDelay?: number;
  /** Custom retry condition */
  shouldRetry?: (error: MutationError) => boolean;
  /** Callback before mutation executes */
  onBefore?: () => void;
  /** Callback after mutation completes */
  onAfter?: (result: MutationResult<unknown>) => void;
  /** Optimistic update callback */
  optimisticUpdate?: () => void;
  /** Rollback callback if mutation fails */
  rollback?: () => void;
}

/**
 * Mutation error types
 */
export type MutationErrorType =
  | "NETWORK_ERROR"
  | "TIMEOUT_ERROR"
  | "VALIDATION_ERROR"
  | "AUTHORIZATION_ERROR"
  | "NOT_FOUND_ERROR"
  | "CONFLICT_ERROR"
  | "RATE_LIMIT_ERROR"
  | "SERVER_ERROR"
  | "UNKNOWN_ERROR";

/**
 * Comprehensive mutation error
 */
export class MutationError extends Error {
  readonly type: MutationErrorType;
  readonly userErrors: UserError[];
  readonly statusCode?: number;
  readonly requestId?: string;
  readonly retryable: boolean;
  readonly originalError?: Error;

  constructor(
    message: string,
    type: MutationErrorType,
    options: {
      userErrors?: UserError[];
      statusCode?: number;
      requestId?: string;
      retryable?: boolean;
      originalError?: Error;
    } = {}
  ) {
    super(message);
    this.name = "MutationError";
    this.type = type;
    this.userErrors = options.userErrors ?? [];
    this.statusCode = options.statusCode;
    this.requestId = options.requestId;
    this.retryable = options.retryable ?? this.isRetryable(type);
    this.originalError = options.originalError;
  }

  private isRetryable(type: MutationErrorType): boolean {
    return [
      "NETWORK_ERROR",
      "TIMEOUT_ERROR",
      "RATE_LIMIT_ERROR",
      "SERVER_ERROR",
    ].includes(type);
  }
}

/**
 * Mutation executor configuration
 */
export interface MutationExecutorConfig {
  /** GraphQL endpoint URL */
  endpoint: string;
  /** Default headers for all requests */
  headers?: Record<string, string>;
  /** Default timeout in milliseconds */
  timeout?: number;
  /** Default retry count */
  retryCount?: number;
  /** Default retry delay in milliseconds */
  retryDelay?: number;
  /** Request interceptor */
  requestInterceptor?: (request: RequestInit) => RequestInit;
  /** Response interceptor */
  responseInterceptor?: (response: Response) => Response;
  /** Error handler */
  onError?: (error: MutationError) => void;
}

// ============================================================================
// Mutation Executor
// ============================================================================

/**
 * Type-safe mutation executor
 */
export class MutationExecutor {
  private config: Required<MutationExecutorConfig>;

  constructor(config: MutationExecutorConfig) {
    this.config = {
      endpoint: config.endpoint,
      headers: config.headers ?? {},
      timeout: config.timeout ?? 30000,
      retryCount: config.retryCount ?? 3,
      retryDelay: config.retryDelay ?? 1000,
      requestInterceptor: config.requestInterceptor ?? ((r) => r),
      responseInterceptor: config.responseInterceptor ?? ((r) => r),
      onError: config.onError ?? (() => {}),
    };
  }

  /**
   * Execute a mutation with full type safety
   */
  async execute<TData, TVariables = Record<string, unknown>>(
    mutation: string,
    options: MutationOptions<TVariables> = {}
  ): Promise<MutationResult<TData>> {
    const {
      variables,
      operationName,
      headers = {},
      timeout = this.config.timeout,
      retryCount = this.config.retryCount,
      retryDelay = this.config.retryDelay,
      shouldRetry,
      onBefore,
      onAfter,
      optimisticUpdate,
      rollback,
    } = options;

    // Execute optimistic update
    if (optimisticUpdate) {
      optimisticUpdate();
    }

    // Pre-mutation callback
    if (onBefore) {
      onBefore();
    }

    let lastError: MutationError | undefined;
    let result: MutationResult<TData> | undefined;

    // Retry loop
    for (let attempt = 0; attempt <= retryCount; attempt++) {
      try {
        result = await this.executeOnce<TData, TVariables>(
          mutation,
          variables,
          operationName,
          { ...this.config.headers, ...headers },
          timeout
        );

        // If we got errors but not a thrown exception, check if we should retry
        if (!result.success && result.errors.length > 0) {
          const error = new MutationError(
            result.errors[0]?.message ?? "Mutation failed",
            this.classifyError(result.errors),
            { userErrors: result.errors }
          );

          if (
            attempt < retryCount &&
            (shouldRetry?.(error) ?? error.retryable)
          ) {
            lastError = error;
            await this.delay(retryDelay * Math.pow(2, attempt));
            continue;
          }
        }

        break;
      } catch (error) {
        lastError = this.wrapError(error);

        if (
          attempt < retryCount &&
          (shouldRetry?.(lastError) ?? lastError.retryable)
        ) {
          await this.delay(retryDelay * Math.pow(2, attempt));
          continue;
        }

        // Rollback optimistic update on failure
        if (rollback) {
          rollback();
        }

        this.config.onError(lastError);

        result = {
          success: false,
          errors:
            lastError.userErrors.length > 0
              ? lastError.userErrors
              : [{ message: lastError.message }],
        };
        break;
      }
    }

    // Post-mutation callback
    if (onAfter && result) {
      onAfter(result);
    }

    return result!;
  }

  /**
   * Execute mutation once without retry logic
   */
  private async executeOnce<TData, TVariables>(
    mutation: string,
    variables: TVariables | undefined,
    operationName: string | undefined,
    headers: Record<string, string>,
    timeout: number
  ): Promise<MutationResult<TData>> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      let request: RequestInit = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...headers,
        },
        body: JSON.stringify({
          query: mutation,
          variables,
          operationName,
        }),
        signal: controller.signal,
      };

      request = this.config.requestInterceptor(request);

      let response = await fetch(this.config.endpoint, request);
      response = this.config.responseInterceptor(response);

      if (!response.ok) {
        throw new MutationError(
          `HTTP error: ${response.status} ${response.statusText}`,
          this.httpStatusToErrorType(response.status),
          { statusCode: response.status }
        );
      }

      const json = await response.json();

      // Handle GraphQL errors
      if (json.errors && json.errors.length > 0) {
        const userErrors = this.extractUserErrors(json.errors);
        return {
          success: false,
          errors: userErrors,
          requestId: response.headers.get("x-request-id") ?? undefined,
        };
      }

      // Extract mutation result from data
      const mutationResult = this.extractMutationResult<TData>(json.data);

      return {
        success: mutationResult.success,
        data: mutationResult.data,
        errors: mutationResult.errors,
        requestId: response.headers.get("x-request-id") ?? undefined,
      };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Extract mutation result from GraphQL response
   */
  private extractMutationResult<TData>(
    data: Record<string, unknown>
  ): MutationResult<TData> {
    // GraphQL mutations typically return an object with the mutation name as key
    const mutationKey = Object.keys(data)[0];
    const result = data[mutationKey] as Record<string, unknown>;

    if (!result) {
      return { success: false, errors: [{ message: "No result returned" }] };
    }

    // Check for standard mutation response fields
    if ("success" in result && "errors" in result) {
      return {
        success: result.success as boolean,
        data: result.data as TData | undefined,
        errors: (result.errors as UserError[]) ?? [],
      };
    }

    // Check for userErrors field (common pattern)
    if ("userErrors" in result && Array.isArray(result.userErrors)) {
      const userErrors = result.userErrors as UserError[];
      return {
        success: userErrors.length === 0,
        data: result as TData,
        errors: userErrors,
      };
    }

    // Assume success if we got data without errors
    return {
      success: true,
      data: result as TData,
      errors: [],
    };
  }

  /**
   * Extract user errors from GraphQL errors
   */
  private extractUserErrors(
    errors: Array<{
      message: string;
      extensions?: Record<string, unknown>;
      path?: string[];
    }>
  ): UserError[] {
    return errors.map((error) => ({
      message: error.message,
      field: error.path,
      code: error.extensions?.code as string | undefined,
    }));
  }

  /**
   * Classify error type from user errors
   */
  private classifyError(errors: UserError[]): MutationErrorType {
    const codes = errors.map((e) => e.code).filter(Boolean);

    if (
      codes.includes("VALIDATION_ERROR") ||
      codes.includes("BAD_USER_INPUT")
    ) {
      return "VALIDATION_ERROR";
    }
    if (codes.includes("UNAUTHORIZED") || codes.includes("FORBIDDEN")) {
      return "AUTHORIZATION_ERROR";
    }
    if (codes.includes("NOT_FOUND")) {
      return "NOT_FOUND_ERROR";
    }
    if (codes.includes("CONFLICT")) {
      return "CONFLICT_ERROR";
    }
    if (codes.includes("RATE_LIMITED")) {
      return "RATE_LIMIT_ERROR";
    }
    if (codes.includes("INTERNAL_SERVER_ERROR")) {
      return "SERVER_ERROR";
    }

    return "UNKNOWN_ERROR";
  }

  /**
   * Map HTTP status to error type
   */
  private httpStatusToErrorType(status: number): MutationErrorType {
    if (status === 400) return "VALIDATION_ERROR";
    if (status === 401 || status === 403) return "AUTHORIZATION_ERROR";
    if (status === 404) return "NOT_FOUND_ERROR";
    if (status === 409) return "CONFLICT_ERROR";
    if (status === 429) return "RATE_LIMIT_ERROR";
    if (status >= 500) return "SERVER_ERROR";
    return "UNKNOWN_ERROR";
  }

  /**
   * Wrap unknown error in MutationError
   */
  private wrapError(error: unknown): MutationError {
    if (error instanceof MutationError) {
      return error;
    }

    if (error instanceof Error) {
      if (error.name === "AbortError") {
        return new MutationError("Request timed out", "TIMEOUT_ERROR", {
          originalError: error,
        });
      }

      if (
        error.message.includes("fetch") ||
        error.message.includes("network")
      ) {
        return new MutationError("Network error", "NETWORK_ERROR", {
          originalError: error,
        });
      }

      return new MutationError(error.message, "UNKNOWN_ERROR", {
        originalError: error,
      });
    }

    return new MutationError("Unknown error occurred", "UNKNOWN_ERROR");
  }

  /**
   * Delay helper
   */
  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Create a mutation executor instance
 */
export function createMutationExecutor(
  config: MutationExecutorConfig
): MutationExecutor {
  return new MutationExecutor(config);
}

/**
 * Execute a mutation with the global executor
 */
let globalExecutor: MutationExecutor | null = null;

export function setGlobalMutationExecutor(executor: MutationExecutor): void {
  globalExecutor = executor;
}

export async function executeMutation<
  TData,
  TVariables = Record<string, unknown>,
>(
  mutation: string,
  options?: MutationOptions<TVariables>
): Promise<MutationResult<TData>> {
  if (!globalExecutor) {
    throw new Error(
      "Global mutation executor not initialized. Call setGlobalMutationExecutor first."
    );
  }
  return globalExecutor.execute<TData, TVariables>(mutation, options);
}

// ============================================================================
// Type-Safe Mutation Builders
// ============================================================================

/**
 * Builder for type-safe mutation definitions
 */
export function defineMutation<TData, TVariables = Record<string, unknown>>(
  mutation: string,
  defaultOptions?: Partial<MutationOptions<TVariables>>
) {
  return {
    /**
     * Execute the mutation
     */
    execute: async (
      executor: MutationExecutor,
      variables?: TVariables,
      options?: Partial<MutationOptions<TVariables>>
    ): Promise<MutationResult<TData>> => {
      return executor.execute<TData, TVariables>(mutation, {
        ...defaultOptions,
        ...options,
        variables: { ...defaultOptions?.variables, ...variables } as TVariables,
      });
    },

    /**
     * Get the raw mutation string
     */
    toGraphQL: () => mutation,
  };
}

// ============================================================================
// Pre-defined Common Mutations
// ============================================================================

/**
 * Common mutation definitions
 */
export const MUTATIONS = {
  // Agent mutations
  CREATE_AGENT: defineMutation<
    { agent: { id: string; name: string } },
    { input: { name: string; description?: string; capabilities?: string[] } }
  >(`
    mutation CreateAgent($input: CreateAgentInput!) {
      createAgent(input: $input) {
        success
        errors { message field code }
        agent {
          id
          name
          description
          capabilities { id name }
        }
      }
    }
  `),

  UPDATE_AGENT: defineMutation<
    { agent: { id: string } },
    {
      id: string;
      input: { name?: string; description?: string; status?: string };
    }
  >(`
    mutation UpdateAgent($id: ID!, $input: UpdateAgentInput!) {
      updateAgent(id: $id, input: $input) {
        success
        errors { message field code }
        agent {
          id
          name
          status
        }
      }
    }
  `),

  DELETE_AGENT: defineMutation<{ deletedId: string }, { id: string }>(`
    mutation DeleteAgent($id: ID!) {
      deleteAgent(id: $id) {
        success
        errors { message field code }
        deletedId
      }
    }
  `),

  // Workflow mutations
  CREATE_WORKFLOW: defineMutation<
    { workflow: { id: string; name: string } },
    {
      input: {
        name: string;
        description?: string;
        nodes: unknown[];
        edges: unknown[];
      };
    }
  >(`
    mutation CreateWorkflow($input: CreateWorkflowInput!) {
      createWorkflow(input: $input) {
        success
        errors { message field code }
        workflow {
          id
          name
          version
        }
      }
    }
  `),

  EXECUTE_WORKFLOW: defineMutation<
    { execution: { id: string; status: string } },
    { workflowId: string; input?: Record<string, unknown> }
  >(`
    mutation ExecuteWorkflow($workflowId: ID!, $input: JSON) {
      executeWorkflow(workflowId: $workflowId, input: $input) {
        success
        errors { message field code }
        execution {
          id
          status
          startedAt
        }
      }
    }
  `),

  // Conversation mutations
  CREATE_CONVERSATION: defineMutation<
    { conversation: { id: string; title: string } },
    {
      input: {
        title?: string;
        agentId: string;
        metadata?: Record<string, unknown>;
      };
    }
  >(`
    mutation CreateConversation($input: CreateConversationInput!) {
      createConversation(input: $input) {
        success
        errors { message field code }
        conversation {
          id
          title
          createdAt
        }
      }
    }
  `),

  SEND_MESSAGE: defineMutation<
    { message: { id: string; content: string } },
    { conversationId: string; content: string; role?: string }
  >(`
    mutation SendMessage($conversationId: ID!, $content: String!, $role: MessageRole) {
      sendMessage(conversationId: $conversationId, content: $content, role: $role) {
        success
        errors { message field code }
        message {
          id
          content
          role
          timestamp
        }
      }
    }
  `),

  // Knowledge base mutations
  UPLOAD_DOCUMENT: defineMutation<
    { document: { id: string; title: string } },
    {
      input: {
        title: string;
        content: string;
        type: string;
        metadata?: Record<string, unknown>;
      };
    }
  >(`
    mutation UploadDocument($input: UploadDocumentInput!) {
      uploadDocument(input: $input) {
        success
        errors { message field code }
        document {
          id
          title
          status
        }
      }
    }
  `),

  DELETE_DOCUMENT: defineMutation<{ deletedId: string }, { id: string }>(`
    mutation DeleteDocument($id: ID!) {
      deleteDocument(id: $id) {
        success
        errors { message field code }
        deletedId
      }
    }
  `),

  // User mutations
  UPDATE_USER_PREFERENCES: defineMutation<
    { user: { id: string; preferences: Record<string, unknown> } },
    { preferences: Record<string, unknown> }
  >(`
    mutation UpdateUserPreferences($preferences: JSON!) {
      updateUserPreferences(preferences: $preferences) {
        success
        errors { message field code }
        user {
          id
          preferences
        }
      }
    }
  `),
} as const;

// ============================================================================
// Validation Helpers
// ============================================================================

/**
 * Validate mutation input before execution
 */
export function validateInput<T>(
  input: T,
  schema: {
    required?: (keyof T)[];
    validators?: Partial<Record<keyof T, (value: unknown) => boolean>>;
  }
): UserError[] {
  const errors: UserError[] = [];

  // Check required fields
  if (schema.required) {
    for (const field of schema.required) {
      if (
        input[field] === undefined ||
        input[field] === null ||
        input[field] === ""
      ) {
        errors.push({
          message: `${String(field)} is required`,
          field: [String(field)],
          code: "REQUIRED_FIELD",
        });
      }
    }
  }

  // Run custom validators
  if (schema.validators) {
    for (const [field, validator] of Object.entries(schema.validators)) {
      const value = input[field as keyof T];
      if (value !== undefined && validator && !validator(value)) {
        errors.push({
          message: `Invalid value for ${field}`,
          field: [field],
          code: "INVALID_VALUE",
        });
      }
    }
  }

  return errors;
}

/**
 * Create a validated mutation executor
 */
export function withValidation<
  TData,
  TVariables extends Record<string, unknown>,
>(
  executor: MutationExecutor,
  mutation: string,
  validationSchema: {
    required?: (keyof TVariables)[];
    validators?: Partial<Record<keyof TVariables, (value: unknown) => boolean>>;
  }
) {
  return async (
    variables: TVariables,
    options?: Omit<MutationOptions<TVariables>, "variables">
  ): Promise<MutationResult<TData>> => {
    // Validate input
    const validationErrors = validateInput(variables, validationSchema);
    if (validationErrors.length > 0) {
      return {
        success: false,
        errors: validationErrors,
      };
    }

    return executor.execute<TData, TVariables>(mutation, {
      ...options,
      variables,
    });
  };
}
