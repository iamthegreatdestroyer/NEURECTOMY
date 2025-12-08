/**
 * useMutation Hook
 *
 * Custom React hook for GraphQL mutations with TypeScript support.
 * Wraps URQL's useMutation with enhanced error handling and optimistic updates.
 */

import { useMutation as useUrqlMutation, UseMutationState } from "urql";
import { useCallback, useEffect, useRef } from "react";

export interface UseMutationOptions<TData = any, TVariables = object> {
  mutation: string;
  onSuccess?: (data: TData, variables: TVariables) => void;
  onError?: (error: Error, variables: TVariables) => void;
  onSettled?: (
    data: TData | undefined,
    error: Error | undefined,
    variables: TVariables
  ) => void;
}

export interface UseMutationResult<TData = any, TVariables = object> {
  data: TData | undefined;
  fetching: boolean;
  error: Error | undefined;
  executeMutation: (
    variables: TVariables,
    context?: any
  ) => Promise<UseMutationState<TData, TVariables>>;
  reset: () => void;
}

/**
 * Custom useMutation hook with enhanced features
 *
 * @example
 * ```tsx
 * const { executeMutation, fetching, error } = useMutation<CreateUserResult, CreateUserInput>({
 *   mutation: `
 *     mutation CreateUser($input: CreateUserInput!) {
 *       createUser(input: $input) {
 *         id
 *         name
 *         email
 *       }
 *     }
 *   `,
 *   onSuccess: (data) => {
 *     console.log('User created:', data.createUser);
 *     navigate(`/users/${data.createUser.id}`);
 *   },
 *   onError: (error) => {
 *     toast.error(`Failed to create user: ${error.message}`);
 *   },
 * });
 *
 * const handleSubmit = async (formData: CreateUserInput) => {
 *   await executeMutation({ input: formData });
 * };
 * ```
 */
export function useMutation<TData = any, TVariables extends object = object>({
  mutation,
  onSuccess,
  onError,
  onSettled,
}: UseMutationOptions<TData, TVariables>): UseMutationResult<
  TData,
  TVariables
> {
  // Use URQL's useMutation
  const [result, executeUrqlMutation] = useUrqlMutation<TData, TVariables>(
    mutation
  );

  // Track last executed variables for callbacks
  const lastVariablesRef = useRef<TVariables | undefined>();

  // Call callbacks when mutation completes
  useEffect(() => {
    if (!result.fetching && lastVariablesRef.current) {
      const variables = lastVariablesRef.current;

      if (result.data && !result.error) {
        onSuccess?.(result.data, variables);
      }

      if (result.error) {
        const error = new Error(result.error.message);
        error.name = result.error.name;
        (error as any).graphQLErrors = result.error.graphQLErrors;
        (error as any).networkError = result.error.networkError;
        onError?.(error, variables);
      }

      onSettled?.(result.data, result.error as Error | undefined, variables);

      // Clear variables after callbacks
      lastVariablesRef.current = undefined;
    }
  }, [
    result.data,
    result.error,
    result.fetching,
    onSuccess,
    onError,
    onSettled,
  ]);

  // Enhanced execute function with promise interface
  const executeMutation = useCallback(
    async (
      variables: TVariables,
      context?: any
    ): Promise<UseMutationState<TData, TVariables>> => {
      lastVariablesRef.current = variables;

      const mutationResult = await executeUrqlMutation(variables, context);

      return mutationResult;
    },
    [executeUrqlMutation]
  );

  // Reset function to clear mutation state
  const reset = useCallback(() => {
    lastVariablesRef.current = undefined;
  }, []);

  return {
    data: result.data,
    fetching: result.fetching,
    error: result.error as Error | undefined,
    executeMutation,
    reset,
  };
}

/**
 * Hook for mutations with optimistic updates
 *
 * @example
 * ```tsx
 * const { executeMutation } = useOptimisticMutation<UpdateUserResult, UpdateUserInput>({
 *   mutation: UPDATE_USER_MUTATION,
 *   optimisticData: (variables) => ({
 *     updateUser: {
 *       id: variables.id,
 *       name: variables.input.name,
 *       __typename: 'User',
 *     },
 *   }),
 *   onSuccess: (data) => {
 *     toast.success('User updated successfully');
 *   },
 * });
 * ```
 */
export interface UseOptimisticMutationOptions<
  TData = any,
  TVariables = object,
> extends UseMutationOptions<TData, TVariables> {
  optimisticData: (variables: TVariables) => TData;
}

export function useOptimisticMutation<
  TData = any,
  TVariables extends object = object,
>({
  mutation,
  optimisticData,
  onSuccess,
  onError,
  onSettled,
}: UseOptimisticMutationOptions<TData, TVariables>): UseMutationResult<
  TData,
  TVariables
> {
  const result = useMutation<TData, TVariables>({
    mutation,
    onSuccess,
    onError,
    onSettled,
  });

  const executeMutationWithOptimistic = useCallback(
    async (
      variables: TVariables
    ): Promise<UseMutationState<TData, TVariables>> => {
      const context = {
        optimistic: optimisticData(variables),
      };

      return result.executeMutation(variables, context);
    },
    [result.executeMutation, optimisticData]
  );

  return {
    ...result,
    executeMutation: executeMutationWithOptimistic,
  };
}

export type { UseMutationState };
