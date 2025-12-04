/**
 * API Hooks
 * React Query hooks for data fetching with caching and real-time updates
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { agentsApi, executionsApi, workflowsApi, systemApi, userApi } from '../lib/graphql';
import type { Agent, Execution, Workflow, DashboardStats, SystemStatus } from '../lib/graphql';

// Query keys for cache management
export const queryKeys = {
  agents: {
    all: ['agents'] as const,
    lists: () => [...queryKeys.agents.all, 'list'] as const,
    list: (filters: Record<string, unknown>) => [...queryKeys.agents.lists(), filters] as const,
    details: () => [...queryKeys.agents.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.agents.details(), id] as const,
  },
  executions: {
    all: ['executions'] as const,
    lists: () => [...queryKeys.executions.all, 'list'] as const,
    list: (filters: Record<string, unknown>) => [...queryKeys.executions.lists(), filters] as const,
    details: () => [...queryKeys.executions.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.executions.details(), id] as const,
  },
  workflows: {
    all: ['workflows'] as const,
    lists: () => [...queryKeys.workflows.all, 'list'] as const,
    list: (filters: Record<string, unknown>) => [...queryKeys.workflows.lists(), filters] as const,
    details: () => [...queryKeys.workflows.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.workflows.details(), id] as const,
  },
  system: {
    status: ['system', 'status'] as const,
    dashboard: ['system', 'dashboard'] as const,
  },
  user: {
    me: ['user', 'me'] as const,
  },
};

// ============================================================================
// AGENT HOOKS
// ============================================================================

interface UseAgentsOptions {
  status?: string;
  tier?: number;
  limit?: number;
  offset?: number;
  enabled?: boolean;
}

export function useAgents(options: UseAgentsOptions = {}) {
  const { enabled = true, ...params } = options;
  
  return useQuery({
    queryKey: queryKeys.agents.list(params),
    queryFn: () => agentsApi.getAll(params),
    enabled,
    staleTime: 30_000, // 30 seconds
    refetchInterval: 60_000, // 1 minute
  });
}

export function useAgent(id: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.agents.detail(id),
    queryFn: () => agentsApi.getById(id),
    enabled: enabled && !!id,
    staleTime: 30_000,
  });
}

export function useCreateAgent() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (input: Partial<Agent>) => agentsApi.create(input),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents.lists() });
    },
  });
}

export function useUpdateAgent() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ id, input }: { id: string; input: Partial<Agent> }) => 
      agentsApi.update(id, input),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents.lists() });
      queryClient.setQueryData(queryKeys.agents.detail(data.id), data);
    },
  });
}

export function useDeleteAgent() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => agentsApi.delete(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents.lists() });
      queryClient.removeQueries({ queryKey: queryKeys.agents.detail(id) });
    },
  });
}

export function useStartAgent() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => agentsApi.start(id),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents.lists() });
      queryClient.setQueryData(
        queryKeys.agents.detail(data.id),
        (old: Agent | undefined) => old ? { ...old, status: data.status } : undefined
      );
    },
  });
}

export function useStopAgent() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => agentsApi.stop(id),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents.lists() });
      queryClient.setQueryData(
        queryKeys.agents.detail(data.id),
        (old: Agent | undefined) => old ? { ...old, status: data.status } : undefined
      );
    },
  });
}

export function useExecuteAgent() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ agentId, input, stream = false }: { agentId: string; input: unknown; stream?: boolean }) =>
      agentsApi.execute(agentId, input, stream),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.executions.lists() });
    },
  });
}

// ============================================================================
// EXECUTION HOOKS
// ============================================================================

interface UseExecutionsOptions {
  agentId?: string;
  status?: string;
  limit?: number;
  offset?: number;
  enabled?: boolean;
}

export function useExecutions(options: UseExecutionsOptions = {}) {
  const { enabled = true, ...params } = options;
  
  return useQuery({
    queryKey: queryKeys.executions.list(params),
    queryFn: () => executionsApi.getAll(params),
    enabled,
    staleTime: 10_000, // 10 seconds
    refetchInterval: 30_000, // 30 seconds
  });
}

export function useExecution(id: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.executions.detail(id),
    queryFn: () => executionsApi.getById(id),
    enabled: enabled && !!id,
    staleTime: 5_000, // Refresh more frequently for active executions
    refetchInterval: (query) => {
      const data = query.state.data as Execution | undefined;
      // Refresh every 2 seconds if execution is running
      return data?.status === 'running' ? 2_000 : false;
    },
  });
}

export function useCancelExecution() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => executionsApi.cancel(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.executions.lists() });
      queryClient.invalidateQueries({ queryKey: queryKeys.executions.detail(id) });
    },
  });
}

// ============================================================================
// WORKFLOW HOOKS
// ============================================================================

interface UseWorkflowsOptions {
  status?: string;
  limit?: number;
  offset?: number;
  enabled?: boolean;
}

export function useWorkflows(options: UseWorkflowsOptions = {}) {
  const { enabled = true, ...params } = options;
  
  return useQuery({
    queryKey: queryKeys.workflows.list(params),
    queryFn: () => workflowsApi.getAll(params),
    enabled,
    staleTime: 30_000,
    refetchInterval: 60_000,
  });
}

export function useWorkflow(id: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.workflows.detail(id),
    queryFn: () => workflowsApi.getById(id),
    enabled: enabled && !!id,
    staleTime: 30_000,
  });
}

export function useCreateWorkflow() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (input: Partial<Workflow>) => workflowsApi.create(input),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.workflows.lists() });
    },
  });
}

export function useUpdateWorkflow() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ id, input }: { id: string; input: Partial<Workflow> }) =>
      workflowsApi.update(id, input),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.workflows.lists() });
      queryClient.setQueryData(queryKeys.workflows.detail(data.id), data);
    },
  });
}

export function useDeleteWorkflow() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => workflowsApi.delete(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.workflows.lists() });
      queryClient.removeQueries({ queryKey: queryKeys.workflows.detail(id) });
    },
  });
}

export function useExecuteWorkflow() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ id, input }: { id: string; input?: unknown }) =>
      workflowsApi.execute(id, input),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.executions.lists() });
    },
  });
}

// ============================================================================
// SYSTEM HOOKS
// ============================================================================

export function useSystemStatus(enabled = true) {
  return useQuery({
    queryKey: queryKeys.system.status,
    queryFn: () => systemApi.getStatus(),
    enabled,
    staleTime: 10_000,
    refetchInterval: 30_000,
  });
}

export function useDashboardStats(enabled = true) {
  return useQuery({
    queryKey: queryKeys.system.dashboard,
    queryFn: () => systemApi.getDashboardStats(),
    enabled,
    staleTime: 30_000,
    refetchInterval: 60_000,
  });
}

// ============================================================================
// USER HOOKS
// ============================================================================

export function useCurrentUser(enabled = true) {
  return useQuery({
    queryKey: queryKeys.user.me,
    queryFn: () => userApi.getMe(),
    enabled,
    staleTime: 300_000, // 5 minutes
  });
}

export function useUpdatePreferences() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (preferences: unknown) => userApi.updatePreferences(preferences),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.user.me, (old: ReturnType<typeof userApi.getMe> | undefined) =>
        old ? { ...old, preferences: data.preferences } : undefined
      );
    },
  });
}

// ============================================================================
// REALTIME UPDATES HOOK
// ============================================================================

import { useEffect } from 'react';
import { useWebSocket } from './useWebSocket';

/**
 * Hook to sync WebSocket updates with React Query cache
 */
export function useRealtimeSync() {
  const queryClient = useQueryClient();
  
  const { lastMessage } = useWebSocket({
    onMessage: (message) => {
      switch (message.type) {
        case 'agent_status_changed':
          // Update agent in cache
          const agentUpdate = message.payload as { id: string; status: string };
          queryClient.setQueryData(
            queryKeys.agents.detail(agentUpdate.id),
            (old: Agent | undefined) => old ? { ...old, status: agentUpdate.status } : undefined
          );
          queryClient.invalidateQueries({ queryKey: queryKeys.agents.lists() });
          break;
          
        case 'execution_progress':
          // Update execution in cache
          const execUpdate = message.payload as { executionId: string; status: string; progress: number };
          queryClient.setQueryData(
            queryKeys.executions.detail(execUpdate.executionId),
            (old: Execution | undefined) => old ? { ...old, status: execUpdate.status } : undefined
          );
          break;
          
        case 'execution_completed':
          // Invalidate related queries
          queryClient.invalidateQueries({ queryKey: queryKeys.executions.lists() });
          queryClient.invalidateQueries({ queryKey: queryKeys.system.dashboard });
          break;
          
        case 'system_event':
          // Refresh system status on important events
          queryClient.invalidateQueries({ queryKey: queryKeys.system.status });
          break;
      }
    },
  });
  
  return { lastMessage };
}

// ============================================================================
// PREFETCH UTILITIES
// ============================================================================

/**
 * Prefetch agent data for faster navigation
 */
export function usePrefetchAgent() {
  const queryClient = useQueryClient();
  
  return (id: string) => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.agents.detail(id),
      queryFn: () => agentsApi.getById(id),
      staleTime: 30_000,
    });
  };
}

/**
 * Prefetch workflow data for faster navigation
 */
export function usePrefetchWorkflow() {
  const queryClient = useQueryClient();
  
  return (id: string) => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.workflows.detail(id),
      queryFn: () => workflowsApi.getById(id),
      staleTime: 30_000,
    });
  };
}

/**
 * Prefetch dashboard data on app load
 */
export function usePrefetchDashboard() {
  const queryClient = useQueryClient();
  
  useEffect(() => {
    // Prefetch commonly needed data
    queryClient.prefetchQuery({
      queryKey: queryKeys.system.status,
      queryFn: () => systemApi.getStatus(),
    });
    
    queryClient.prefetchQuery({
      queryKey: queryKeys.system.dashboard,
      queryFn: () => systemApi.getDashboardStats(),
    });
    
    queryClient.prefetchQuery({
      queryKey: queryKeys.agents.list({}),
      queryFn: () => agentsApi.getAll(),
    });
  }, [queryClient]);
}
