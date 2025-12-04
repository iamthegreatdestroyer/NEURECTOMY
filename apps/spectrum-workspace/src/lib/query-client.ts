/**
 * TanStack Query Configuration
 * React Query setup with default options and query keys
 */

import { QueryClient } from '@tanstack/react-query';

// Create query client with defaults
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Stale time - how long data is considered fresh (5 minutes)
      staleTime: 5 * 60 * 1000,
      
      // Cache time - how long inactive data is kept (30 minutes)
      gcTime: 30 * 60 * 1000,
      
      // Retry configuration
      retry: (failureCount, error) => {
        // Don't retry on 4xx errors
        if (error instanceof Error && 'status' in error) {
          const status = (error as any).status;
          if (status >= 400 && status < 500) {
            return false;
          }
        }
        return failureCount < 3;
      },
      
      // Retry delay with exponential backoff
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      
      // Refetch on window focus for fresh data
      refetchOnWindowFocus: true,
      
      // Don't refetch on reconnect by default
      refetchOnReconnect: 'always',
    },
    mutations: {
      // Retry mutations once on failure
      retry: 1,
    },
  },
});

// Query key factory for type-safe and consistent keys
export const queryKeys = {
  // Agents
  agents: {
    all: ['agents'] as const,
    lists: () => [...queryKeys.agents.all, 'list'] as const,
    list: (filters: Record<string, unknown>) => [...queryKeys.agents.lists(), filters] as const,
    details: () => [...queryKeys.agents.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.agents.details(), id] as const,
    metrics: (id: string) => [...queryKeys.agents.detail(id), 'metrics'] as const,
    conversations: (id: string) => [...queryKeys.agents.detail(id), 'conversations'] as const,
  },
  
  // Containers
  containers: {
    all: ['containers'] as const,
    lists: () => [...queryKeys.containers.all, 'list'] as const,
    list: (filters?: Record<string, unknown>) => 
      filters ? [...queryKeys.containers.lists(), filters] as const : queryKeys.containers.lists(),
    details: () => [...queryKeys.containers.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.containers.details(), id] as const,
    logs: (id: string, options?: { tail?: number }) => 
      [...queryKeys.containers.detail(id), 'logs', options] as const,
    stats: (id: string) => [...queryKeys.containers.detail(id), 'stats'] as const,
  },
  
  // Clusters
  clusters: {
    all: ['clusters'] as const,
    lists: () => [...queryKeys.clusters.all, 'list'] as const,
    list: () => queryKeys.clusters.lists(),
    details: () => [...queryKeys.clusters.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.clusters.details(), id] as const,
    pods: (id: string, namespace?: string) => 
      [...queryKeys.clusters.detail(id), 'pods', namespace] as const,
    deployments: (id: string, namespace?: string) => 
      [...queryKeys.clusters.detail(id), 'deployments', namespace] as const,
  },
  
  // Models
  models: {
    all: ['models'] as const,
    lists: () => [...queryKeys.models.all, 'list'] as const,
    list: (filters?: Record<string, unknown>) => 
      filters ? [...queryKeys.models.lists(), filters] as const : queryKeys.models.lists(),
    details: () => [...queryKeys.models.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.models.details(), id] as const,
    versions: (id: string) => [...queryKeys.models.detail(id), 'versions'] as const,
  },
  
  // Training
  training: {
    all: ['training'] as const,
    jobs: () => [...queryKeys.training.all, 'jobs'] as const,
    job: (id: string) => [...queryKeys.training.jobs(), id] as const,
  },
  
  // Research
  research: {
    all: ['research'] as const,
    search: (query: string) => [...queryKeys.research.all, 'search', query] as const,
    documents: () => [...queryKeys.research.all, 'documents'] as const,
    document: (id: string) => [...queryKeys.research.documents(), id] as const,
  },
  
  // System
  system: {
    health: ['system', 'health'] as const,
    metrics: ['system', 'metrics'] as const,
  },
} as const;

// Invalidation helpers
export const invalidateQueries = {
  agents: () => queryClient.invalidateQueries({ queryKey: queryKeys.agents.all }),
  agent: (id: string) => queryClient.invalidateQueries({ queryKey: queryKeys.agents.detail(id) }),
  
  containers: () => queryClient.invalidateQueries({ queryKey: queryKeys.containers.all }),
  container: (id: string) => queryClient.invalidateQueries({ queryKey: queryKeys.containers.detail(id) }),
  
  clusters: () => queryClient.invalidateQueries({ queryKey: queryKeys.clusters.all }),
  cluster: (id: string) => queryClient.invalidateQueries({ queryKey: queryKeys.clusters.detail(id) }),
  
  models: () => queryClient.invalidateQueries({ queryKey: queryKeys.models.all }),
  model: (id: string) => queryClient.invalidateQueries({ queryKey: queryKeys.models.detail(id) }),
  
  training: () => queryClient.invalidateQueries({ queryKey: queryKeys.training.all }),
  
  all: () => queryClient.invalidateQueries(),
};

export default queryClient;
