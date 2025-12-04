/**
 * GraphQL Client Configuration
 * Type-safe GraphQL client for Rust Core backend
 */

import { GraphQLClient, gql } from 'graphql-request';

// Configuration
const GRAPHQL_URL = import.meta.env.VITE_GRAPHQL_URL || 'http://localhost:8080/graphql';

// Create GraphQL client instance
export const graphqlClient = new GraphQLClient(GRAPHQL_URL, {
  headers: () => {
    const token = localStorage.getItem('auth_token');
    return token ? { Authorization: `Bearer ${token}` } : {};
  },
});

// ============================================================================
// AGENT QUERIES & MUTATIONS
// ============================================================================

// Fragments
export const AGENT_FRAGMENT = gql`
  fragment AgentFields on Agent {
    id
    name
    description
    agentType
    status
    model
    tier
    capabilities
    config
    createdAt
    updatedAt
  }
`;

export const AGENT_METRICS_FRAGMENT = gql`
  fragment AgentMetricsFields on AgentMetrics {
    tasksCompleted
    successRate
    averageResponseTimeMs
    tokensUsed
    errorCount
    lastActiveAt
  }
`;

// Queries
export const GET_AGENTS = gql`
  ${AGENT_FRAGMENT}
  query GetAgents($status: AgentStatus, $tier: Int, $limit: Int, $offset: Int) {
    agents(status: $status, tier: $tier, limit: $limit, offset: $offset) {
      ...AgentFields
    }
  }
`;

export const GET_AGENT = gql`
  ${AGENT_FRAGMENT}
  ${AGENT_METRICS_FRAGMENT}
  query GetAgent($id: ID!) {
    agent(id: $id) {
      ...AgentFields
      metrics {
        ...AgentMetricsFields
      }
      recentExecutions(limit: 10) {
        id
        status
        startedAt
        completedAt
        durationMs
      }
    }
  }
`;

export const GET_AGENT_TEMPLATES = gql`
  query GetAgentTemplates {
    agentTemplates {
      id
      name
      description
      tier
      icon
      capabilities
      defaultConfig
    }
  }
`;

// Mutations
export const CREATE_AGENT = gql`
  ${AGENT_FRAGMENT}
  mutation CreateAgent($input: CreateAgentInput!) {
    createAgent(input: $input) {
      ...AgentFields
    }
  }
`;

export const UPDATE_AGENT = gql`
  ${AGENT_FRAGMENT}
  mutation UpdateAgent($id: ID!, $input: UpdateAgentInput!) {
    updateAgent(id: $id, input: $input) {
      ...AgentFields
    }
  }
`;

export const DELETE_AGENT = gql`
  mutation DeleteAgent($id: ID!) {
    deleteAgent(id: $id)
  }
`;

export const START_AGENT = gql`
  mutation StartAgent($id: ID!) {
    startAgent(id: $id) {
      id
      status
    }
  }
`;

export const STOP_AGENT = gql`
  mutation StopAgent($id: ID!) {
    stopAgent(id: $id) {
      id
      status
    }
  }
`;

// ============================================================================
// EXECUTION QUERIES & MUTATIONS
// ============================================================================

export const EXECUTION_FRAGMENT = gql`
  fragment ExecutionFields on Execution {
    id
    agentId
    status
    input
    output
    errorMessage
    startedAt
    completedAt
    durationMs
    tokensUsed
  }
`;

export const GET_EXECUTIONS = gql`
  ${EXECUTION_FRAGMENT}
  query GetExecutions($agentId: ID, $status: ExecutionStatus, $limit: Int, $offset: Int) {
    executions(agentId: $agentId, status: $status, limit: $limit, offset: $offset) {
      ...ExecutionFields
      agent {
        id
        name
      }
    }
  }
`;

export const GET_EXECUTION = gql`
  ${EXECUTION_FRAGMENT}
  query GetExecution($id: ID!) {
    execution(id: $id) {
      ...ExecutionFields
      agent {
        id
        name
        model
      }
      steps {
        id
        stepNumber
        name
        status
        input
        output
        startedAt
        completedAt
        durationMs
      }
    }
  }
`;

export const EXECUTE_AGENT = gql`
  ${EXECUTION_FRAGMENT}
  mutation ExecuteAgent($agentId: ID!, $input: JSON!, $stream: Boolean) {
    executeAgent(agentId: $agentId, input: $input, stream: $stream) {
      ...ExecutionFields
    }
  }
`;

export const CANCEL_EXECUTION = gql`
  mutation CancelExecution($id: ID!) {
    cancelExecution(id: $id)
  }
`;

// ============================================================================
// WORKFLOW QUERIES & MUTATIONS
// ============================================================================

export const WORKFLOW_FRAGMENT = gql`
  fragment WorkflowFields on Workflow {
    id
    name
    description
    status
    nodes
    edges
    createdAt
    updatedAt
  }
`;

export const GET_WORKFLOWS = gql`
  ${WORKFLOW_FRAGMENT}
  query GetWorkflows($status: WorkflowStatus, $limit: Int, $offset: Int) {
    workflows(status: $status, limit: $limit, offset: $offset) {
      ...WorkflowFields
    }
  }
`;

export const GET_WORKFLOW = gql`
  ${WORKFLOW_FRAGMENT}
  query GetWorkflow($id: ID!) {
    workflow(id: $id) {
      ...WorkflowFields
      executions(limit: 10) {
        id
        status
        startedAt
        completedAt
      }
    }
  }
`;

export const CREATE_WORKFLOW = gql`
  ${WORKFLOW_FRAGMENT}
  mutation CreateWorkflow($input: CreateWorkflowInput!) {
    createWorkflow(input: $input) {
      ...WorkflowFields
    }
  }
`;

export const UPDATE_WORKFLOW = gql`
  ${WORKFLOW_FRAGMENT}
  mutation UpdateWorkflow($id: ID!, $input: UpdateWorkflowInput!) {
    updateWorkflow(id: $id, input: $input) {
      ...WorkflowFields
    }
  }
`;

export const DELETE_WORKFLOW = gql`
  mutation DeleteWorkflow($id: ID!) {
    deleteWorkflow(id: $id)
  }
`;

export const EXECUTE_WORKFLOW = gql`
  mutation ExecuteWorkflow($id: ID!, $input: JSON) {
    executeWorkflow(id: $id, input: $input) {
      id
      status
      startedAt
    }
  }
`;

// ============================================================================
// USER & AUTH QUERIES
// ============================================================================

export const GET_ME = gql`
  query GetMe {
    me {
      id
      email
      name
      role
      preferences
      createdAt
    }
  }
`;

export const UPDATE_PREFERENCES = gql`
  mutation UpdatePreferences($preferences: JSON!) {
    updatePreferences(preferences: $preferences) {
      id
      preferences
    }
  }
`;

// ============================================================================
// SYSTEM QUERIES
// ============================================================================

export const GET_SYSTEM_STATUS = gql`
  query GetSystemStatus {
    systemStatus {
      healthy
      version
      uptime
      services {
        name
        status
        latencyMs
      }
    }
  }
`;

export const GET_DASHBOARD_STATS = gql`
  query GetDashboardStats {
    dashboardStats {
      totalAgents
      activeAgents
      totalExecutions
      successfulExecutions
      totalWorkflows
      activeWorkflows
      tokensUsedToday
      averageResponseTimeMs
    }
  }
`;

// ============================================================================
// SUBSCRIPTION DEFINITIONS (for reference - used via WebSocket)
// ============================================================================

export const AGENT_STATUS_SUBSCRIPTION = gql`
  subscription OnAgentStatusChanged($agentId: ID) {
    agentStatusChanged(agentId: $agentId) {
      id
      status
      lastActiveAt
    }
  }
`;

export const EXECUTION_PROGRESS_SUBSCRIPTION = gql`
  subscription OnExecutionProgress($executionId: ID!) {
    executionProgress(executionId: $executionId) {
      executionId
      status
      progress
      currentStep
      output
      completedAt
    }
  }
`;

export const SYSTEM_EVENTS_SUBSCRIPTION = gql`
  subscription OnSystemEvent {
    systemEvent {
      type
      severity
      message
      timestamp
      metadata
    }
  }
`;

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface Agent {
  id: string;
  name: string;
  description?: string;
  agentType: string;
  status: 'idle' | 'active' | 'processing' | 'error' | 'offline';
  model?: string;
  tier: number;
  capabilities: string[];
  config?: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface AgentMetrics {
  tasksCompleted: number;
  successRate: number;
  averageResponseTimeMs: number;
  tokensUsed: number;
  errorCount: number;
  lastActiveAt?: string;
}

export interface Execution {
  id: string;
  agentId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  input: unknown;
  output?: unknown;
  errorMessage?: string;
  startedAt: string;
  completedAt?: string;
  durationMs?: number;
  tokensUsed?: number;
}

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  status: 'draft' | 'active' | 'archived';
  nodes: unknown[];
  edges: unknown[];
  createdAt: string;
  updatedAt: string;
}

export interface DashboardStats {
  totalAgents: number;
  activeAgents: number;
  totalExecutions: number;
  successfulExecutions: number;
  totalWorkflows: number;
  activeWorkflows: number;
  tokensUsedToday: number;
  averageResponseTimeMs: number;
}

export interface SystemStatus {
  healthy: boolean;
  version: string;
  uptime: number;
  services: {
    name: string;
    status: 'connected' | 'disconnected' | 'degraded';
    latencyMs?: number;
  }[];
}

// ============================================================================
// API FUNCTIONS
// ============================================================================

export const agentsApi = {
  getAll: async (params?: { status?: string; tier?: number; limit?: number; offset?: number }) => {
    const data = await graphqlClient.request<{ agents: Agent[] }>(GET_AGENTS, params);
    return data.agents;
  },
  
  getById: async (id: string) => {
    const data = await graphqlClient.request<{ agent: Agent & { metrics: AgentMetrics } }>(GET_AGENT, { id });
    return data.agent;
  },
  
  create: async (input: Partial<Agent>) => {
    const data = await graphqlClient.request<{ createAgent: Agent }>(CREATE_AGENT, { input });
    return data.createAgent;
  },
  
  update: async (id: string, input: Partial<Agent>) => {
    const data = await graphqlClient.request<{ updateAgent: Agent }>(UPDATE_AGENT, { id, input });
    return data.updateAgent;
  },
  
  delete: async (id: string) => {
    await graphqlClient.request(DELETE_AGENT, { id });
  },
  
  start: async (id: string) => {
    const data = await graphqlClient.request<{ startAgent: { id: string; status: string } }>(START_AGENT, { id });
    return data.startAgent;
  },
  
  stop: async (id: string) => {
    const data = await graphqlClient.request<{ stopAgent: { id: string; status: string } }>(STOP_AGENT, { id });
    return data.stopAgent;
  },
  
  execute: async (agentId: string, input: unknown, stream = false) => {
    const data = await graphqlClient.request<{ executeAgent: Execution }>(EXECUTE_AGENT, { agentId, input, stream });
    return data.executeAgent;
  },
};

export const executionsApi = {
  getAll: async (params?: { agentId?: string; status?: string; limit?: number; offset?: number }) => {
    const data = await graphqlClient.request<{ executions: Execution[] }>(GET_EXECUTIONS, params);
    return data.executions;
  },
  
  getById: async (id: string) => {
    const data = await graphqlClient.request<{ execution: Execution }>(GET_EXECUTION, { id });
    return data.execution;
  },
  
  cancel: async (id: string) => {
    await graphqlClient.request(CANCEL_EXECUTION, { id });
  },
};

export const workflowsApi = {
  getAll: async (params?: { status?: string; limit?: number; offset?: number }) => {
    const data = await graphqlClient.request<{ workflows: Workflow[] }>(GET_WORKFLOWS, params);
    return data.workflows;
  },
  
  getById: async (id: string) => {
    const data = await graphqlClient.request<{ workflow: Workflow }>(GET_WORKFLOW, { id });
    return data.workflow;
  },
  
  create: async (input: Partial<Workflow>) => {
    const data = await graphqlClient.request<{ createWorkflow: Workflow }>(CREATE_WORKFLOW, { input });
    return data.createWorkflow;
  },
  
  update: async (id: string, input: Partial<Workflow>) => {
    const data = await graphqlClient.request<{ updateWorkflow: Workflow }>(UPDATE_WORKFLOW, { id, input });
    return data.updateWorkflow;
  },
  
  delete: async (id: string) => {
    await graphqlClient.request(DELETE_WORKFLOW, { id });
  },
  
  execute: async (id: string, input?: unknown) => {
    const data = await graphqlClient.request<{ executeWorkflow: { id: string; status: string; startedAt: string } }>(
      EXECUTE_WORKFLOW,
      { id, input }
    );
    return data.executeWorkflow;
  },
};

export const systemApi = {
  getStatus: async () => {
    const data = await graphqlClient.request<{ systemStatus: SystemStatus }>(GET_SYSTEM_STATUS);
    return data.systemStatus;
  },
  
  getDashboardStats: async () => {
    const data = await graphqlClient.request<{ dashboardStats: DashboardStats }>(GET_DASHBOARD_STATS);
    return data.dashboardStats;
  },
};

export const userApi = {
  getMe: async () => {
    const data = await graphqlClient.request<{ me: { id: string; email: string; name: string; role: string; preferences: unknown } }>(GET_ME);
    return data.me;
  },
  
  updatePreferences: async (preferences: unknown) => {
    const data = await graphqlClient.request<{ updatePreferences: { id: string; preferences: unknown } }>(
      UPDATE_PREFERENCES,
      { preferences }
    );
    return data.updatePreferences;
  },
};
