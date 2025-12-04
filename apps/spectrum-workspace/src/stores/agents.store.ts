/**
 * Agents Store
 * Manages Elite Agent Collective state
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { Agent, AgentTemplate, AgentMetrics } from '@neurectomy/types';

// Agent status type
type AgentStatus = 'idle' | 'active' | 'processing' | 'error' | 'offline';

// Extended agent with runtime state
interface RuntimeAgent extends Agent {
  status: AgentStatus;
  lastActivity: Date | null;
  currentTask: string | null;
  metrics: AgentMetrics;
}

// Conversation message
interface AgentMessage {
  id: string;
  agentId: string;
  role: 'user' | 'agent' | 'system';
  content: string;
  timestamp: Date;
  metadata?: Record<string, unknown>;
}

// Agent conversation
interface AgentConversation {
  id: string;
  agentId: string;
  title: string;
  messages: AgentMessage[];
  createdAt: Date;
  updatedAt: Date;
}

// Store interface
interface AgentsStore {
  // Agents
  agents: RuntimeAgent[];
  templates: AgentTemplate[];
  selectedAgentId: string | null;
  
  // Actions
  setAgents: (agents: RuntimeAgent[]) => void;
  addAgent: (agent: RuntimeAgent) => void;
  updateAgent: (id: string, updates: Partial<RuntimeAgent>) => void;
  removeAgent: (id: string) => void;
  selectAgent: (id: string | null) => void;
  
  // Templates
  setTemplates: (templates: AgentTemplate[]) => void;
  
  // Conversations
  conversations: AgentConversation[];
  activeConversationId: string | null;
  
  startConversation: (agentId: string) => string;
  addMessage: (conversationId: string, message: Omit<AgentMessage, 'id' | 'timestamp'>) => void;
  setActiveConversation: (id: string | null) => void;
  
  // Agent Metrics
  getAgentMetrics: (id: string) => AgentMetrics | null;
  updateAgentMetrics: (id: string, metrics: Partial<AgentMetrics>) => void;
  
  // Filters
  statusFilter: AgentStatus | 'all';
  tierFilter: number | 'all';
  searchQuery: string;
  
  setStatusFilter: (status: AgentStatus | 'all') => void;
  setTierFilter: (tier: number | 'all') => void;
  setSearchQuery: (query: string) => void;
  
  // Computed
  filteredAgents: () => RuntimeAgent[];
  activeAgentsCount: () => number;
}

// Default metrics
const defaultMetrics: AgentMetrics = {
  tasksCompleted: 0,
  successRate: 0,
  averageResponseTime: 0,
  tokensUsed: 0,
  lastActive: null,
};

export const useAgentsStore = create<AgentsStore>()(
  devtools(
    (set, get) => ({
      // Initial state
      agents: [],
      templates: [],
      selectedAgentId: null,
      conversations: [],
      activeConversationId: null,
      
      // Filters
      statusFilter: 'all',
      tierFilter: 'all',
      searchQuery: '',
      
      // Agent actions
      setAgents: (agents) => set({ agents }),
      
      addAgent: (agent) => set((state) => ({
        agents: [...state.agents, agent],
      })),
      
      updateAgent: (id, updates) => set((state) => ({
        agents: state.agents.map((a) =>
          a.id === id ? { ...a, ...updates } : a
        ),
      })),
      
      removeAgent: (id) => set((state) => ({
        agents: state.agents.filter((a) => a.id !== id),
        selectedAgentId: state.selectedAgentId === id ? null : state.selectedAgentId,
      })),
      
      selectAgent: (id) => set({ selectedAgentId: id }),
      
      // Templates
      setTemplates: (templates) => set({ templates }),
      
      // Conversations
      startConversation: (agentId) => {
        const id = crypto.randomUUID();
        const agent = get().agents.find((a) => a.id === agentId);
        
        set((state) => ({
          conversations: [
            {
              id,
              agentId,
              title: `Conversation with ${agent?.name ?? 'Agent'}`,
              messages: [],
              createdAt: new Date(),
              updatedAt: new Date(),
            },
            ...state.conversations,
          ],
          activeConversationId: id,
        }));
        
        return id;
      },
      
      addMessage: (conversationId, message) => set((state) => ({
        conversations: state.conversations.map((c) =>
          c.id === conversationId
            ? {
                ...c,
                messages: [
                  ...c.messages,
                  {
                    ...message,
                    id: crypto.randomUUID(),
                    timestamp: new Date(),
                  },
                ],
                updatedAt: new Date(),
              }
            : c
        ),
      })),
      
      setActiveConversation: (id) => set({ activeConversationId: id }),
      
      // Metrics
      getAgentMetrics: (id) => {
        const agent = get().agents.find((a) => a.id === id);
        return agent?.metrics ?? null;
      },
      
      updateAgentMetrics: (id, metrics) => set((state) => ({
        agents: state.agents.map((a) =>
          a.id === id
            ? { ...a, metrics: { ...a.metrics, ...metrics } }
            : a
        ),
      })),
      
      // Filter actions
      setStatusFilter: (status) => set({ statusFilter: status }),
      setTierFilter: (tier) => set({ tierFilter: tier }),
      setSearchQuery: (query) => set({ searchQuery: query }),
      
      // Computed values
      filteredAgents: () => {
        const { agents, statusFilter, tierFilter, searchQuery } = get();
        
        return agents.filter((agent) => {
          // Status filter
          if (statusFilter !== 'all' && agent.status !== statusFilter) {
            return false;
          }
          
          // Tier filter
          if (tierFilter !== 'all' && agent.tier !== tierFilter) {
            return false;
          }
          
          // Search query
          if (searchQuery) {
            const query = searchQuery.toLowerCase();
            return (
              agent.name.toLowerCase().includes(query) ||
              agent.codename.toLowerCase().includes(query) ||
              agent.capabilities.some((c) => c.toLowerCase().includes(query))
            );
          }
          
          return true;
        });
      },
      
      activeAgentsCount: () => {
        return get().agents.filter((a) => a.status === 'active').length;
      },
    }),
    { name: 'AgentsStore' }
  )
);

// Selectors
export const useSelectedAgent = () => {
  const agents = useAgentsStore((state) => state.agents);
  const selectedId = useAgentsStore((state) => state.selectedAgentId);
  return agents.find((a) => a.id === selectedId) ?? null;
};

export const useAgentById = (id: string) => {
  return useAgentsStore((state) => state.agents.find((a) => a.id === id));
};

export const useActiveConversation = () => {
  const conversations = useAgentsStore((state) => state.conversations);
  const activeId = useAgentsStore((state) => state.activeConversationId);
  return conversations.find((c) => c.id === activeId) ?? null;
};
