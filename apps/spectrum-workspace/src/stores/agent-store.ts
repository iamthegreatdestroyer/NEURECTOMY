/**
 * Agent Store
 *
 * State management for AI agents.
 * Handles agent creation, configuration, execution, and monitoring.
 */

import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { devtools } from "zustand/middleware";

export type AgentStatus = "idle" | "running" | "paused" | "error" | "completed";
export type AgentType =
  | "sequential"
  | "parallel"
  | "conditional"
  | "loop"
  | "swarm";

export interface AgentNode {
  id: string;
  name: string;
  type: AgentType;
  codename?: string; // e.g., '@APEX', '@CIPHER'
  description?: string;

  // Position in 3D space
  position: { x: number; y: number; z: number };

  // Visual properties
  color?: string;
  scale?: number;
  visible?: boolean;

  // Configuration
  config: Record<string, unknown>;

  // State
  status: AgentStatus;
  metadata?: Record<string, unknown>;

  // Connections
  inputs: string[]; // IDs of connected agents
  outputs: string[]; // IDs of connected agents

  // Metrics
  metrics?: {
    executionTime?: number;
    tokensUsed?: number;
    cost?: number;
    successRate?: number;
  };

  // Timestamps
  createdAt: string;
  updatedAt: string;
  lastExecutedAt?: string;
}

export interface AgentConnection {
  id: string;
  sourceId: string;
  targetId: string;
  type: "data" | "control" | "feedback";
  condition?: string;
  weight?: number;
}

export interface AgentWorkflow {
  id: string;
  name: string;
  description?: string;
  nodes: AgentNode[];
  connections: AgentConnection[];
  status: AgentStatus;
  createdAt: string;
  updatedAt: string;
}

export interface AgentState {
  // Workflows
  workflows: AgentWorkflow[];
  activeWorkflowId: string | null;

  // Selection
  selectedNodeIds: string[];
  selectedConnectionIds: string[];

  // Execution
  executionHistory: Array<{
    workflowId: string;
    nodeId: string;
    timestamp: string;
    status: AgentStatus;
    result?: unknown;
    error?: string;
  }>;

  // Actions - Workflows
  createWorkflow: (
    workflow: Omit<AgentWorkflow, "createdAt" | "updatedAt">
  ) => void;
  updateWorkflow: (workflowId: string, updates: Partial<AgentWorkflow>) => void;
  deleteWorkflow: (workflowId: string) => void;
  setActiveWorkflow: (workflowId: string | null) => void;

  // Actions - Nodes
  addNode: (
    workflowId: string,
    node: Omit<AgentNode, "createdAt" | "updatedAt">
  ) => void;
  updateNode: (
    workflowId: string,
    nodeId: string,
    updates: Partial<AgentNode>
  ) => void;
  deleteNode: (workflowId: string, nodeId: string) => void;
  moveNode: (
    workflowId: string,
    nodeId: string,
    position: { x: number; y: number; z: number }
  ) => void;

  // Actions - Connections
  addConnection: (workflowId: string, connection: AgentConnection) => void;
  deleteConnection: (workflowId: string, connectionId: string) => void;

  // Actions - Selection
  selectNode: (nodeId: string, addToSelection?: boolean) => void;
  selectConnection: (connectionId: string, addToSelection?: boolean) => void;
  clearSelection: () => void;

  // Actions - Execution
  startWorkflow: (workflowId: string) => void;
  pauseWorkflow: (workflowId: string) => void;
  stopWorkflow: (workflowId: string) => void;
  addExecutionRecord: (record: AgentState["executionHistory"][0]) => void;

  reset: () => void;
}

export const useAgentStore = create<AgentState>()(
  devtools(
    immer((set) => ({
      // Initial state
      workflows: [],
      activeWorkflowId: null,
      selectedNodeIds: [],
      selectedConnectionIds: [],
      executionHistory: [],

      // Workflow actions
      createWorkflow: (workflow) =>
        set((state) => {
          const now = new Date().toISOString();
          const newWorkflow: AgentWorkflow = {
            ...workflow,
            createdAt: now,
            updatedAt: now,
          };
          state.workflows.push(newWorkflow);
          state.activeWorkflowId = newWorkflow.id;
        }),

      updateWorkflow: (workflowId, updates) =>
        set((state) => {
          const workflow = state.workflows.find((w) => w.id === workflowId);
          if (workflow) {
            Object.assign(workflow, updates);
            workflow.updatedAt = new Date().toISOString();
          }
        }),

      deleteWorkflow: (workflowId) =>
        set((state) => {
          const index = state.workflows.findIndex((w) => w.id === workflowId);
          if (index !== -1) {
            state.workflows.splice(index, 1);
            if (state.activeWorkflowId === workflowId) {
              state.activeWorkflowId = state.workflows[0]?.id || null;
            }
          }
        }),

      setActiveWorkflow: (workflowId) =>
        set((state) => {
          state.activeWorkflowId = workflowId;
        }),

      // Node actions
      addNode: (workflowId, node) =>
        set((state) => {
          const workflow = state.workflows.find((w) => w.id === workflowId);
          if (workflow) {
            const now = new Date().toISOString();
            const newNode: AgentNode = {
              ...node,
              createdAt: now,
              updatedAt: now,
            };
            workflow.nodes.push(newNode);
            workflow.updatedAt = now;
          }
        }),

      updateNode: (workflowId, nodeId, updates) =>
        set((state) => {
          const workflow = state.workflows.find((w) => w.id === workflowId);
          if (workflow) {
            const node = workflow.nodes.find((n) => n.id === nodeId);
            if (node) {
              Object.assign(node, updates);
              node.updatedAt = new Date().toISOString();
              workflow.updatedAt = node.updatedAt;
            }
          }
        }),

      deleteNode: (workflowId, nodeId) =>
        set((state) => {
          const workflow = state.workflows.find((w) => w.id === workflowId);
          if (workflow) {
            const nodeIndex = workflow.nodes.findIndex((n) => n.id === nodeId);
            if (nodeIndex !== -1) {
              workflow.nodes.splice(nodeIndex, 1);

              // Remove connections involving this node
              workflow.connections = workflow.connections.filter(
                (conn) => conn.sourceId !== nodeId && conn.targetId !== nodeId
              );

              workflow.updatedAt = new Date().toISOString();
            }
          }
        }),

      moveNode: (workflowId, nodeId, position) =>
        set((state) => {
          const workflow = state.workflows.find((w) => w.id === workflowId);
          if (workflow) {
            const node = workflow.nodes.find((n) => n.id === nodeId);
            if (node) {
              node.position = position;
              node.updatedAt = new Date().toISOString();
              workflow.updatedAt = node.updatedAt;
            }
          }
        }),

      // Connection actions
      addConnection: (workflowId, connection) =>
        set((state) => {
          const workflow = state.workflows.find((w) => w.id === workflowId);
          if (workflow) {
            workflow.connections.push(connection);

            // Update node inputs/outputs
            const sourceNode = workflow.nodes.find(
              (n) => n.id === connection.sourceId
            );
            const targetNode = workflow.nodes.find(
              (n) => n.id === connection.targetId
            );

            if (
              sourceNode &&
              !sourceNode.outputs.includes(connection.targetId)
            ) {
              sourceNode.outputs.push(connection.targetId);
            }
            if (
              targetNode &&
              !targetNode.inputs.includes(connection.sourceId)
            ) {
              targetNode.inputs.push(connection.sourceId);
            }

            workflow.updatedAt = new Date().toISOString();
          }
        }),

      deleteConnection: (workflowId, connectionId) =>
        set((state) => {
          const workflow = state.workflows.find((w) => w.id === workflowId);
          if (workflow) {
            const connection = workflow.connections.find(
              (c) => c.id === connectionId
            );
            if (connection) {
              // Update node inputs/outputs
              const sourceNode = workflow.nodes.find(
                (n) => n.id === connection.sourceId
              );
              const targetNode = workflow.nodes.find(
                (n) => n.id === connection.targetId
              );

              if (sourceNode) {
                sourceNode.outputs = sourceNode.outputs.filter(
                  (id) => id !== connection.targetId
                );
              }
              if (targetNode) {
                targetNode.inputs = targetNode.inputs.filter(
                  (id) => id !== connection.sourceId
                );
              }

              // Remove connection
              workflow.connections = workflow.connections.filter(
                (c) => c.id !== connectionId
              );
              workflow.updatedAt = new Date().toISOString();
            }
          }
        }),

      // Selection actions
      selectNode: (nodeId, addToSelection = false) =>
        set((state) => {
          if (addToSelection) {
            if (!state.selectedNodeIds.includes(nodeId)) {
              state.selectedNodeIds.push(nodeId);
            }
          } else {
            state.selectedNodeIds = [nodeId];
            state.selectedConnectionIds = [];
          }
        }),

      selectConnection: (connectionId, addToSelection = false) =>
        set((state) => {
          if (addToSelection) {
            if (!state.selectedConnectionIds.includes(connectionId)) {
              state.selectedConnectionIds.push(connectionId);
            }
          } else {
            state.selectedConnectionIds = [connectionId];
            state.selectedNodeIds = [];
          }
        }),

      clearSelection: () =>
        set((state) => {
          state.selectedNodeIds = [];
          state.selectedConnectionIds = [];
        }),

      // Execution actions
      startWorkflow: (workflowId) =>
        set((state) => {
          const workflow = state.workflows.find((w) => w.id === workflowId);
          if (workflow) {
            workflow.status = "running";
            workflow.updatedAt = new Date().toISOString();
          }
        }),

      pauseWorkflow: (workflowId) =>
        set((state) => {
          const workflow = state.workflows.find((w) => w.id === workflowId);
          if (workflow) {
            workflow.status = "paused";
            workflow.updatedAt = new Date().toISOString();
          }
        }),

      stopWorkflow: (workflowId) =>
        set((state) => {
          const workflow = state.workflows.find((w) => w.id === workflowId);
          if (workflow) {
            workflow.status = "idle";
            workflow.updatedAt = new Date().toISOString();
          }
        }),

      addExecutionRecord: (record) =>
        set((state) => {
          state.executionHistory.push(record);
          // Keep only last 1000 records
          if (state.executionHistory.length > 1000) {
            state.executionHistory = state.executionHistory.slice(-1000);
          }
        }),

      reset: () =>
        set({
          workflows: [],
          activeWorkflowId: null,
          selectedNodeIds: [],
          selectedConnectionIds: [],
          executionHistory: [],
        }),
    })),
    { name: "AgentStore" }
  )
);
