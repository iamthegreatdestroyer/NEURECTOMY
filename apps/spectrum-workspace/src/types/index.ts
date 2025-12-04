// Core Agent Types
export interface Agent {
  id: string;
  name: string;
  type: AgentType;
  status: AgentStatus;
  model?: string;
  capabilities?: string[];
  createdAt: string;
  updatedAt: string;
  metrics?: AgentMetrics;
}

export type AgentType = 
  | 'assistant'
  | 'coding'
  | 'research'
  | 'creative'
  | 'security'
  | 'devops'
  | 'custom';

export type AgentStatus = 
  | 'active'
  | 'idle'
  | 'busy'
  | 'error'
  | 'offline';

export interface AgentMetrics {
  messagesProcessed: number;
  avgResponseTime: number;
  uptime: number;
  errorRate: number;
  tokensUsed: number;
}

export interface AgentTemplate {
  id: string;
  name: string;
  description: string;
  icon: string;
  type: AgentType;
  systemPrompt: string;
  capabilities: string[];
  defaultModel: string;
}

// Conversation Types
export interface Conversation {
  id: string;
  agentId: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

// Container Types
export interface Container {
  id: string;
  name: string;
  image: string;
  status: ContainerStatus;
  ports: PortMapping[];
  cpu: number;
  memory: number;
  createdAt: string;
}

export type ContainerStatus = 
  | 'running'
  | 'stopped'
  | 'paused'
  | 'restarting'
  | 'exited';

export interface PortMapping {
  hostPort: number;
  containerPort: number;
  protocol: 'tcp' | 'udp';
}

export interface KubernetesCluster {
  id: string;
  name: string;
  status: 'active' | 'inactive' | 'provisioning';
  nodes: number;
  pods: number;
  services: number;
  namespace: string;
}

// ML/Training Types
export interface TrainingJob {
  id: string;
  name: string;
  model: string;
  status: TrainingStatus;
  progress: number;
  epochs: number;
  currentEpoch: number;
  loss: number;
  accuracy: number;
  startedAt: string;
  estimatedCompletion?: string;
}

export type TrainingStatus = 
  | 'queued'
  | 'preparing'
  | 'training'
  | 'evaluating'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface ModelVersion {
  id: string;
  name: string;
  version: string;
  framework: 'pytorch' | 'tensorflow' | 'onnx';
  size: number;
  metrics: ModelMetrics;
  createdAt: string;
}

export interface ModelMetrics {
  accuracy: number;
  loss: number;
  f1Score?: number;
  precision?: number;
  recall?: number;
}

// Research/Discovery Types
export interface ResearchTopic {
  id: string;
  title: string;
  description: string;
  status: 'active' | 'completed' | 'archived';
  sources: number;
  insights: number;
  createdAt: string;
}

export interface KnowledgeNode {
  id: string;
  label: string;
  type: 'concept' | 'entity' | 'document' | 'insight';
  connections: string[];
  metadata?: Record<string, unknown>;
}

// Notification Types
export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  action?: NotificationAction;
}

export interface NotificationAction {
  label: string;
  href?: string;
  onClick?: () => void;
}

// Settings Types
export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  sidebarCollapsed: boolean;
  notifications: NotificationPreferences;
  editor: EditorPreferences;
  ai: AIPreferences;
}

export interface NotificationPreferences {
  email: boolean;
  push: boolean;
  desktop: boolean;
  sound: boolean;
}

export interface EditorPreferences {
  fontSize: number;
  fontFamily: string;
  tabSize: number;
  lineNumbers: boolean;
  minimap: boolean;
  wordWrap: boolean;
}

export interface AIPreferences {
  defaultModel: string;
  temperature: number;
  maxTokens: number;
  streamResponses: boolean;
  saveHistory: boolean;
}

// API Response Types
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  error?: ApiError;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

// WebSocket Types
export interface WebSocketMessage {
  type: string;
  payload: unknown;
  timestamp: string;
}

export interface AgentStreamEvent {
  type: 'token' | 'complete' | 'error';
  agentId: string;
  content?: string;
  error?: string;
}
