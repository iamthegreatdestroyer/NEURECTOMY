# GraphQL Types

## Scalar Types

### UUID
A universally unique identifier.
```graphql
scalar UUID
```

### DateTime
ISO 8601 formatted date-time string.
```graphql
scalar DateTime
```

### JSON
Arbitrary JSON object.
```graphql
scalar JSON
```

---

## Enums

### UserRole
```graphql
enum UserRole {
  GUEST
  USER
  MODERATOR
  ADMIN
  SUPERADMIN
}
```

### AgentStatus
```graphql
enum AgentStatus {
  DRAFT
  ACTIVE
  PAUSED
  ERROR
  ARCHIVED
}
```

### ModelProvider
```graphql
enum ModelProvider {
  OPENAI
  ANTHROPIC
  GOOGLE
  LOCAL
  CUSTOM
}
```

### WorkflowStatus
```graphql
enum WorkflowStatus {
  DRAFT
  ACTIVE
  PAUSED
  ARCHIVED
}
```

### ExecutionStatus
```graphql
enum ExecutionStatus {
  PENDING
  RUNNING
  PAUSED
  COMPLETED
  FAILED
  CANCELLED
}
```

### MessageRole
```graphql
enum MessageRole {
  SYSTEM
  USER
  ASSISTANT
  FUNCTION
  TOOL
}
```

### LogLevel
```graphql
enum LogLevel {
  DEBUG
  INFO
  WARN
  ERROR
}
```

### SortDirection
```graphql
enum SortDirection {
  ASC
  DESC
}
```

---

## Object Types

### User
```graphql
type User {
  id: UUID!
  email: String!
  username: String!
  role: UserRole!
  avatarUrl: String
  settings: UserSettings!
  agents(first: Int, after: String): AgentConnection!
  apiKeys: [ApiKey!]!
  createdAt: DateTime!
  updatedAt: DateTime!
}
```

### UserSettings
```graphql
type UserSettings {
  theme: String!
  notifications: Boolean!
  timezone: String!
  language: String!
}
```

### Agent
```graphql
type Agent {
  id: UUID!
  name: String!
  description: String
  status: AgentStatus!
  systemPrompt: String!
  model: ModelConfig!
  config: JSON!
  tools: [Tool!]!
  knowledgeBases: [KnowledgeBase!]!
  conversations(first: Int, after: String): ConversationConnection!
  metrics: AgentMetrics!
  owner: User!
  createdAt: DateTime!
  updatedAt: DateTime!
}
```

### ModelConfig
```graphql
type ModelConfig {
  provider: ModelProvider!
  name: String!
  temperature: Float!
  maxTokens: Int!
  topP: Float
  frequencyPenalty: Float
  presencePenalty: Float
  stop: [String!]
}
```

### Tool
```graphql
type Tool {
  id: String!
  name: String!
  description: String!
  type: String!
  config: JSON!
  enabled: Boolean!
}
```

### AgentMetrics
```graphql
type AgentMetrics {
  totalRequests: Int!
  totalTokens: Int!
  averageLatency: Float!
  errorRate: Float!
  requestsPerMinute: Float!
  activeConversations: Int!
  lastActivity: DateTime
}
```

### Conversation
```graphql
type Conversation {
  id: UUID!
  title: String
  agent: Agent!
  messages(first: Int, after: String): MessageConnection!
  messageCount: Int!
  totalTokens: Int!
  lastActivity: DateTime!
  metadata: JSON
  createdAt: DateTime!
  updatedAt: DateTime!
}
```

### Message
```graphql
type Message {
  id: UUID!
  role: MessageRole!
  content: String!
  tokens: Int!
  model: String
  latency: Int
  metadata: JSON
  createdAt: DateTime!
}
```

### Workflow
```graphql
type Workflow {
  id: UUID!
  name: String!
  description: String
  status: WorkflowStatus!
  definition: JSON!
  triggers: [WorkflowTrigger!]!
  steps: [WorkflowStep!]!
  executions(first: Int, after: String): ExecutionConnection!
  runCount: Int!
  lastRun: DateTime
  owner: User!
  createdAt: DateTime!
  updatedAt: DateTime!
}
```

### WorkflowTrigger
```graphql
type WorkflowTrigger {
  id: String!
  type: String!
  config: JSON!
  enabled: Boolean!
}
```

### WorkflowStep
```graphql
type WorkflowStep {
  id: String!
  name: String!
  type: String!
  config: JSON!
  position: Position!
  connections: [StepConnection!]!
}
```

### Position
```graphql
type Position {
  x: Float!
  y: Float!
}
```

### Execution
```graphql
type Execution {
  id: UUID!
  workflow: Workflow!
  status: ExecutionStatus!
  input: JSON
  output: JSON
  error: String
  steps: [StepExecution!]!
  startedAt: DateTime!
  completedAt: DateTime
  duration: Int
}
```

### KnowledgeBase
```graphql
type KnowledgeBase {
  id: UUID!
  name: String!
  description: String
  documents(first: Int, after: String): DocumentConnection!
  documentCount: Int!
  totalTokens: Int!
  embeddingModel: String!
  owner: User!
  createdAt: DateTime!
  updatedAt: DateTime!
}
```

### Document
```graphql
type Document {
  id: UUID!
  title: String!
  content: String
  contentType: String!
  size: Int!
  chunks: Int!
  status: String!
  metadata: JSON
  createdAt: DateTime!
  updatedAt: DateTime!
}
```

### SearchResult
```graphql
type SearchResult {
  id: UUID!
  content: String!
  score: Float!
  metadata: JSON
  document: Document!
}
```

### ApiKey
```graphql
type ApiKey {
  id: UUID!
  name: String!
  prefix: String!
  scopes: [String!]!
  lastUsed: DateTime
  expiresAt: DateTime
  createdAt: DateTime!
}
```

### SystemHealth
```graphql
type SystemHealth {
  status: String!
  services: [ServiceHealth!]!
  uptime: Int!
  version: String!
}
```

### ServiceHealth
```graphql
type ServiceHealth {
  name: String!
  status: String!
  latency: Int!
  message: String
}
```

---

## Connection Types (Relay)

### PageInfo
```graphql
type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
```

### AgentConnection
```graphql
type AgentConnection {
  edges: [AgentEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type AgentEdge {
  cursor: String!
  node: Agent!
}
```

### ConversationConnection
```graphql
type ConversationConnection {
  edges: [ConversationEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type ConversationEdge {
  cursor: String!
  node: Conversation!
}
```

### MessageConnection
```graphql
type MessageConnection {
  edges: [MessageEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type MessageEdge {
  cursor: String!
  node: Message!
}
```

---

## Input Types

### CreateAgentInput
```graphql
input CreateAgentInput {
  name: String!
  description: String
  systemPrompt: String!
  model: ModelConfigInput!
  config: JSON
  tools: [String!]
  knowledgeBaseIds: [UUID!]
}
```

### UpdateAgentInput
```graphql
input UpdateAgentInput {
  name: String
  description: String
  systemPrompt: String
  model: ModelConfigInput
  config: JSON
  tools: [String!]
  knowledgeBaseIds: [UUID!]
  status: AgentStatus
}
```

### ModelConfigInput
```graphql
input ModelConfigInput {
  provider: ModelProvider!
  name: String!
  temperature: Float
  maxTokens: Int
  topP: Float
  frequencyPenalty: Float
  presencePenalty: Float
  stop: [String!]
}
```

### AgentFilterInput
```graphql
input AgentFilterInput {
  status: AgentStatus
  createdAfter: DateTime
  createdBefore: DateTime
  search: String
}
```

### AgentOrderInput
```graphql
input AgentOrderInput {
  field: AgentOrderField!
  direction: SortDirection!
}

enum AgentOrderField {
  NAME
  STATUS
  CREATED_AT
  UPDATED_AT
  LAST_ACTIVITY
}
```

### SendMessageInput
```graphql
input SendMessageInput {
  conversationId: UUID!
  content: String!
  metadata: JSON
}
```

### CreateApiKeyInput
```graphql
input CreateApiKeyInput {
  name: String!
  scopes: [String!]!
  expiresAt: DateTime
}
```
