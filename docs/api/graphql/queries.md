# GraphQL Queries

## User Queries

### me
Get the currently authenticated user.

```graphql
query {
  me {
    id
    email
    username
    role
    createdAt
    settings {
      theme
      notifications
    }
  }
}
```

**Response:**
```json
{
  "data": {
    "me": {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "email": "user@example.com",
      "username": "johndoe",
      "role": "USER",
      "createdAt": "2024-01-15T10:30:00Z",
      "settings": {
        "theme": "dark",
        "notifications": true
      }
    }
  }
}
```

### user
Get a user by ID (admin only).

```graphql
query GetUser($id: UUID!) {
  user(id: $id) {
    id
    email
    username
    role
    agents {
      totalCount
    }
  }
}
```

---

## Agent Queries

### agent
Get a single agent by ID.

```graphql
query GetAgent($id: UUID!) {
  agent(id: $id) {
    id
    name
    description
    status
    config
    systemPrompt
    model {
      provider
      name
      temperature
      maxTokens
    }
    metrics {
      totalRequests
      averageLatency
      errorRate
    }
    createdAt
    updatedAt
  }
}
```

**Variables:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### agents
List agents with pagination and filtering.

```graphql
query ListAgents(
  $first: Int
  $after: String
  $filter: AgentFilterInput
  $orderBy: AgentOrderInput
) {
  agents(first: $first, after: $after, filter: $filter, orderBy: $orderBy) {
    edges {
      cursor
      node {
        id
        name
        status
        createdAt
      }
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
      startCursor
      endCursor
    }
    totalCount
  }
}
```

**Variables:**
```json
{
  "first": 10,
  "filter": {
    "status": "ACTIVE"
  },
  "orderBy": {
    "field": "CREATED_AT",
    "direction": "DESC"
  }
}
```

---

## Conversation Queries

### conversation
Get a conversation by ID.

```graphql
query GetConversation($id: UUID!) {
  conversation(id: $id) {
    id
    title
    agent {
      id
      name
    }
    messages(first: 50) {
      edges {
        node {
          id
          role
          content
          createdAt
          tokens
        }
      }
    }
    createdAt
    updatedAt
  }
}
```

### conversations
List conversations for the current user.

```graphql
query ListConversations($agentId: UUID, $first: Int, $after: String) {
  conversations(agentId: $agentId, first: $first, after: $after) {
    edges {
      cursor
      node {
        id
        title
        agent {
          id
          name
        }
        messageCount
        lastActivity
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
```

---

## Workflow Queries

### workflow
Get a workflow by ID.

```graphql
query GetWorkflow($id: UUID!) {
  workflow(id: $id) {
    id
    name
    description
    status
    definition
    triggers {
      type
      config
    }
    steps {
      id
      type
      config
      position {
        x
        y
      }
    }
    createdAt
    updatedAt
  }
}
```

### workflows
List workflows.

```graphql
query ListWorkflows($first: Int, $status: WorkflowStatus) {
  workflows(first: $first, filter: { status: $status }) {
    edges {
      node {
        id
        name
        status
        lastRun
        runCount
      }
    }
    totalCount
  }
}
```

---

## Knowledge Base Queries

### knowledgeBase
Get a knowledge base entry.

```graphql
query GetKnowledgeBase($id: UUID!) {
  knowledgeBase(id: $id) {
    id
    name
    description
    documentCount
    totalTokens
    documents(first: 20) {
      edges {
        node {
          id
          title
          contentType
          size
          createdAt
        }
      }
    }
  }
}
```

### semanticSearch
Search knowledge base using semantic similarity.

```graphql
query SemanticSearch($query: String!, $knowledgeBaseId: UUID!, $limit: Int) {
  semanticSearch(query: $query, knowledgeBaseId: $knowledgeBaseId, limit: $limit) {
    results {
      id
      content
      score
      metadata
      document {
        id
        title
      }
    }
  }
}
```

---

## System Queries

### systemHealth
Get system health status.

```graphql
query {
  systemHealth {
    status
    services {
      name
      status
      latency
    }
    uptime
    version
  }
}
```

### metrics
Get system metrics (admin only).

```graphql
query GetMetrics($timeRange: TimeRange!) {
  metrics(timeRange: $timeRange) {
    requests {
      timestamp
      count
      latencyP50
      latencyP99
    }
    agents {
      active
      total
    }
    users {
      active
      total
    }
  }
}
```
