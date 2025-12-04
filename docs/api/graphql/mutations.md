# GraphQL Mutations

## Authentication Mutations

### login
Authenticate user and receive tokens.

```graphql
mutation Login($email: String!, $password: String!) {
  login(email: $email, password: $password) {
    accessToken
    refreshToken
    expiresIn
    user {
      id
      email
      username
      role
    }
  }
}
```

**Variables:**
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!"
}
```

### refreshToken
Refresh access token using refresh token.

```graphql
mutation RefreshToken($refreshToken: String!) {
  refreshToken(refreshToken: $refreshToken) {
    accessToken
    refreshToken
    expiresIn
  }
}
```

### logout
Invalidate current session.

```graphql
mutation Logout {
  logout {
    success
  }
}
```

### register
Create a new user account.

```graphql
mutation Register($input: RegisterInput!) {
  register(input: $input) {
    user {
      id
      email
      username
    }
    accessToken
    refreshToken
  }
}
```

**Input:**
```json
{
  "input": {
    "email": "newuser@example.com",
    "username": "newuser",
    "password": "SecurePassword123!"
  }
}
```

---

## Agent Mutations

### createAgent
Create a new agent.

```graphql
mutation CreateAgent($input: CreateAgentInput!) {
  createAgent(input: $input) {
    agent {
      id
      name
      status
      createdAt
    }
  }
}
```

**Input:**
```json
{
  "input": {
    "name": "Customer Support Agent",
    "description": "Handles customer inquiries",
    "systemPrompt": "You are a helpful customer support agent...",
    "model": {
      "provider": "OPENAI",
      "name": "gpt-4",
      "temperature": 0.7,
      "maxTokens": 4096
    },
    "config": {
      "streaming": true,
      "tools": ["search", "knowledge_base"]
    }
  }
}
```

### updateAgent
Update an existing agent.

```graphql
mutation UpdateAgent($id: UUID!, $input: UpdateAgentInput!) {
  updateAgent(id: $id, input: $input) {
    agent {
      id
      name
      status
      updatedAt
    }
  }
}
```

### deleteAgent
Delete an agent.

```graphql
mutation DeleteAgent($id: UUID!) {
  deleteAgent(id: $id) {
    success
    deletedAt
  }
}
```

### startAgent
Start an agent (change status to ACTIVE).

```graphql
mutation StartAgent($id: UUID!) {
  startAgent(id: $id) {
    agent {
      id
      status
    }
  }
}
```

### stopAgent
Stop an agent (change status to PAUSED).

```graphql
mutation StopAgent($id: UUID!) {
  stopAgent(id: $id) {
    agent {
      id
      status
    }
  }
}
```

---

## Conversation Mutations

### createConversation
Start a new conversation with an agent.

```graphql
mutation CreateConversation($input: CreateConversationInput!) {
  createConversation(input: $input) {
    conversation {
      id
      title
      agent {
        id
        name
      }
      createdAt
    }
  }
}
```

**Input:**
```json
{
  "input": {
    "agentId": "550e8400-e29b-41d4-a716-446655440000",
    "title": "Support Request #123"
  }
}
```

### sendMessage
Send a message in a conversation.

```graphql
mutation SendMessage($input: SendMessageInput!) {
  sendMessage(input: $input) {
    message {
      id
      role
      content
      tokens
      createdAt
    }
    response {
      id
      role
      content
      tokens
      model
      latency
    }
  }
}
```

**Input:**
```json
{
  "input": {
    "conversationId": "550e8400-e29b-41d4-a716-446655440000",
    "content": "How do I reset my password?"
  }
}
```

### deleteConversation
Delete a conversation and all its messages.

```graphql
mutation DeleteConversation($id: UUID!) {
  deleteConversation(id: $id) {
    success
  }
}
```

---

## Workflow Mutations

### createWorkflow
Create a new workflow.

```graphql
mutation CreateWorkflow($input: CreateWorkflowInput!) {
  createWorkflow(input: $input) {
    workflow {
      id
      name
      status
      createdAt
    }
  }
}
```

### updateWorkflow
Update a workflow definition.

```graphql
mutation UpdateWorkflow($id: UUID!, $input: UpdateWorkflowInput!) {
  updateWorkflow(id: $id, input: $input) {
    workflow {
      id
      name
      status
      updatedAt
    }
  }
}
```

### executeWorkflow
Execute a workflow.

```graphql
mutation ExecuteWorkflow($id: UUID!, $input: JSON) {
  executeWorkflow(id: $id, input: $input) {
    execution {
      id
      status
      startedAt
    }
  }
}
```

### cancelWorkflow
Cancel a running workflow execution.

```graphql
mutation CancelWorkflow($executionId: UUID!) {
  cancelWorkflow(executionId: $executionId) {
    success
    execution {
      id
      status
      completedAt
    }
  }
}
```

---

## Knowledge Base Mutations

### createKnowledgeBase
Create a new knowledge base.

```graphql
mutation CreateKnowledgeBase($input: CreateKnowledgeBaseInput!) {
  createKnowledgeBase(input: $input) {
    knowledgeBase {
      id
      name
      description
      createdAt
    }
  }
}
```

### addDocument
Add a document to a knowledge base.

```graphql
mutation AddDocument($input: AddDocumentInput!) {
  addDocument(input: $input) {
    document {
      id
      title
      contentType
      status
      chunks
      createdAt
    }
  }
}
```

**Input:**
```json
{
  "input": {
    "knowledgeBaseId": "550e8400-e29b-41d4-a716-446655440000",
    "title": "Product Manual",
    "content": "...",
    "contentType": "text/markdown",
    "metadata": {
      "version": "2.0",
      "category": "documentation"
    }
  }
}
```

### removeDocument
Remove a document from a knowledge base.

```graphql
mutation RemoveDocument($id: UUID!) {
  removeDocument(id: $id) {
    success
  }
}
```

---

## API Key Mutations

### createApiKey
Create a new API key.

```graphql
mutation CreateApiKey($input: CreateApiKeyInput!) {
  createApiKey(input: $input) {
    apiKey {
      id
      name
      key  # Only returned once at creation!
      prefix
      scopes
      expiresAt
      createdAt
    }
  }
}
```

**Input:**
```json
{
  "input": {
    "name": "CI/CD Pipeline",
    "scopes": ["agents:read", "agents:write"],
    "expiresAt": "2025-01-15T00:00:00Z"
  }
}
```

### revokeApiKey
Revoke an API key.

```graphql
mutation RevokeApiKey($id: UUID!) {
  revokeApiKey(id: $id) {
    success
    revokedAt
  }
}
```

---

## User Settings Mutations

### updateProfile
Update user profile.

```graphql
mutation UpdateProfile($input: UpdateProfileInput!) {
  updateProfile(input: $input) {
    user {
      id
      username
      email
      updatedAt
    }
  }
}
```

### changePassword
Change user password.

```graphql
mutation ChangePassword($currentPassword: String!, $newPassword: String!) {
  changePassword(currentPassword: $currentPassword, newPassword: $newPassword) {
    success
  }
}
```

### updateSettings
Update user settings.

```graphql
mutation UpdateSettings($input: UpdateSettingsInput!) {
  updateSettings(input: $input) {
    settings {
      theme
      notifications
      timezone
      language
    }
  }
}
```
