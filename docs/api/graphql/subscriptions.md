# GraphQL Subscriptions

Subscriptions provide real-time updates via WebSocket connection.

## Connection

Connect to the WebSocket endpoint:
```
ws://localhost:8080/graphql
```

With authentication:
```javascript
const client = new GraphQLWsClient('ws://localhost:8080/graphql', {
  connectionParams: {
    authorization: `Bearer ${token}`
  }
});
```

---

## Agent Subscriptions

### agentUpdates
Subscribe to updates for a specific agent.

```graphql
subscription AgentUpdates($agentId: UUID!) {
  agentUpdates(agentId: $agentId) {
    id
    status
    lastActivity
    metrics {
      requestsPerMinute
      averageLatency
      errorRate
      activeConversations
    }
  }
}
```

**Events:**
- Status changes (ACTIVE, PAUSED, ERROR)
- Metric updates (every 10 seconds)
- Configuration changes

### agentLogs
Subscribe to agent log stream.

```graphql
subscription AgentLogs($agentId: UUID!, $level: LogLevel) {
  agentLogs(agentId: $agentId, level: $level) {
    timestamp
    level
    message
    metadata
  }
}
```

**Log Levels:**
- `DEBUG`
- `INFO`
- `WARN`
- `ERROR`

---

## Conversation Subscriptions

### messageStream
Subscribe to streaming message responses.

```graphql
subscription MessageStream($conversationId: UUID!) {
  messageStream(conversationId: $conversationId) {
    messageId
    chunk
    index
    isComplete
    tokens
  }
}
```

**Response chunks:**
```json
{
  "data": {
    "messageStream": {
      "messageId": "msg-123",
      "chunk": "Hello",
      "index": 0,
      "isComplete": false,
      "tokens": 1
    }
  }
}
```

### conversationUpdates
Subscribe to conversation events.

```graphql
subscription ConversationUpdates($conversationId: UUID!) {
  conversationUpdates(conversationId: $conversationId) {
    type
    message {
      id
      role
      content
      createdAt
    }
    typing {
      isTyping
      participant
    }
  }
}
```

**Event Types:**
- `NEW_MESSAGE`
- `TYPING_START`
- `TYPING_STOP`
- `PARTICIPANT_JOIN`
- `PARTICIPANT_LEAVE`

---

## Workflow Subscriptions

### workflowExecution
Subscribe to workflow execution progress.

```graphql
subscription WorkflowExecution($executionId: UUID!) {
  workflowExecution(executionId: $executionId) {
    executionId
    status
    currentStep {
      id
      name
      status
      startedAt
      completedAt
      output
      error
    }
    progress {
      completedSteps
      totalSteps
      percentage
    }
  }
}
```

**Execution Status:**
- `PENDING`
- `RUNNING`
- `PAUSED`
- `COMPLETED`
- `FAILED`
- `CANCELLED`

---

## System Subscriptions

### systemEvents
Subscribe to system-wide events.

```graphql
subscription SystemEvents {
  systemEvents {
    type
    severity
    message
    timestamp
    metadata
  }
}
```

**Event Types:**
- `MAINTENANCE_SCHEDULED`
- `MAINTENANCE_STARTED`
- `MAINTENANCE_COMPLETED`
- `SERVICE_DEGRADED`
- `SERVICE_RESTORED`
- `NEW_VERSION_AVAILABLE`

### metricsUpdates
Subscribe to real-time system metrics (admin only).

```graphql
subscription MetricsUpdates($interval: Int) {
  metricsUpdates(interval: $interval) {
    timestamp
    cpu {
      usage
      cores
    }
    memory {
      used
      total
      percentage
    }
    requests {
      perSecond
      latencyP50
      latencyP99
    }
    errors {
      count
      rate
    }
  }
}
```

---

## User Subscriptions

### notifications
Subscribe to user notifications.

```graphql
subscription Notifications {
  notifications {
    id
    type
    title
    message
    read
    createdAt
    action {
      type
      url
      label
    }
  }
}
```

**Notification Types:**
- `AGENT_ERROR`
- `WORKFLOW_COMPLETED`
- `WORKFLOW_FAILED`
- `SYSTEM_ALERT`
- `MENTION`
- `SHARE`

---

## Error Handling

Subscription errors are sent as messages:

```json
{
  "type": "error",
  "payload": {
    "message": "Subscription error occurred",
    "extensions": {
      "code": "SUBSCRIPTION_ERROR"
    }
  }
}
```

## Reconnection

Clients should implement reconnection logic:

```javascript
const client = new GraphQLWsClient(url, {
  retryAttempts: 5,
  retryWait: async (retries) => {
    // Exponential backoff
    await new Promise(resolve => 
      setTimeout(resolve, Math.min(1000 * 2 ** retries, 30000))
    );
  },
  on: {
    connected: () => console.log('Connected'),
    closed: () => console.log('Closed'),
    error: (err) => console.error('Error', err),
  }
});
```

## Rate Limiting

Subscription limits per user:
- Maximum concurrent subscriptions: 10
- Maximum events per second: 100
- Connection idle timeout: 5 minutes (with heartbeat)
