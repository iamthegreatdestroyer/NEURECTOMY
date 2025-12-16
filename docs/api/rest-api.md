# REST API Reference

Base URL: `http://localhost:8000`

## Endpoints

### Generate Text

```http
POST /v1/generate
```

**Request:**

```json
{
  "prompt": "Your prompt here",
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**

```json
{
  "id": "gen_abc123",
  "text": "Generated text...",
  "finish_reason": "stop",
  "prompt_tokens": 10,
  "completion_tokens": 50,
  "latency_ms": 150.5
}
```

### Stream Generation

```http
POST /v1/generate/stream
```

Returns Server-Sent Events:

```
data: {"text": "Hello"}
data: {"text": " world"}
data: {"done": true}
```

### List Agents

```http
GET /v1/agents
```

**Response:**

```json
{
    "total": 40,
    "teams": {
        "inference": ["inference_commander", "inference_specialist_1", ...],
        "compression": ["compression_commander", "compression_specialist_1", ...]
    },
    "agents": [...]
}
```

### Execute Agent Task

```http
POST /v1/agents/task
```

**Request:**

```json
{
  "task_type": "summarize",
  "payload": { "text": "Long text..." },
  "team": "analysis"
}
```

**Response:**

```json
{
  "task_id": "task_abc123",
  "status": "completed",
  "executing_agent": "analysis_commander",
  "result": "Summary of text..."
}
```

### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "components": {
    "inference": { "status": "healthy" },
    "compression": { "status": "healthy" },
    "storage": { "status": "healthy" }
  },
  "timestamp": "2025-12-16T10:30:00Z"
}
```

### Metrics

```http
GET /metrics
```

Returns Prometheus-format metrics with:

- Generation latencies
- Token throughput
- Error rates
- Agent utilization
- Compression ratios

## Error Responses

All endpoints return standard error responses:

```json
{
    "error": "Error message",
    "code": "error_code",
    "details": {...}
}
```

Common status codes:

- `200`: Success
- `400`: Bad request
- `429`: Rate limited
- `500`: Internal error
- `503`: Service unavailable

## Rate Limiting

API requests are rate-limited based on your tier:

- Free: 100 req/min
- Pro: 1,000 req/min
- Enterprise: Custom

Rate limit headers:

- `X-RateLimit-Limit`: Total limit
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Reset time (Unix timestamp)

## Authentication

Include API key in header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8000/v1/generate
```

Or in URL parameter:

```bash
curl http://localhost:8000/v1/generate?api_key=YOUR_API_KEY
```

## Examples

### Generate with curl

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short poem about nature",
    "max_tokens": 100,
    "temperature": 0.8
  }'
```

### Stream with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/generate/stream",
    json={"prompt": "Tell a story", "max_tokens": 200},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```
