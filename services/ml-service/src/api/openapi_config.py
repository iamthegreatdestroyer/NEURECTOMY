# ============================================================================
# NEURECTOMY ML Service - OpenAPI Configuration
# FastAPI automatic documentation enhancement
# ============================================================================

"""
OpenAPI/Swagger configuration for ML Service API documentation.

This module configures:
- API metadata (title, version, description)
- Security schemes (JWT, API Key)
- Custom documentation endpoints
- Schema customization
"""

from typing import Dict, Any

# ============================================================================
# API Metadata
# ============================================================================

API_TITLE = "NEURECTOMY ML Service API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
# NEURECTOMY Intelligence Layer API

The ML Service provides AI/ML capabilities for the NEURECTOMY platform, including:

## Features

### 🧠 Agent Training
- Neural network training with customizable architectures
- Transfer learning and fine-tuning
- Distributed training support
- Real-time training monitoring via WebSocket

### 📊 Predictive Analytics
- Time series forecasting
- Anomaly detection
- Performance prediction
- Resource optimization

### 🔐 Security
- JWT authentication with refresh tokens
- API key management
- Rate limiting
- Audit logging

### 🔌 Real-Time Communication
- WebSocket connections for live updates
- Training progress streaming
- System event notifications

## Authentication

The API uses JWT tokens for authentication. Include the token in the `Authorization` header:

```
Authorization: Bearer <your-jwt-token>
```

For API key authentication, use the `X-API-Key` header:

```
X-API-Key: <your-api-key>
```

## Rate Limiting

- Default: 100 requests per minute
- Training endpoints: 10 requests per minute
- Authentication endpoints: 20 requests per minute

## Errors

The API uses standard HTTP status codes:

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 422 | Validation Error |
| 429 | Too Many Requests |
| 500 | Internal Server Error |

## WebSocket

Connect to `/ws` for real-time updates:

```javascript
const ws = new WebSocket('wss://api.neurectomy.io/ws');
ws.onmessage = (event) => console.log(JSON.parse(event.data));
```
"""

# ============================================================================
# Tags Metadata
# ============================================================================

TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Health check and system status endpoints",
    },
    {
        "name": "Authentication",
        "description": "User authentication, token management, and API keys",
    },
    {
        "name": "Training",
        "description": "Agent training operations and job management",
    },
    {
        "name": "Inference",
        "description": "Model inference and prediction endpoints",
    },
    {
        "name": "Analytics",
        "description": "Predictive analytics, forecasting, and anomaly detection",
    },
    {
        "name": "Models",
        "description": "Model management and versioning",
    },
    {
        "name": "Metrics",
        "description": "System metrics and monitoring",
    },
    {
        "name": "WebSocket",
        "description": "Real-time communication endpoints",
    },
]

# ============================================================================
# Security Schemes
# ============================================================================

SECURITY_SCHEMES = {
    "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "JWT access token from /auth/login endpoint"
    },
    "apiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API key for service-to-service authentication"
    }
}

# ============================================================================
# Contact & License
# ============================================================================

CONTACT_INFO = {
    "name": "NEURECTOMY Development Team",
    "url": "https://neurectomy.io",
    "email": "api@neurectomy.io"
}

LICENSE_INFO = {
    "name": "Proprietary",
    "url": "https://neurectomy.io/license"
}

# ============================================================================
# External Documentation
# ============================================================================

EXTERNAL_DOCS = {
    "description": "Full Documentation",
    "url": "https://docs.neurectomy.io"
}

# ============================================================================
# OpenAPI Configuration Factory
# ============================================================================

def get_openapi_config() -> Dict[str, Any]:
    """
    Get complete OpenAPI configuration for FastAPI.
    
    Returns:
        Dictionary with all OpenAPI configuration options.
    """
    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "openapi_tags": TAGS_METADATA,
        "contact": CONTACT_INFO,
        "license_info": LICENSE_INFO,
        "external_docs": EXTERNAL_DOCS,
    }


def customize_openapi_schema(openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Customize the generated OpenAPI schema.
    
    Args:
        openapi_schema: The auto-generated OpenAPI schema
        
    Returns:
        Modified OpenAPI schema with customizations
    """
    # Add security schemes
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    openapi_schema["components"]["securitySchemes"] = SECURITY_SCHEMES
    
    # Add global security requirement
    openapi_schema["security"] = [
        {"bearerAuth": []},
        {"apiKeyAuth": []}
    ]
    
    # Add servers
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Local Development"
        },
        {
            "url": "https://dev-ml-api.neurectomy.io",
            "description": "Development"
        },
        {
            "url": "https://api.neurectomy.io",
            "description": "Production"
        }
    ]
    
    # Add x-logo extension for ReDoc
    openapi_schema["info"]["x-logo"] = {
        "url": "https://neurectomy.io/logo.png",
        "altText": "NEURECTOMY Logo"
    }
    
    return openapi_schema


# ============================================================================
# Example Schemas for Documentation
# ============================================================================

TRAINING_REQUEST_EXAMPLE = {
    "agent_id": "550e8400-e29b-41d4-a716-446655440000",
    "model_type": "transformer",
    "config": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "hidden_size": 256,
        "num_layers": 4,
        "dropout": 0.1
    },
    "dataset": {
        "type": "custom",
        "path": "/data/training/agent_550e8400"
    }
}

TRAINING_RESPONSE_EXAMPLE = {
    "job_id": "job_123456789",
    "status": "started",
    "agent_id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2025-01-15T10:30:00Z",
    "estimated_completion": "2025-01-15T12:30:00Z"
}

FORECAST_REQUEST_EXAMPLE = {
    "metric": "accuracy",
    "periods": 30,
    "method": "exponential_smoothing",
    "confidence_level": 0.95
}

ANOMALY_DETECTION_EXAMPLE = {
    "values": [1.0, 1.1, 0.9, 1.2, 1.0, 5.5, 1.1, 0.9],
    "metric_type": "latency",
    "sensitivity": 0.95,
    "min_samples": 10
}
