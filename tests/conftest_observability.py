"""
Observability Stack Test Fixtures and Mocking Infrastructure
Provides shared test configuration for metrics, tracing, logging, and alerting tests
"""

import pytest
import asyncio
import json
import logging
from typing import Generator, Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from io import StringIO
import sys

# Prometheus
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
)

# OpenTelemetry
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import get_tracer_provider
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.jaeger import JaegerPropagator
from opentelemetry.context import get_current

# Logging
import structlog


# ============================================================================
# PROMETHEUS MOCK FIXTURES
# ============================================================================

@pytest.fixture
def prometheus_registry() -> Generator[CollectorRegistry, None, None]:
    """Isolated Prometheus registry for testing"""
    registry = CollectorRegistry()
    yield registry
    # Cleanup: clear all metrics
    registry._collector_to_names.clear()
    registry._names_to_collectors.clear()


@pytest.fixture
def prometheus_test_metrics(prometheus_registry: CollectorRegistry) -> Dict[str, Any]:
    """Create test metrics with isolated registry"""
    return {
        'counter': Counter(
            'test_requests_total',
            'Test request counter',
            ['method', 'status'],
            registry=prometheus_registry
        ),
        'histogram': Histogram(
            'test_request_duration_seconds',
            'Test request duration',
            ['method'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
            registry=prometheus_registry
        ),
        'gauge': Gauge(
            'test_active_connections',
            'Test active connections',
            ['service'],
            registry=prometheus_registry
        ),
        'registry': prometheus_registry,
    }


@pytest.fixture
def mock_prometheus_scraper() -> Mock:
    """Mock Prometheus scraper for integration testing"""
    scraper = Mock()
    scraper.scrape = AsyncMock(return_value={
        'http_requests_total': [
            {'labels': {'method': 'GET', 'status': '200'}, 'value': 100},
            {'labels': {'method': 'POST', 'status': '201'}, 'value': 50},
        ],
        'http_request_duration_seconds_bucket': [
            {'labels': {'method': 'GET', 'le': '0.1'}, 'value': 80},
            {'labels': {'method': 'GET', 'le': '0.5'}, 'value': 95},
        ],
    })
    return scraper


@pytest.fixture
def label_cardinality_tracker() -> Dict[str, int]:
    """Track label cardinality for explosion detection"""
    return {
        'max_labels_per_metric': 1000,
        'warnings': [],
        'critical': [],
    }


# ============================================================================
# OPENTELEMETRY / JAEGER MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_jaeger_exporter() -> Mock:
    """Mock Jaeger exporter"""
    exporter = Mock()
    exporter.export = Mock(return_value=True)
    exporter.shutdown = Mock()
    exporter.force_flush = Mock(return_value=True)
    return exporter


@pytest.fixture
def test_tracer_provider(mock_jaeger_exporter: Mock) -> TracerProvider:
    """Create isolated TracerProvider for testing"""
    resource = Resource.create({"service.name": "neurectomy-test"})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(mock_jaeger_exporter)
    )
    return tracer_provider


@pytest.fixture
def test_tracer(test_tracer_provider: TracerProvider):
    """Get tracer from test provider"""
    return test_tracer_provider.get_tracer(__name__)


@pytest.fixture
def mock_span_exporter() -> AsyncMock:
    """Mock OpenTelemetry span exporter"""
    exporter = AsyncMock()
    exporter.export = AsyncMock(return_value=True)
    exporter.shutdown = AsyncMock()
    exporter.force_flush = AsyncMock(return_value=True)
    return exporter


@pytest.fixture
def mock_trace_context() -> Dict[str, str]:
    """Mock trace context for correlation"""
    return {
        'trace_id': '4bf92f3577b34da6a3ce929d0e0e4736',
        'span_id': '00f067aa0ba902b7',
        'trace_flags': '01',
        'trace_state': 'congo=t61rcWpm1t1z',
    }


# ============================================================================
# STRUCTURED LOGGING MOCK FIXTURES
# ============================================================================

@pytest.fixture
def log_capture() -> StringIO:
    """Capture log output"""
    return StringIO()


@pytest.fixture
def structured_logger(log_capture: StringIO):
    """Configure structured logger for testing"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield logger
    
    logger.removeHandler(handler)


@pytest.fixture
def mock_loki_client() -> AsyncMock:
    """Mock Loki log aggregation client"""
    client = AsyncMock()
    client.push = AsyncMock(return_value={'status': 'ok'})
    client.query = AsyncMock(return_value={
        'data': {
            'result': [
                {
                    'stream': {'job': 'neurectomy', 'level': 'error'},
                    'values': [['1702828800000000000', 'error message']]
                }
            ]
        }
    })
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_log_aggregator() -> Mock:
    """Mock log aggregator for batch processing"""
    aggregator = Mock()
    aggregator.add_log = Mock()
    aggregator.aggregate = Mock(return_value=[
        {'level': 'INFO', 'count': 100},
        {'level': 'ERROR', 'count': 5},
        {'level': 'WARN', 'count': 15},
    ])
    aggregator.get_logs_by_service = Mock(return_value={
        'ml-service': 150,
        'rust-core': 200,
    })
    aggregator.get_logs_by_level = Mock(return_value={
        'DEBUG': 50,
        'INFO': 200,
        'WARN': 30,
        'ERROR': 20,
    })
    return aggregator


# ============================================================================
# ALERTING MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_alert_manager() -> Mock:
    """Mock AlertManager API"""
    manager = Mock()
    manager.get_alerts = Mock(return_value=[
        {
            'labels': {'alertname': 'HighErrorRate', 'severity': 'critical'},
            'state': 'firing',
            'generatorURL': 'http://prometheus:9090/graph',
        }
    ])
    manager.get_silences = Mock(return_value=[
        {
            'id': 'silence-123',
            'matchers': [{'name': 'alertname', 'value': 'HighErrorRate'}],
            'startsAt': '2025-12-17T00:00:00Z',
            'endsAt': '2025-12-17T04:00:00Z',
        }
    ])
    manager.create_silence = Mock(return_value={'silenceID': 'silence-456'})
    manager.delete_silence = Mock()
    return manager


@pytest.fixture
def mock_webhook_receiver() -> AsyncMock:
    """Mock webhook receiver for alert notifications"""
    receiver = AsyncMock()
    receiver.receive = AsyncMock()
    receiver.verify_signature = Mock(return_value=True)
    receiver.get_recent_alerts = Mock(return_value=[])
    return receiver


@pytest.fixture
def mock_pagerduty_client() -> AsyncMock:
    """Mock PagerDuty integration"""
    client = AsyncMock()
    client.create_incident = AsyncMock(return_value={
        'incident': {'id': 'incident-123', 'status': 'triggered'}
    })
    client.resolve_incident = AsyncMock()
    client.acknowledge_incident = AsyncMock()
    client.get_incidents = AsyncMock(return_value={'incidents': []})
    return client


@pytest.fixture
def mock_notification_router() -> Mock:
    """Mock notification routing"""
    router = Mock()
    router.route_alert = Mock(return_value=[
        {'channel': 'slack', 'status': 'delivered'},
        {'channel': 'email', 'status': 'delivered'},
    ])
    router.get_routing_rules = Mock(return_value=[
        {
            'matcher': {'severity': 'critical'},
            'routes': ['slack', 'pagerduty', 'email']
        }
    ])
    return router


# ============================================================================
# INTEGRATION TEST FIXTURES
# ============================================================================

@pytest.fixture
def mock_service_client() -> AsyncMock:
    """Mock service client for integration tests"""
    client = AsyncMock()
    client.request = AsyncMock(return_value={
        'status': 200,
        'data': {'result': 'success'},
        'headers': {
            'x-trace-id': '4bf92f3577b34da6a3ce929d0e0e4736',
            'x-span-id': '00f067aa0ba902b7',
        }
    })
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_database_connection() -> Mock:
    """Mock database connection for metrics"""
    conn = Mock()
    conn.query = Mock(return_value=[
        {'metric': 'cpu_usage', 'value': 45.2},
        {'metric': 'memory_usage', 'value': 2048},
    ])
    conn.execute = Mock()
    conn.close = Mock()
    return conn


@pytest.fixture
async def async_event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Provide event loop for async tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ============================================================================
# PERFORMANCE TESTING FIXTURES
# ============================================================================

@pytest.fixture
def performance_timer() -> Dict[str, float]:
    """Track performance metrics"""
    import time
    return {
        'start': time.perf_counter(),
        'measurements': [],
    }


@pytest.fixture
def performance_assertions():
    """Helper for performance assertions"""
    class PerformanceAssertions:
        @staticmethod
        def assert_latency_under(duration: float, threshold: float, message: str = ""):
            assert duration < threshold, (
                f"Latency {duration:.3f}s exceeded threshold {threshold:.3f}s. {message}"
            )
        
        @staticmethod
        def assert_memory_under(memory_mb: float, threshold_mb: float, message: str = ""):
            assert memory_mb < threshold_mb, (
                f"Memory {memory_mb:.1f}MB exceeded threshold {threshold_mb:.1f}MB. {message}"
            )
        
        @staticmethod
        def assert_throughput_above(items_per_sec: float, threshold: float, message: str = ""):
            assert items_per_sec >= threshold, (
                f"Throughput {items_per_sec:.1f}ops/s below threshold {threshold:.1f}. {message}"
            )
    
    return PerformanceAssertions()


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

@pytest.fixture
def metric_data_generator():
    """Generate test metric data"""
    class MetricGenerator:
        @staticmethod
        def generate_counter_data(name: str, count: int = 100) -> List[Dict]:
            return [
                {'name': f'{name}_total', 'value': count, 'timestamp': datetime.utcnow().isoformat()}
            ]
        
        @staticmethod
        def generate_histogram_data(name: str, samples: int = 100) -> List[Dict]:
            import random
            return [
                {
                    'name': f'{name}_seconds',
                    'value': random.uniform(0.01, 1.0),
                    'timestamp': datetime.utcnow().isoformat()
                }
                for _ in range(samples)
            ]
        
        @staticmethod
        def generate_gauge_data(name: str, value: float = 50.0) -> Dict:
            return {
                'name': name,
                'value': value,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    return MetricGenerator()


@pytest.fixture
def trace_data_generator():
    """Generate test trace data"""
    class TraceGenerator:
        @staticmethod
        def generate_trace(span_count: int = 5) -> Dict[str, Any]:
            import uuid
            trace_id = str(uuid.uuid4())
            return {
                'traceID': trace_id,
                'spans': [
                    {
                        'traceID': trace_id,
                        'spanID': str(uuid.uuid4()),
                        'operationName': f'operation_{i}',
                        'startTime': datetime.utcnow().timestamp() * 1e6,
                        'duration': 100000,  # 100ms in microseconds
                        'tags': {'http.status_code': 200},
                    }
                    for i in range(span_count)
                ]
            }
        
        @staticmethod
        def generate_span(operation: str, duration_us: int = 100000) -> Dict[str, Any]:
            import uuid
            return {
                'spanID': str(uuid.uuid4()),
                'operationName': operation,
                'startTime': datetime.utcnow().timestamp() * 1e6,
                'duration': duration_us,
            }
    
    return TraceGenerator()


@pytest.fixture
def log_data_generator():
    """Generate test log data"""
    class LogGenerator:
        @staticmethod
        def generate_log(level: str = 'INFO', message: str = "Test message") -> Dict[str, Any]:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'level': level,
                'message': message,
                'service': 'neurectomy',
                'correlation_id': '12345-67890',
            }
        
        @staticmethod
        def generate_logs(count: int, level: str = 'INFO') -> List[Dict[str, Any]]:
            return [
                LogGenerator.generate_log(level, f"Message {i}")
                for i in range(count)
            ]
    
    return LogGenerator()


@pytest.fixture
def alert_data_generator():
    """Generate test alert data"""
    class AlertGenerator:
        @staticmethod
        def generate_alert(name: str, severity: str = 'warning') -> Dict[str, Any]:
            return {
                'labels': {
                    'alertname': name,
                    'severity': severity,
                    'service': 'neurectomy',
                },
                'annotations': {
                    'summary': f'{name} alert',
                    'description': f'Alert for {name}',
                },
                'startsAt': datetime.utcnow().isoformat(),
                'status': 'firing',
            }
        
        @staticmethod
        def generate_alerts(count: int, severity: str = 'warning') -> List[Dict[str, Any]]:
            severities = ['critical', 'warning', 'info']
            return [
                AlertGenerator.generate_alert(f'Alert_{i}', severities[i % len(severities)])
                for i in range(count)
            ]
    
    return AlertGenerator()


# ============================================================================
# CLEANUP AND TEARDOWN
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Clear any remaining test data
    import gc
    gc.collect()
