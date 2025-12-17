"""
Prometheus Metrics for ΣLANG Compression Service
Performance monitoring for compression operations with sub-linear optimization tracking
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
from typing import Callable, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Compression Operation Metrics
# ============================================================================

compression_operations_total = Counter(
    'sigmalang_compression_operations_total',
    'Total compression operations',
    ['algorithm', 'data_type', 'status'],  # status: success, error, skipped
    help='Count of compression operations by algorithm and data type'
)

decompression_operations_total = Counter(
    'sigmalang_decompression_operations_total',
    'Total decompression operations',
    ['algorithm', 'status'],  # status: success, error
    help='Count of decompression operations'
)

compression_skipped_total = Counter(
    'sigmalang_compression_skipped_total',
    'Compression operations skipped (data already compressed)',
    ['algorithm'],
    help='Operations skipped due to pre-compression'
)


# ============================================================================
# Compression Performance Metrics
# ============================================================================

compression_duration_seconds = Histogram(
    'sigmalang_compression_duration_seconds',
    'Compression operation duration in seconds',
    ['algorithm', 'data_type'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
    help='Time required for compression'
)

decompression_duration_seconds = Histogram(
    'sigmalang_decompression_duration_seconds',
    'Decompression operation duration in seconds',
    ['algorithm'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    help='Time required for decompression'
)

compression_throughput_bytes_per_second = Gauge(
    'sigmalang_compression_throughput_bytes_per_second',
    'Compression throughput (bytes/second)',
    ['algorithm'],
    help='Real-time compression speed'
)

decompression_throughput_bytes_per_second = Gauge(
    'sigmalang_decompression_throughput_bytes_per_second',
    'Decompression throughput (bytes/second)',
    ['algorithm'],
    help='Real-time decompression speed'
)


# ============================================================================
# Compression Ratio and Size Metrics
# ============================================================================

compression_ratio = Histogram(
    'sigmalang_compression_ratio',
    'Compression ratio (original/compressed size)',
    ['algorithm', 'data_type'],
    buckets=(1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0),
    help='Data size reduction achieved (5-50x is typical)'
)

original_data_size_bytes = Histogram(
    'sigmalang_original_data_size_bytes',
    'Original data size in bytes',
    ['algorithm', 'data_type'],
    buckets=(100, 1024, 10*1024, 100*1024, 1024*1024, 10*1024*1024, 100*1024*1024, 1024*1024*1024),
    help='Input data size distribution'
)

compressed_data_size_bytes = Histogram(
    'sigmalang_compressed_data_size_bytes',
    'Compressed data size in bytes',
    ['algorithm', 'data_type'],
    buckets=(100, 1024, 10*1024, 100*1024, 1024*1024, 10*1024*1024, 100*1024*1024, 1024*1024*1024),
    help='Output data size distribution'
)

total_bytes_compressed = Counter(
    'sigmalang_total_bytes_compressed',
    'Total bytes processed for compression',
    ['algorithm', 'data_type'],
    help='Cumulative input bytes'
)

total_bytes_saved = Counter(
    'sigmalang_total_bytes_saved',
    'Total bytes saved through compression',
    ['algorithm', 'data_type'],
    help='Cumulative bytes of space savings (original - compressed)'
)

space_efficiency_ratio = Gauge(
    'sigmalang_space_efficiency_ratio',
    'Space efficiency (saved/original)',
    ['algorithm'],
    help='Percentage of space saved (0.0 to 1.0)'
)


# ============================================================================
# Algorithm-Specific Metrics
# ============================================================================

# For algorithms supporting compression levels
compression_level = Gauge(
    'sigmalang_compression_level',
    'Compression level (1-9 or algorithm-specific)',
    ['algorithm'],
    help='Current compression aggressiveness setting'
)

# For dictionary-based compression
dictionary_hit_ratio = Gauge(
    'sigmalang_dictionary_hit_ratio',
    'Dictionary cache hit rate (0.0 to 1.0)',
    ['algorithm'],
    help='Effectiveness of dictionary/vocabulary'
)

dictionary_size_bytes = Gauge(
    'sigmalang_dictionary_size_bytes',
    'Dictionary size in bytes',
    ['algorithm'],
    help='Size of compression dictionary'
)

# For pattern-based compression
pattern_matches_total = Counter(
    'sigmalang_pattern_matches_total',
    'Number of pattern matches found',
    ['algorithm'],
    help='Pattern matching effectiveness'
)

average_pattern_length = Gauge(
    'sigmalang_average_pattern_length',
    'Average matched pattern length',
    ['algorithm'],
    help='Pattern length affecting compression'
)


# ============================================================================
# Cache and Optimization Metrics
# ============================================================================

compression_cache_hit_ratio = Gauge(
    'sigmalang_compression_cache_hit_ratio',
    'Compression cache hit rate',
    ['algorithm'],
    help='Effectiveness of result caching'
)

cache_memory_bytes = Gauge(
    'sigmalang_cache_memory_bytes',
    'Memory used by compression cache',
    ['algorithm'],
    help='Current cache size'
)

# Sub-linear algorithm tracking
sublinear_accelerations_total = Counter(
    'sigmalang_sublinear_accelerations_total',
    'Times sub-linear optimizations were applied',
    ['optimization_type'],  # bloom_filter, hyperloglog, lsh, etc.
    help='Count of sub-linear optimizations used'
)

sublinear_speedup_factor = Gauge(
    'sigmalang_sublinear_speedup_factor',
    'Speedup factor from sub-linear optimizations',
    ['optimization_type'],
    help='How much faster operations are with optimization'
)


# ============================================================================
# Error Tracking
# ============================================================================

compression_errors_total = Counter(
    'sigmalang_compression_errors_total',
    'Compression errors',
    ['algorithm', 'error_type'],  # error_type: invalid_input, oom, corrupted, timeout
    help='Error counts by type'
)

incompressible_data_total = Counter(
    'sigmalang_incompressible_data_total',
    'Data that could not be meaningfully compressed',
    ['algorithm', 'data_type'],
    help='Input that yielded < 1.5x compression ratio'
)


# ============================================================================
# Resource Utilization
# ============================================================================

compression_cpu_usage_percent = Gauge(
    'sigmalang_compression_cpu_usage_percent',
    'CPU usage during compression (0-100)',
    ['algorithm'],
    help='CPU utilization percentage'
)

compression_memory_usage_bytes = Gauge(
    'sigmalang_compression_memory_usage_bytes',
    'Memory used during compression',
    ['algorithm'],
    help='Peak memory consumption'
)

compression_io_throughput_bytes_per_second = Gauge(
    'sigmalang_compression_io_throughput_bytes_per_second',
    'I/O throughput for compression',
    ['algorithm'],
    help='Disk I/O rate'
)


# ============================================================================
# System Information
# ============================================================================

system_info = Info(
    'sigmalang_system',
    'ΣLANG compression system information',
)

algorithm_info = Info(
    'sigmalang_algorithm',
    'Information about available compression algorithms',
)


# ============================================================================
# Decorators for Compression Tracking
# ============================================================================

def track_compression(algorithm: str = 'default', data_type: str = 'generic'):
    """
    Decorator to track compression operations
    
    Usage:
        @track_compression(algorithm='zstd', data_type='json')
        def compress_data(data: bytes):
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            original_size = 0
            compressed_size = 0
            
            try:
                # Get original size if available
                if args and isinstance(args[0], (bytes, str)):
                    original_size = len(args[0])
                
                result = func(*args, **kwargs)
                
                # Extract sizes from result if available
                if hasattr(result, 'original_size'):
                    original_size = result.original_size
                if hasattr(result, 'compressed_size'):
                    compressed_size = result.compressed_size
                
                # Calculate and record metrics
                if original_size > 0 and compressed_size > 0:
                    ratio = original_size / compressed_size
                    compression_ratio.labels(algorithm=algorithm, data_type=data_type).observe(ratio)
                    
                    bytes_saved = original_size - compressed_size
                    total_bytes_saved.labels(algorithm=algorithm, data_type=data_type).inc(bytes_saved)
                    
                    efficiency = bytes_saved / original_size
                    # Note: Space efficiency is aggregate, would need to track separately
                
                return result
            except ValueError:
                status = "error"
                compression_errors_total.labels(algorithm=algorithm, error_type='invalid_input').inc()
                raise
            except MemoryError:
                status = "error"
                compression_errors_total.labels(algorithm=algorithm, error_type='oom').inc()
                raise
            except Exception as e:
                status = "error"
                compression_errors_total.labels(algorithm=algorithm, error_type='unknown').inc()
                logger.error(f"Compression error: {e}")
                raise
            finally:
                duration = time.time() - start_time
                compression_operations_total.labels(algorithm=algorithm, data_type=data_type, status=status).inc()
                compression_duration_seconds.labels(algorithm=algorithm, data_type=data_type).observe(duration)
                
                # Record sizes if available
                if original_size > 0:
                    original_data_size_bytes.labels(algorithm=algorithm, data_type=data_type).observe(original_size)
                    total_bytes_compressed.labels(algorithm=algorithm, data_type=data_type).inc(original_size)
                
                if compressed_size > 0:
                    compressed_data_size_bytes.labels(algorithm=algorithm, data_type=data_type).observe(compressed_size)
                
                # Record throughput
                if duration > 0 and original_size > 0:
                    throughput = original_size / duration
                    compression_throughput_bytes_per_second.labels(algorithm=algorithm).set(throughput)
        
        return wrapper
    return decorator


def track_decompression(algorithm: str = 'default'):
    """
    Decorator to track decompression operations
    
    Usage:
        @track_decompression(algorithm='zstd')
        def decompress_data(data: bytes):
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            compressed_size = 0
            
            try:
                # Get compressed size
                if args and isinstance(args[0], (bytes, str)):
                    compressed_size = len(args[0])
                
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                compression_errors_total.labels(algorithm=algorithm, error_type='corrupted').inc()
                logger.error(f"Decompression error: {e}")
                raise
            finally:
                duration = time.time() - start_time
                decompression_operations_total.labels(algorithm=algorithm, status=status).inc()
                decompression_duration_seconds.labels(algorithm=algorithm).observe(duration)
                
                # Record throughput
                if duration > 0 and compressed_size > 0:
                    throughput = compressed_size / duration
                    decompression_throughput_bytes_per_second.labels(algorithm=algorithm).set(throughput)
        
        return wrapper
    return decorator


# ============================================================================
# Context Managers
# ============================================================================

class CompressionContext:
    """Context manager for tracking compression operations"""
    
    def __init__(self, algorithm: str = 'default', data_type: str = 'generic'):
        self.algorithm = algorithm
        self.data_type = data_type
        self.start_time = None
        self.original_size = 0
        self.compressed_size = 0
        self.status = "success"
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            self.status = "error"
            error_type = exc_type.__name__.lower()
            compression_errors_total.labels(algorithm=self.algorithm, error_type=error_type).inc()
        
        compression_operations_total.labels(
            algorithm=self.algorithm,
            data_type=self.data_type,
            status=self.status
        ).inc()
        
        compression_duration_seconds.labels(
            algorithm=self.algorithm,
            data_type=self.data_type
        ).observe(duration)
        
        # Record sizes
        if self.original_size > 0:
            original_data_size_bytes.labels(
                algorithm=self.algorithm,
                data_type=self.data_type
            ).observe(self.original_size)
            total_bytes_compressed.labels(
                algorithm=self.algorithm,
                data_type=self.data_type
            ).inc(self.original_size)
        
        if self.compressed_size > 0:
            compressed_data_size_bytes.labels(
                algorithm=self.algorithm,
                data_type=self.data_type
            ).observe(self.compressed_size)
            
            # Calculate and record compression ratio
            if self.original_size > 0:
                ratio = self.original_size / self.compressed_size
                compression_ratio.labels(
                    algorithm=self.algorithm,
                    data_type=self.data_type
                ).observe(ratio)
                
                # Track bytes saved
                bytes_saved = self.original_size - self.compressed_size
                total_bytes_saved.labels(
                    algorithm=self.algorithm,
                    data_type=self.data_type
                ).inc(bytes_saved)
        
        # Record throughput
        if duration > 0 and self.original_size > 0:
            throughput = self.original_size / duration
            compression_throughput_bytes_per_second.labels(algorithm=self.algorithm).set(throughput)
        
        return False  # Don't suppress exceptions
    
    def set_sizes(self, original_size: int, compressed_size: int):
        """Set original and compressed sizes"""
        self.original_size = original_size
        self.compressed_size = compressed_size
    
    def record_incompressible(self):
        """Mark data as incompressible"""
        incompressible_data_total.labels(
            algorithm=self.algorithm,
            data_type=self.data_type
        ).inc()


# ============================================================================
# Helper Functions
# ============================================================================

def record_sublinear_optimization(optimization_type: str, speedup_factor: float):
    """Record use of sub-linear optimization"""
    sublinear_accelerations_total.labels(optimization_type=optimization_type).inc()
    sublinear_speedup_factor.labels(optimization_type=optimization_type).set(speedup_factor)


def update_cache_metrics(algorithm: str, hit_ratio: float, cache_size_bytes: int):
    """Update cache performance metrics"""
    compression_cache_hit_ratio.labels(algorithm=algorithm).set(hit_ratio)
    cache_memory_bytes.labels(algorithm=algorithm).set(cache_size_bytes)


def update_resource_metrics(algorithm: str, cpu_percent: int, memory_bytes: int, io_throughput: float):
    """Update resource utilization metrics"""
    compression_cpu_usage_percent.labels(algorithm=algorithm).set(cpu_percent)
    compression_memory_usage_bytes.labels(algorithm=algorithm).set(memory_bytes)
    compression_io_throughput_bytes_per_second.labels(algorithm=algorithm).set(io_throughput)


def update_compression_level(algorithm: str, level: int):
    """Update compression level setting"""
    compression_level.labels(algorithm=algorithm).set(level)


def update_dictionary_metrics(algorithm: str, hit_ratio: float, dict_size_bytes: int):
    """Update dictionary effectiveness metrics"""
    dictionary_hit_ratio.labels(algorithm=algorithm).set(hit_ratio)
    dictionary_size_bytes.labels(algorithm=algorithm).set(dict_size_bytes)


def record_pattern_match(algorithm: str, pattern_length: int):
    """Record pattern match for pattern-based compression"""
    pattern_matches_total.labels(algorithm=algorithm).inc()
