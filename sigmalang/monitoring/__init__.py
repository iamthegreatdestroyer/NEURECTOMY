"""
ΣLANG Monitoring Module
Compression service metrics with sub-linear optimization tracking
"""

from .metrics import (
    # Operation counters
    compression_operations_total,
    decompression_operations_total,
    compression_skipped_total,
    
    # Performance
    compression_duration_seconds,
    decompression_duration_seconds,
    compression_throughput_bytes_per_second,
    decompression_throughput_bytes_per_second,
    
    # Compression ratio and sizes
    compression_ratio,
    original_data_size_bytes,
    compressed_data_size_bytes,
    total_bytes_compressed,
    total_bytes_saved,
    space_efficiency_ratio,
    
    # Algorithm-specific
    compression_level,
    dictionary_hit_ratio,
    dictionary_size_bytes,
    pattern_matches_total,
    average_pattern_length,
    
    # Cache and optimization
    compression_cache_hit_ratio,
    cache_memory_bytes,
    sublinear_accelerations_total,
    sublinear_speedup_factor,
    
    # Error tracking
    compression_errors_total,
    incompressible_data_total,
    
    # Resource utilization
    compression_cpu_usage_percent,
    compression_memory_usage_bytes,
    compression_io_throughput_bytes_per_second,
    
    # System info
    system_info,
    algorithm_info,
    
    # Decorators and context managers
    track_compression,
    track_decompression,
    CompressionContext,
    
    # Helpers
    record_sublinear_optimization,
    update_cache_metrics,
    update_resource_metrics,
    update_compression_level,
    update_dictionary_metrics,
    record_pattern_match,
)

__all__ = [
    "compression_operations_total",
    "decompression_operations_total",
    "compression_skipped_total",
    "compression_duration_seconds",
    "decompression_duration_seconds",
    "compression_throughput_bytes_per_second",
    "decompression_throughput_bytes_per_second",
    "compression_ratio",
    "original_data_size_bytes",
    "compressed_data_size_bytes",
    "total_bytes_compressed",
    "total_bytes_saved",
    "space_efficiency_ratio",
    "compression_level",
    "dictionary_hit_ratio",
    "dictionary_size_bytes",
    "pattern_matches_total",
    "average_pattern_length",
    "compression_cache_hit_ratio",
    "cache_memory_bytes",
    "sublinear_accelerations_total",
    "sublinear_speedup_factor",
    "compression_errors_total",
    "incompressible_data_total",
    "compression_cpu_usage_percent",
    "compression_memory_usage_bytes",
    "compression_io_throughput_bytes_per_second",
    "system_info",
    "algorithm_info",
    "track_compression",
    "track_decompression",
    "CompressionContext",
    "record_sublinear_optimization",
    "update_cache_metrics",
    "update_resource_metrics",
    "update_compression_level",
    "update_dictionary_metrics",
    "record_pattern_match",
]
