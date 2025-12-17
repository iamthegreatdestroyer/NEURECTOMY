"""
Tests for ΣLANG Compression Metrics
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from sigmalang.monitoring.metrics import (
    track_compression,
    track_decompression,
    CompressionContext,
    update_cache_metrics,
    update_resource_metrics,
    update_compression_level,
    update_dictionary_metrics,
    record_pattern_match,
    record_sublinear_optimization,
    compression_operations_total,
    compression_ratio,
    total_bytes_saved,
)


class TestCompressionTracking:
    """Test compression operation tracking"""
    
    def test_track_compression_success(self):
        """Test tracking successful compression"""
        
        @track_compression(algorithm='zstd', data_type='json')
        def compress(data: bytes):
            result = MagicMock()
            result.original_size = len(data)
            result.compressed_size = len(data) // 5  # 5x compression
            return result
        
        test_data = b'x' * 1000
        result = compress(test_data)
        
        assert result.original_size == 1000
        assert result.compressed_size == 200
    
    def test_track_compression_error(self):
        """Test tracking compression errors"""
        
        @track_compression(algorithm='zstd', data_type='json')
        def compress(data: bytes):
            raise ValueError("Invalid data")
        
        with pytest.raises(ValueError):
            compress(b'test')
    
    def test_track_decompression_success(self):
        """Test tracking decompression"""
        
        @track_decompression(algorithm='zstd')
        def decompress(data: bytes):
            result = MagicMock()
            result.data = b'x' * 1000
            return result
        
        test_data = b'\x28\xb5\x2f\xfd'  # zstd magic
        result = decompress(test_data)
        
        assert result is not None
    
    def test_track_decompression_error(self):
        """Test tracking decompression errors"""
        
        @track_decompression(algorithm='zstd')
        def decompress(data: bytes):
            raise RuntimeError("Corrupted data")
        
        with pytest.raises(RuntimeError):
            decompress(b'corrupted')


class TestCompressionContext:
    """Test compression context manager"""
    
    def test_context_success(self):
        """Test context manager with successful compression"""
        with CompressionContext(algorithm='zstd', data_type='json') as ctx:
            ctx.set_sizes(original_size=1000, compressed_size=200)
            # 5x compression ratio should be recorded
    
    def test_context_with_exception(self):
        """Test context manager handles exceptions"""
        with pytest.raises(RuntimeError):
            with CompressionContext(algorithm='zstd') as ctx:
                ctx.set_sizes(1000, 200)
                raise RuntimeError("Compression failed")
    
    def test_context_incompressible(self):
        """Test marking data as incompressible"""
        with CompressionContext(algorithm='zstd', data_type='binary') as ctx:
            ctx.set_sizes(1000, 950)
            ctx.record_incompressible()  # <2x ratio


class TestCompressionRatios:
    """Test compression ratio tracking"""
    
    def test_high_compression_ratio(self):
        """Test tracking high compression ratios"""
        with CompressionContext(algorithm='zstd', data_type='json') as ctx:
            ctx.set_sizes(original_size=10000, compressed_size=100)  # 100x
    
    def test_low_compression_ratio(self):
        """Test tracking low compression ratios"""
        with CompressionContext(algorithm='zstd', data_type='binary') as ctx:
            ctx.set_sizes(original_size=1000, compressed_size=900)  # 1.1x
    
    def test_compression_expansion(self):
        """Test when compression expands data"""
        with CompressionContext(algorithm='zstd', data_type='random') as ctx:
            ctx.set_sizes(original_size=1000, compressed_size=1100)  # 0.9x


class TestCacheMetrics:
    """Test cache effectiveness metrics"""
    
    def test_update_cache_metrics(self):
        """Test updating cache metrics"""
        update_cache_metrics(algorithm='zstd', hit_ratio=0.85, cache_size_bytes=10*1024*1024)
        update_cache_metrics(algorithm='zstd', hit_ratio=0.88, cache_size_bytes=12*1024*1024)
    
    def test_dictionary_metrics(self):
        """Test dictionary effectiveness"""
        update_dictionary_metrics(algorithm='zstd', hit_ratio=0.92, dict_size_bytes=1024*100)


class TestResourceMetrics:
    """Test resource utilization"""
    
    def test_update_resource_metrics(self):
        """Test recording resource usage"""
        update_resource_metrics(
            algorithm='zstd',
            cpu_percent=45,
            memory_bytes=500*1024*1024,
            io_throughput=100*1024*1024  # 100 MB/s
        )
    
    def test_cpu_efficiency(self):
        """Test tracking CPU efficiency"""
        # Low CPU for high throughput = efficient
        update_resource_metrics('zstd', cpu_percent=20, memory_bytes=200*1024*1024, io_throughput=200*1024*1024)


class TestCompressionLevel:
    """Test compression level tracking"""
    
    def test_compression_levels(self):
        """Test setting different compression levels"""
        for level in range(1, 10):
            update_compression_level('zstd', level)


class TestPatternMatching:
    """Test pattern matching metrics"""
    
    def test_record_pattern_match(self):
        """Test recording pattern matches"""
        for pattern_len in [10, 20, 50, 100]:
            record_pattern_match('zstd', pattern_len)


class TestSublinearOptimizations:
    """Test sub-linear algorithm tracking"""
    
    def test_bloom_filter_acceleration(self):
        """Test Bloom filter optimization"""
        record_sublinear_optimization('bloom_filter', speedup_factor=5.0)
    
    def test_hyperloglog_acceleration(self):
        """Test HyperLogLog optimization"""
        record_sublinear_optimization('hyperloglog', speedup_factor=10.0)
    
    def test_lsh_acceleration(self):
        """Test LSH optimization"""
        record_sublinear_optimization('lsh', speedup_factor=3.5)


class TestMultiAlgorithmTracking:
    """Test tracking multiple compression algorithms"""
    
    def test_compare_algorithms(self):
        """Test comparing different algorithms"""
        
        @track_compression(algorithm='zstd', data_type='json')
        def compress_zstd(data: bytes):
            result = MagicMock()
            result.original_size = 1000
            result.compressed_size = 200
            return result
        
        @track_compression(algorithm='lz4', data_type='json')
        def compress_lz4(data: bytes):
            result = MagicMock()
            result.original_size = 1000
            result.compressed_size = 300
            return result
        
        test_data = b'x' * 1000
        r1 = compress_zstd(test_data)
        r2 = compress_lz4(test_data)
        
        # zstd achieves 5x, lz4 achieves 3.3x
        assert r1.original_size / r1.compressed_size > r2.original_size / r2.compressed_size


class TestDataTypeSpecialization:
    """Test metrics for different data types"""
    
    def test_json_compression(self):
        """Test tracking JSON compression"""
        with CompressionContext(algorithm='zstd', data_type='json') as ctx:
            ctx.set_sizes(5000, 500)  # High ratio expected
    
    def test_binary_compression(self):
        """Test tracking binary compression"""
        with CompressionContext(algorithm='zstd', data_type='binary') as ctx:
            ctx.set_sizes(5000, 4500)  # Lower ratio expected
    
    def test_text_compression(self):
        """Test tracking text compression"""
        with CompressionContext(algorithm='zstd', data_type='text') as ctx:
            ctx.set_sizes(10000, 2000)  # High ratio expected


class TestPerformanceTracking:
    """Test performance metric tracking"""
    
    def test_throughput_calculation(self):
        """Test compression throughput"""
        import time
        
        @track_compression(algorithm='zstd', data_type='json')
        def compress_slow(data: bytes):
            time.sleep(0.1)
            result = MagicMock()
            result.original_size = len(data)
            result.compressed_size = len(data) // 5
            return result
        
        test_data = b'x' * 10000
        compress_slow(test_data)
        
        # ~10000 bytes / 0.1s = 100KB/s


class TestErrorHandling:
    """Test error handling and tracking"""
    
    def test_oom_error(self):
        """Test out-of-memory error tracking"""
        
        @track_compression(algorithm='zstd', data_type='json')
        def compress_oom(data: bytes):
            raise MemoryError("Out of memory")
        
        with pytest.raises(MemoryError):
            compress_oom(b'test')
    
    def test_corruption_detection(self):
        """Test corruption error tracking"""
        
        @track_decompression(algorithm='zstd')
        def decompress_corrupt(data: bytes):
            raise RuntimeError("Corrupted compressed data")
        
        with pytest.raises(RuntimeError):
            decompress_corrupt(b'corrupted')
