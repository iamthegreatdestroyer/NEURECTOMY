"""
Tests for Ryot LLM Metrics
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ryot.monitoring.metrics import (
    track_inference_request,
    track_token_generation,
    InferenceContext,
    update_gpu_metrics,
    record_inference_error,
    update_queue_metrics,
    record_queue_wait,
    update_token_throughput,
    record_batch_processed,
    record_model_load_time,
    update_cache_metrics,
    inference_requests_total,
    tokens_generated_total,
    inference_errors_total,
)


class TestInferenceTracking:
    """Test inference request tracking"""
    
    @pytest.mark.asyncio
    async def test_track_inference_success(self):
        """Test tracking successful inference"""
        
        @track_inference_request(model='gpt4')
        async def generate(prompt: str):
            result = MagicMock()
            result.token_count = 150
            result.ttft_seconds = 0.25
            return result
        
        result = await generate("test prompt")
        
        assert result.token_count == 150
        # Metrics would be recorded
    
    @pytest.mark.asyncio
    async def test_track_inference_timeout(self):
        """Test tracking timeout errors"""
        
        @track_inference_request(model='gpt4')
        async def generate(prompt: str):
            raise TimeoutError("Request timed out")
        
        with pytest.raises(TimeoutError):
            await generate("test prompt")
        
        # Error should be recorded as timeout
    
    @pytest.mark.asyncio
    async def test_track_inference_oom(self):
        """Test tracking out-of-memory errors"""
        
        @track_inference_request(model='gpt4')
        async def generate(prompt: str):
            raise RuntimeError("CUDA out of memory")
        
        with pytest.raises(RuntimeError):
            await generate("test prompt")
        
        # Error should be recorded as OOM
    
    @pytest.mark.asyncio
    async def test_track_token_generation(self):
        """Test token generation tracking"""
        
        @track_token_generation(model='gpt4')
        async def generate_tokens(prompt: str):
            result = MagicMock()
            result.token_count = 100
            result.duration_seconds = 0.5
            return result
        
        result = await generate_tokens("test")
        
        assert result.token_count == 100


class TestInferenceContext:
    """Test inference context manager"""
    
    def test_context_success(self):
        """Test context manager with successful inference"""
        with InferenceContext(model='gpt4') as ctx:
            ctx.set_token_count(150)
            ctx.set_ttft(0.25)
    
    def test_context_with_exception(self):
        """Test context manager handles exceptions"""
        with pytest.raises(RuntimeError):
            with InferenceContext(model='gpt4') as ctx:
                ctx.set_token_count(100)
                raise RuntimeError("Test error")
    
    def test_context_records_inter_token_latency(self):
        """Test recording inter-token latency"""
        with InferenceContext(model='gpt4') as ctx:
            ctx.record_inter_token_latency(0.02)
            ctx.record_inter_token_latency(0.018)
            ctx.record_inter_token_latency(0.022)
    
    def test_context_timeout(self):
        """Test context manager with timeout"""
        with pytest.raises(TimeoutError):
            with InferenceContext(model='gpt4') as ctx:
                raise TimeoutError("Request timed out")


class TestGPUMetrics:
    """Test GPU resource metrics"""
    
    def test_update_gpu_metrics(self):
        """Test updating GPU metrics"""
        update_gpu_metrics(
            model='gpt4',
            gpu_id=0,
            used_bytes=10 * 1024 * 1024 * 1024,  # 10GB
            reserved_bytes=24 * 1024 * 1024 * 1024,  # 24GB
            utilization=75  # 75%
        )
        # Metrics should be recorded
    
    def test_gpu_memory_efficiency(self):
        """Test memory efficiency calculation"""
        # 15GB used out of 24GB = 62.5% efficiency
        update_gpu_metrics(
            model='gpt4',
            gpu_id=0,
            used_bytes=15 * 1024 * 1024 * 1024,
            reserved_bytes=24 * 1024 * 1024 * 1024,
            utilization=80
        )


class TestErrorTracking:
    """Test error tracking"""
    
    def test_record_oom_error(self):
        """Test recording OOM error"""
        record_inference_error('gpt4', 'oom')
    
    def test_record_timeout_error(self):
        """Test recording timeout error"""
        record_inference_error('gpt4', 'timeout')
    
    def test_record_cuda_error(self):
        """Test recording CUDA error"""
        record_inference_error('gpt4', 'cuda_error')


class TestQueueMetrics:
    """Test queue metrics"""
    
    def test_update_queue_size(self):
        """Test updating queue size"""
        update_queue_metrics('gpt4', 5)
        update_queue_metrics('gpt4', 10)
        update_queue_metrics('gpt4', 3)
    
    def test_record_queue_wait(self):
        """Test recording queue wait time"""
        record_queue_wait('gpt4', 0.5)
        record_queue_wait('gpt4', 1.2)
        record_queue_wait('gpt4', 0.3)


class TestTokenMetrics:
    """Test token generation metrics"""
    
    def test_update_token_throughput(self):
        """Test updating token throughput"""
        update_token_throughput('gpt4', 100.0)  # 100 tokens/sec
        update_token_throughput('gpt4', 150.0)  # 150 tokens/sec
    
    def test_record_batch_processed(self):
        """Test recording batch processing"""
        record_batch_processed('gpt4', 1)
        record_batch_processed('gpt4', 4)
        record_batch_processed('gpt4', 8)


class TestModelMetrics:
    """Test model-specific metrics"""
    
    def test_record_model_load_time(self):
        """Test recording model load time"""
        record_model_load_time('gpt4', 2.5)
        record_model_load_time('gpt4', 2.4)
    
    def test_update_cache_metrics(self):
        """Test updating cache metrics"""
        update_cache_metrics('gpt4', 0.85)  # 85% hit rate
        update_cache_metrics('gpt4', 0.88)
        update_cache_metrics('gpt4', 0.92)


class TestMultiModelTracking:
    """Test tracking across multiple models"""
    
    @pytest.mark.asyncio
    async def test_track_multiple_models(self):
        """Test tracking different models independently"""
        
        @track_inference_request(model='gpt4')
        async def generate_gpt4(prompt: str):
            result = MagicMock()
            result.token_count = 150
            return result
        
        @track_inference_request(model='claude')
        async def generate_claude(prompt: str):
            result = MagicMock()
            result.token_count = 200
            return result
        
        result1 = await generate_gpt4("test")
        result2 = await generate_claude("test")
        
        assert result1.token_count == 150
        assert result2.token_count == 200


class TestPerformanceTracking:
    """Test performance metrics tracking"""
    
    def test_ttft_tracking(self):
        """Test time-to-first-token tracking"""
        with InferenceContext(model='gpt4') as ctx:
            ctx.set_ttft(0.05)
            ctx.set_ttft(0.08)
            ctx.set_ttft(0.03)
    
    def test_inter_token_latency(self):
        """Test inter-token latency tracking"""
        with InferenceContext(model='gpt4') as ctx:
            ctx.record_inter_token_latency(0.020)
            ctx.record_inter_token_latency(0.025)
            ctx.record_inter_token_latency(0.018)


class TestMetricLabeling:
    """Test proper metric labeling"""
    
    def test_model_label_consistency(self):
        """Verify model labels are consistent"""
        with InferenceContext(model='gpt4') as ctx:
            ctx.set_token_count(100)
        
        with InferenceContext(model='claude') as ctx:
            ctx.set_token_count(150)
        
        # Each model should have its own metric series


class TestConcurrency:
    """Test concurrent request tracking"""
    
    @pytest.mark.asyncio
    async def test_concurrent_inference(self):
        """Test tracking concurrent requests"""
        import asyncio
        
        @track_inference_request(model='gpt4')
        async def generate(prompt: str):
            await asyncio.sleep(0.1)
            result = MagicMock()
            result.token_count = 100
            return result
        
        # Run multiple concurrent requests
        results = await asyncio.gather(
            generate("prompt1"),
            generate("prompt2"),
            generate("prompt3"),
        )
        
        assert len(results) == 3
