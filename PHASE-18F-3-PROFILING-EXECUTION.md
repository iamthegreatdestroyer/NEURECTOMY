# Phase 18F-3: PROFILING EXECUTION GUIDE

**Status:** 🔴 **NOW EXECUTING**  
**Date:** December 17, 2025  
**Duration:** 3-5 days (intensive profiling)  
**Team:** @VELOCITY (Lead)

---

## 🎯 Objective

Execute comprehensive profiling benchmarks across all 4 core services to identify performance bottlenecks and prioritize optimizations for Phase 18G.

**Success Criteria:**
- ✅ All 14 benchmarks executed successfully
- ✅ Baseline metrics established for each component
- ✅ Top 3 bottlenecks identified per service (with ROI scoring)
- ✅ Profiling data analyzed and documented
- ✅ Optimization roadmap created (Phase 18G-ready)

---

## 📊 Performance Targets

| Service | Component | Current Target | Benchmark |
|---------|-----------|----------------|-----------|
| **Ryot** | TTFT | TBD | <100ms | FirstTokenLatencyBenchmark |
| **Ryot** | Throughput | TBD | >50 tok/sec | TokenGenerationBenchmark |
| **ΣLANG** | Ratio | TBD | >3:1 | CompressionRatioBenchmark |
| **ΣLANG** | Speed | TBD | >100MB/s | CompressionThroughputBenchmark |
| **ΣVAULT** | Read Latency | TBD | <5ms p50, <10ms p99 | RSUReadBenchmark |
| **ΣVAULT** | Write Latency | TBD | <10ms p50, <20ms p99 | RSUWriteBenchmark |
| **Agents** | Task Latency | TBD | <50ms p99 | AgentTaskLatencyBenchmark |

---

## 🏃 Execution Plan (5 Days)

### **Day 1-2: Infrastructure Setup & Baseline Microbenchmarks**

#### 1. Environment Preparation (2 hours)
```bash
# Install profiling tools
pip install py-spy memory_profiler line_profiler scalene cProfile-flamegraph

# Configure benchmark environment
cd benchmarks/
python setup_profiling_env.py

# Enable kernel profiling
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

#### 2. Baseline Metrics - Microbenchmarks (Day 1 afternoon)
```bash
# Ryot benchmarks (30-45 minutes)
python runner.py \
  --suite microbenchmarks \
  --component ryot \
  --iterations 100 \
  --output results/day1/ryot_baseline.json

# Expected output:
# - FirstTokenLatencyBenchmark: TTFT in milliseconds
# - TokenGenerationBenchmark: tokens/second throughput
```

```bash
# ΣLANG benchmarks (20-30 minutes)
python runner.py \
  --suite microbenchmarks \
  --component sigmalang \
  --iterations 100 \
  --output results/day1/sigmalang_baseline.json

# Expected output:
# - CompressionRatioBenchmark: output_size / input_size
# - CompressionThroughputBenchmark: MB/s processed
```

```bash
# ΣVAULT benchmarks (20-30 minutes)
python runner.py \
  --suite microbenchmarks \
  --component sigmavault \
  --iterations 100 \
  --output results/day1/sigmavault_baseline.json

# Expected output:
# - RSUReadBenchmark: latency p50, p95, p99
# - RSUWriteBenchmark: latency p50, p95, p99
```

```bash
# Agents benchmarks (15-20 minutes)
python runner.py \
  --suite microbenchmarks \
  --component agents \
  --iterations 50 \
  --output results/day1/agents_baseline.json

# Expected output:
# - AgentTaskLatencyBenchmark: end-to-end task latency
```

**Day 1 End:** Store baseline results in `results/day1/` for comparison

---

### **Day 2: Profiling Macrobenchmarks & Trace Collection**

#### 1. Macrobenchmarks (Day 2 morning - 2 hours each)
```bash
# Full LLM Inference (Macrobenchmark 1) - 20-30 minutes
python runner.py \
  --suite macrobenchmarks \
  --benchmark FullLLMInference \
  --duration 300 \
  --profiler py-spy \
  --output results/day2/ryot_fullstack.json

# Analysis: py-spy will show which functions consume most time
# Expected: Identify attention, FFN, embedding bottlenecks
```

```bash
# Scaling Test (Macrobenchmark 2) - 20-30 minutes
python runner.py \
  --suite macrobenchmarks \
  --benchmark ScalingTest \
  --concurrent_requests 1,5,10,50,100 \
  --profiler py-spy \
  --output results/day2/scaling_results.json

# Analysis: How does performance degrade with concurrent load?
# Expected: Identify queueing, lock contention issues
```

```bash
# Endurance Test (Macrobenchmark 3) - 60 minutes
python runner.py \
  --suite macrobenchmarks \
  --benchmark EnduranceTest \
  --duration 3600 \
  --profile memory \
  --output results/day2/endurance_results.json

# Analysis: Memory leaks, resource degradation over time
# Expected: Identify memory growth patterns
```

```bash
# Collective Workflow (Macrobenchmark 4) - 30-40 minutes
python runner.py \
  --suite macrobenchmarks \
  --benchmark CollectiveWorkflow \
  --profiler py-spy \
  --output results/day2/collective_workflow.json

# Analysis: Latency through all 4 services
# Expected: Identify cross-service bottlenecks
```

**Day 2 End:** Complete macrobenchmarks and store profiling data

---

### **Day 3: Detailed Profiling & Bottleneck Analysis**

#### 1. Component-Specific Profiling (Day 3)

**Ryot LLM Profiling (Morning):**
```bash
# Detailed inference profiling - 45 minutes
python runner.py \
  --suite profiling \
  --benchmark DetailedRyotInferenceProfiling \
  --profiler cProfile \
  --output results/day3/ryot_cprofile.prof

# Generate flame graph
python -m cProfile -o results/day3/ryot.prof -m benchmarks.runner ...
python -c "import pstats; p = pstats.Stats('results/day3/ryot.prof'); p.print_stats()" > results/day3/ryot_hotspots.txt

# Expected: Top functions by cumulative time, self time
```

**ΣVAULT Storage Profiling (Midday):**
```bash
# Memory profiling for storage operations - 45 minutes
python runner.py \
  --suite profiling \
  --benchmark SigmaVaultMemoryProfiling \
  --profiler memory_profiler \
  --iterations 1000 \
  --output results/day3/sigmavault_memory.txt

# Generate memory growth analysis
python analyze_memory_profile.py results/day3/sigmavault_memory.txt > results/day3/sigmavault_memory_analysis.txt

# Expected: Line-by-line memory consumption, growth patterns
```

**Agent Communication Profiling (Afternoon):**
```bash
# Profile agent task execution - 45 minutes
python runner.py \
  --suite profiling \
  --benchmark AgentCommunicationProfiling \
  --profiler py-spy \
  --duration 600 \
  --output results/day3/agents_profile.json

# Expected: Which agent operations are slowest?
```

#### 2. Bottleneck Analysis (Day 3 evening)
```bash
# Run comprehensive bottleneck detection
python bottleneck_analyzer.py \
  --baseline results/day1/ \
  --profiling results/day3/ \
  --output results/bottleneck_report.json

# Generate prioritized optimization roadmap
python generate_optimization_roadmap.py \
  --bottleneck_report results/bottleneck_report.json \
  --output PHASE-18G-OPTIMIZATION-ROADMAP.md
```

**Day 3 End:** Bottleneck analysis complete, optimization priorities determined

---

### **Day 4-5: Data Analysis & Optimization Planning**

#### 1. Data Consolidation (Day 4 morning)
```bash
# Consolidate all profiling data
python consolidate_profiling_results.py \
  --day1 results/day1/ \
  --day2 results/day2/ \
  --day3 results/day3/ \
  --output results/CONSOLIDATED_PROFILING_REPORT.md
```

#### 2. Statistical Analysis & Confidence (Day 4)
```bash
# Determine statistical significance of bottlenecks
python statistical_significance.py \
  --baseline results/day1/ \
  --current results/day3/ \
  --confidence 0.95 \
  --output results/STATISTICAL_ANALYSIS.md

# Expected: Which bottlenecks are real vs noise?
```

#### 3. Optimization Roadmap (Day 4-5)
Create `PHASE-18G-OPTIMIZATION-ROADMAP.md` with:

**For Each Bottleneck:**
- Bottleneck name and location
- Current metric vs target
- Root cause analysis
- 3-5 optimization candidates
- ROI estimation (time savings vs effort)
- Implementation complexity (1-5)
- Risk assessment
- Recommended algorithm/technique

**Example Roadmap Entry:**
```markdown
## Bottleneck: Ryot TTFT - Attention Mechanism (35% of inference time)

**Current State:** 45ms TTFT vs target <100ms
**Root Cause:** Full attention computation O(n²) on full sequence
**Candidates:**
1. Flash Attention (40% faster, low risk)
2. Sparse Attention (50% faster, medium risk)
3. Speculative Decoding (60% faster, high complexity)

**Recommendation:** Flash Attention first (best ROI)
```

---

## 📈 Expected Results Format

### **Baseline Metrics (Day 1)**

```json
{
  "service": "ryot",
  "benchmark": "FirstTokenLatencyBenchmark",
  "timestamp": "2025-12-17T09:00:00Z",
  "metrics": {
    "ttft_mean_ms": 45.3,
    "ttft_p50_ms": 44.1,
    "ttft_p95_ms": 52.8,
    "ttft_p99_ms": 58.2,
    "tokens_per_second": 48.5,
    "peak_memory_mb": 2048,
    "gpu_utilization_percent": 85
  },
  "iterations": 100
}
```

### **Profiling Results (Day 3)**

```json
{
  "service": "ryot",
  "component": "attention_module",
  "profiler": "py-spy",
  "duration_seconds": 300,
  "top_hotspots": [
    {
      "function": "scaled_dot_product_attention",
      "self_time_ms": 1500,
      "total_time_ms": 3200,
      "call_count": 2000,
      "percentage_of_total": 35.5
    },
    {
      "function": "linear_layer",
      "self_time_ms": 1200,
      "total_time_ms": 2400,
      "call_count": 2000,
      "percentage_of_total": 26.7
    }
  ]
}
```

### **Bottleneck Analysis (Day 4)**

```markdown
## Top 5 Bottlenecks (Prioritized by ROI)

### 1. Ryot Attention Mechanism (35% of time, Flash Attention 40% improvement)
- **Effort:** 2 days
- **ROI:** 14ms faster TTFT (31% improvement)
- **Risk:** Low (well-tested)
- **Priority:** 🔴 CRITICAL

### 2. ΣLANG Dictionary Lookup (28% of compression time, 50% improvement via binary search)
- **Effort:** 1 day
- **ROI:** 50MB/s faster throughput
- **Risk:** Low
- **Priority:** 🔴 CRITICAL

### 3. ΣVAULT LRU Cache (22% of storage latency, skip-list 3× faster)
- **Effort:** 3 days
- **ROI:** 5ms faster read latency
- **Risk:** Medium (complex data structure)
- **Priority:** 🟡 HIGH

### 4. Agent Lock Contention (18% of agent time, lock-free queue)
- **Effort:** 4 days
- **ROI:** 8ms faster task latency
- **Risk:** High (concurrency complexity)
- **Priority:** 🟡 MEDIUM

### 5. Memory Fragmentation (15% extra memory, slab allocator)
- **Effort:** 5 days
- **ROI:** 40% memory reduction
- **Risk:** High
- **Priority:** 🟢 LOW
```

---

## 🛠️ Tools & Commands Reference

### **Profiling Tools**

```bash
# py-spy (sampling profiler, minimal overhead)
pip install py-spy
py-spy record -o profile.svg -- python your_script.py

# cProfile (deterministic profiler, precise timing)
python -m cProfile -o profile.prof your_script.py
python -m pstats profile.prof

# line_profiler (line-by-line profiling)
pip install line_profiler
kernprof -l -v your_script.py

# memory_profiler (memory usage tracking)
pip install memory_profiler
python -m memory_profiler your_script.py

# flamegraph (visualize profiling data)
pip install flamegraph
py-spy dump profile.prof > profile.txt
flamegraph.pl profile.txt > profile.svg
```

### **Benchmark Execution**

```bash
# Run specific benchmark
python benchmarks/runner.py \
  --suite microbenchmarks \
  --benchmark FirstTokenLatencyBenchmark \
  --iterations 100

# Run all microbenchmarks with profiling
python benchmarks/runner.py \
  --suite microbenchmarks \
  --profiler py-spy \
  --output results/full_profiling.json

# Run with different configurations
python benchmarks/runner.py \
  --config configs/low_latency.yaml \
  --output results/low_latency_benchmark.json
```

---

## 📋 Checklist for Success

**Day 1:**
- [ ] Environment setup complete
- [ ] Profiling tools installed and verified
- [ ] Microbenchmarks running for all 4 services
- [ ] Baseline metrics recorded

**Day 2:**
- [ ] All macrobenchmarks executed
- [ ] Profiling data collected
- [ ] Scaling tests show performance degradation curves
- [ ] Endurance test identifies memory leaks (if any)

**Day 3:**
- [ ] Component-specific profiling complete
- [ ] Flame graphs generated for each service
- [ ] Memory profiles analyzed
- [ ] Bottleneck analysis report generated

**Day 4:**
- [ ] All data consolidated into single report
- [ ] Statistical significance calculated
- [ ] Top bottlenecks ranked by ROI
- [ ] Optimization roadmap ready for Phase 18G

**Day 5:**
- [ ] Executive summary created
- [ ] Presentations prepared for each service optimization
- [ ] Phase 18G implementation plan finalized
- [ ] Hand-off to @VELOCITY for optimization execution

---

## 📊 Deliverables

**By End of Day 5:**

1. ✅ **CONSOLIDATED_PROFILING_REPORT.md** (500+ lines)
   - Executive summary
   - Baseline metrics for all benchmarks
   - Profiling hotspots
   - Bottleneck analysis
   - ROI scoring for each optimization

2. ✅ **PHASE-18G-OPTIMIZATION-ROADMAP.md** (1,000+ lines)
   - Top 20 bottlenecks ranked by ROI
   - Implementation complexity and risk
   - Estimated time savings per optimization
   - Recommended execution order
   - Sub-linear algorithm applications

3. ✅ **Profiling Data** (all raw results)
   - Day 1: Baseline metrics (JSON)
   - Day 2: Macrobenchmark results (JSON)
   - Day 3: Component profiles (prof, txt, JSON)
   - Day 4: Analysis and consolidated report (MD, JSON)

4. ✅ **Flame Graphs** (visualization)
   - Ryot inference flame graph (SVG)
   - ΣLANG compression flame graph (SVG)
   - ΣVAULT storage flame graph (SVG)
   - Agents coordination flame graph (SVG)

5. ✅ **Excel Dashboard** (Optional)
   - Baseline vs targets comparison
   - Bottleneck ranking table
   - ROI comparison matrix
   - Timeline estimates

---

## 🚀 Next Phase (18G) Dependency

**Phase 18G will use:**
- Bottleneck rankings from Day 4
- ROI scores to prioritize optimizations
- Implementation complexity estimates
- Risk assessments for safe execution

**Phase 18G Timeline:**
- Week 1: Implement top 5 optimizations
- Week 2: Execute Phase 18G-2 through 18G-5 in parallel
- Week 3: Validate improvements, prepare Phase 18H

---

## 💡 Tips & Best Practices

1. **Isolation:** Run each benchmark in isolation (no concurrent services)
2. **Warmup:** Include warmup iterations to eliminate JIT compilation noise
3. **Sampling:** Use sampling profilers first (py-spy), then deterministic (cProfile)
4. **Verification:** Validate profiling results with multiple tools
5. **Baselines:** Document all baseline metrics for future comparisons
6. **Statistical:** Run 100+ iterations for microbenchmarks, 5+ for macrobenchmarks

---

**Status: 🟢 READY TO EXECUTE**

Start Day 1 profiling immediately. Report bottleneck findings by EOD Day 4.

