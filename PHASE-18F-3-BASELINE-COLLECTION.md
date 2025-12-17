# Phase 18F-3: Profiling Baseline Collection Guide

**Status:** Starting Day 1  
**Date:** December 17, 2025  
**Duration:** 5 days  
**Goal:** Establish baseline metrics and identify top 20 bottlenecks

---

## 🚀 QUICK START (5 MINUTES)

### Prerequisites Check

```bash
# Verify Python 3.9+
python --version

# Verify profiling tools
pip install py-spy memory-profiler line_profiler scalene

# Verify directory structure
mkdir -p results/phase_18f/{raw_profiles,flame_graphs,metrics_json,daily_reports}
```

### Run Day 1 Baseline

```bash
cd c:\Users\sgbil\NEURECTOMY
python benchmarks/runner_18f.py 1
```

**Expected Output:**
```
============================================================
# PHASE 18F-3 PROFILING - DAY 1
# 2025-12-17T...

Running: FirstTokenLatencyBenchmark
...
✓ TTFT Mean: 45.30ms, p99: 58.20ms
✓ Throughput: 52.45 tokens/sec

Running: TokenGenerationBenchmark
...
✓ Daily report saved: results/phase_18f/daily_reports/day_1_report_....json
  Total benchmarks: 2
  Successful: 2
  Failed: 0
```

---

## 📅 5-DAY EXECUTION SCHEDULE

### Day 1: Ryot LLM Baselines (Today - Dec 17)

**Commands:**
```bash
# Run Day 1 benchmarks (30-45 minutes)
python benchmarks/runner_18f.py 1

# Check results
ls -la results/phase_18f/daily_reports/
cat results/phase_18f/daily_reports/day_1_report_*.json | jq '.results[] | {name: .benchmark_name, ttft: .mean_ms, p99: .p99_ms}'
```

**Benchmarks Running:**
1. **FirstTokenLatencyBenchmark** - Time to first token
   - Target: <100ms
   - Expected baseline: ~45ms (exceeds target ✓)

2. **TokenGenerationBenchmark** - Throughput
   - Target: >50 tokens/sec
   - Expected baseline: ~48-52 tokens/sec (meets target ✓)

**Expected Output:**
```json
{
  "name": "FirstTokenLatencyBenchmark",
  "ttft": 45.3,
  "p99": 58.2,
  "throughput": 52.45,
  "cpu": 85.0
}
```

**Deliverable:** `day_1_report_*.json` with 2 benchmarks

---

### Day 2: ΣLANG, ΣVAULT, Agents Baselines (Dec 18)

**Commands:**
```bash
# Run Day 2 benchmarks (1.5-2 hours)
python benchmarks/runner_18f.py 2

# View results
python -c "
import json
with open('results/phase_18f/daily_reports/day_2_report_*.json'.glob_latest()) as f:
    report = json.load(f)
    for r in report['results']:
        print(f\"{r['benchmark_name']:40} | Throughput: {r.get('throughput', 'N/A'):8} {r.get('throughput_unit', '')}\")
"
```

**Benchmarks Running:**
1. **CompressionRatioBenchmark** (ΣLANG)
   - Expected baseline: ~3.5:1 (exceeds 3:1 target ✓)

2. **CompressionThroughputBenchmark** (ΣLANG)
   - Expected baseline: ~125 MB/s (exceeds 100 MB/s target ✓)

3. **RSUReadBenchmark** (ΣVAULT)
   - Expected baseline: p99=9.2ms (meets <10ms target ✓)

4. **RSUWriteBenchmark** (ΣVAULT)
   - Expected baseline: p99=18.5ms (meets <20ms target ✓)

5. **AgentTaskLatencyBenchmark** (Agents)
   - Expected baseline: p99=45ms (meets <50ms target ✓)

**Deliverable:** `day_2_report_*.json` with 5 benchmarks

---

### Day 3: Macrobenchmarks & Profiling (Dec 19)

**Commands:**
```bash
# Run Day 3 macrobenchmarks (1-2 hours each)
python benchmarks/runner_18f.py 3

# Monitor in separate terminal
watch -n 5 'ps aux | grep python | grep runner'
```

**Benchmarks Running:**
1. **FullLLMInference** (10 min) - Complete inference stack
2. **ScalingTest** (15 min) - Concurrent request handling
3. **EnduranceTest** (60 min) - Extended stress test
4. **CollectiveWorkflow** (10 min) - All 4 services together

**Expected Output:**
```
Completed FullLLMInference: avg_latency=52.3ms, p99=78.5ms
Completed ScalingTest: throughput degradation at 50 concurrent: -12%
Completed EnduranceTest: memory growth: 2048MB → 2156MB (5.3%)
Completed CollectiveWorkflow: end-to-end latency: 156ms
```

**Deliverable:** `day_3_report_*.json` with profiling data

---

### Day 4: Detailed Analysis & Bottleneck Identification (Dec 20)

**Commands:**
```bash
# Run profiling suite
python benchmarks/runner_18f.py 4

# Analyze bottlenecks
python benchmarks/bottleneck_analyzer.py --day 1-4 --output bottleneck_analysis.json

# Generate summary
cat bottleneck_analysis.json | jq '.identified_bottlenecks[] | {component: .component, severity: .severity, roi: .estimated_roi}'
```

**Expected Bottleneck Analysis:**
```
TOP 5 BOTTLENECKS (by ROI):
┌─────────────────────────────────────────────────────────────┐
│ 1. Ryot Attention Mechanism (35% time)                      │
│    → Flash Attention: 40% faster, 2 days, LOW risk         │
│    → ROI: 14ms improvement                                  │
│                                                              │
│ 2. ΣLANG Dictionary Lookup (28% time)                      │
│    → Binary Search: 50% faster, 1 day, LOW risk            │
│    → ROI: 50MB/s improvement                               │
│                                                              │
│ 3. ΣVAULT LRU Cache (22% time)                             │
│    → Skip-List: 3× faster, 3 days, MEDIUM risk            │
│    → ROI: 5ms improvement                                  │
│                                                              │
│ 4. Agent Lock Contention (18% time)                        │
│    → Lock-Free Queue: 2× faster, 4 days, HIGH risk         │
│    → ROI: 8ms improvement                                  │
│                                                              │
│ 5. Memory Fragmentation (15% memory)                        │
│    → Slab Allocator: 40% less memory, 5 days, HIGH risk    │
│    → ROI: 40% memory reduction                             │
└─────────────────────────────────────────────────────────────┘
```

**Deliverable:** `bottleneck_analysis.json` with ROI scores

---

### Day 5: Optimization Roadmap Creation (Dec 21)

**Commands:**
```bash
# Generate roadmap
python benchmarks/generate_optimization_roadmap.py \
  --bottleneck_analysis bottleneck_analysis.json \
  --output PHASE-18G-OPTIMIZATION-ROADMAP.md

# Review roadmap
cat PHASE-18G-OPTIMIZATION-ROADMAP.md
```

**Expected Roadmap Structure:**
```markdown
# Phase 18G Optimization Roadmap

## Week 1 (High ROI, Low Risk)
- [ ] Flash Attention (Ryot) - 2 days
- [ ] Binary Search Dict (ΣLANG) - 1 day

## Week 2 (Medium ROI/Risk)
- [ ] Skip-List LRU Cache (ΣVAULT) - 3 days
- [ ] Lock-Free Queue (Agents) - 4 days

## Week 3 (Validation & Cleanup)
- [ ] Performance validation - 2 days
- [ ] Regression testing - 1 day
```

**Deliverable:** `PHASE-18G-OPTIMIZATION-ROADMAP.md` (ready for Phase 18G)

---

## 📊 BASELINE METRICS REFERENCE

### Expected Baseline Values (Pre-Optimization)

| Service | Metric | Baseline | Target | Status |
|---------|--------|----------|--------|--------|
| Ryot | TTFT | 45ms | <100ms | ✅ EXCEEDS |
| Ryot | Throughput | 52 tok/sec | >50 | ✅ EXCEEDS |
| ΣLANG | Ratio | 3.5:1 | >3:1 | ✅ EXCEEDS |
| ΣLANG | Speed | 125 MB/s | >100 | ✅ EXCEEDS |
| ΣVAULT | Read p99 | 9.2ms | <10ms | ✅ MEETS |
| ΣVAULT | Write p99 | 18.5ms | <20ms | ✅ MEETS |
| Agents | Task p99 | 45ms | <50ms | ✅ MEETS |

---

## 🔍 PROFILING DATA INTERPRETATION

### Reading Raw Profiling Output

```bash
# View py-spy sampling results
cat results/phase_18f/raw_profiles/ryot_sampling.prof | head -20

# Generate flame graph
python -m flamegraph results/phase_18f/raw_profiles/ryot_sampling.prof \
  > results/phase_18f/flame_graphs/ryot_flames.svg
```

### Bottleneck Severity Classification

**🔴 CRITICAL** (>25% of total time, ROI >3× effort)
- Implement immediately in Week 1

**🟡 HIGH** (15-25% of total time, ROI >2× effort)
- Implement in Week 2

**🟢 MEDIUM** (5-15% of total time, ROI >1× effort)
- Implement in Week 3 if time permits

**⚪ LOW** (<5% of total time, ROI <1× effort)
- Document for future optimization

---

## 📋 DATA COLLECTION CHECKLIST

**Day 1:**
- [ ] Run Ryot benchmarks
- [ ] Verify metrics saved
- [ ] Check for errors
- [ ] Document baseline

**Day 2:**
- [ ] Run ΣLANG/ΣVAULT/Agents benchmarks
- [ ] Compare against targets
- [ ] Identify quick wins
- [ ] Document findings

**Day 3:**
- [ ] Run macrobenchmarks (long duration)
- [ ] Monitor resource usage
- [ ] Capture scaling characteristics
- [ ] Analyze degradation

**Day 4:**
- [ ] Run detailed profiling
- [ ] Generate bottleneck report
- [ ] Calculate ROI scores
- [ ] Rank optimizations

**Day 5:**
- [ ] Consolidate all data
- [ ] Create optimization roadmap
- [ ] Finalize recommendations
- [ ] Prepare for Phase 18G

---

## 🛠️ TROUBLESHOOTING

### Issue: Benchmarks fail to run

**Solution:**
```bash
# Check environment
pip list | grep -E "py-spy|memory-profiler|line_profiler"

# Reinstall if needed
pip install --upgrade py-spy memory-profiler line_profiler scalene

# Test profiling
python -m cProfile -o test.prof -c "import sys; sys.exit(0)"
```

### Issue: Memory usage grows during endurance test

**Solution:**
```bash
# Check for memory leaks
python -m memory_profiler benchmarks/runner_18f.py 3

# Review memory profile output
```

### Issue: Results directory not created

**Solution:**
```bash
mkdir -p results/phase_18f/{raw_profiles,flame_graphs,metrics_json,daily_reports}
```

---

## 🎯 PHASE 18F-3 COMPLETION CRITERIA

✅ **When ALL of these are true:**

1. ✅ Day 1 Ryot benchmarks completed and saved
2. ✅ Day 2 multi-component benchmarks completed
3. ✅ Day 3 macrobenchmarks data collected
4. ✅ Day 4 profiling analysis complete
5. ✅ Day 5 optimization roadmap created
6. ✅ All results in `results/phase_18f/` directory
7. ✅ Bottleneck ranking with ROI scores
8. ✅ Top 10 optimizations identified
9. ✅ Implementation complexity assessed
10. ✅ Risk ratings assigned

**Phase 18F-3 Status:** 🔴 **IN PROGRESS - START TODAY**

---

## 📞 NEXT STEPS

When Day 5 is complete:

1. **Review Roadmap** - Ensure top optimizations prioritized by ROI
2. **Delegate Phase 18G** - Hand off to @VELOCITY for implementation
3. **Update Todo** - Mark 18F-3 complete, start 18G
4. **Commit Results** - Push profiling data to repository

---

**Ready to begin Day 1? Run: `python benchmarks/runner_18f.py 1`**

