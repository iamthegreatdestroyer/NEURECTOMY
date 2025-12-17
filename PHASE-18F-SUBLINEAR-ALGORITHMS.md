# Phase 18F: Sub-Linear Algorithms Reference Guide

**@VELOCITY Mode - Streaming, Probabilistic, and Approximate Algorithms**

---

## Quick Reference: Algorithm Selection Matrix

| Problem            | Algorithm        | Space      | Time          | Error           | Use Case                          |
| ------------------ | ---------------- | ---------- | ------------- | --------------- | --------------------------------- |
| **Distinct Count** | HyperLogLog      | O(log n)   | O(1)          | ~2%             | Cardinality estimation in streams |
| **Frequency**      | Count-Min Sketch | O(log 1/δ) | O(1)          | Conservative    | Heavy hitter detection            |
| **Set Membership** | Bloom Filter     | O(n) bits  | O(k)          | FP only         | "Does item exist?" queries        |
| **Similarity**     | MinHash + LSH    | O(n)       | O(1) expected | Approximate     | Finding similar items             |
| **Range Queries**  | t-digest         | O(δ)       | O(log δ)      | ε-bounded       | Quantile approximation            |
| **Top-K Items**    | Misra-Gries      | O(1/ε)     | O(1)          | Top-k guarantee | Most frequent items               |

---

## 1. HyperLogLog - Cardinality Estimation

### Problem

Counting unique items in a stream without storing all items.

**Naive approach:** Store all in set → O(n) space

**HyperLogLog approach:** Use probabilistic data structure → O(log n) space with ~2% error

### Algorithm Overview

```
1. Hash each item to uniform random binary
2. Count leading zeros (position of first 1)
3. Keep max position seen per register
4. Final count = magic_constant * 2^(avg(registers))
```

### Implementation

```python
from hyperloglog import HyperLogLog

# Create HyperLogLog with precision 14
# precision = log2(number of registers)
# 2^14 = 16K registers
hll = HyperLogLog(precision=14)  # Uses ~2KB memory

# Add items
for item in stream:
    hll.add(item)

# Query cardinality (estimated)
count = hll.cardinality()  # Estimated count ±2%

# Space used: ~16K * 5 bits ≈ 10KB (vs millions for set)
```

### Neurectomy Application

```python
# Phase 18F: Track unique agents in collective
from benchmarks.phase_18f_profiling_utils import BottleneckAnalyzer

agents_seen = HyperLogLog(precision=12)  # 4K registers

for task in task_stream:
    agent_id = task.assigned_agent
    agents_seen.add(agent_id)

    # Report statistics
    unique_agents = agents_seen.cardinality()
    print(f"Estimated unique agents: {unique_agents}")
```

### Trade-offs

| Aspect       | Cost                                           |
| ------------ | ---------------------------------------------- |
| **Space**    | ~0.5-1 bytes per unique item (vs full storage) |
| **Time**     | O(1) per add, O(1) per query                   |
| **Error**    | ~2% standard deviation (configurable)          |
| **Accuracy** | No false negatives (conservative estimate)     |

---

## 2. Count-Min Sketch - Frequency Estimation

### Problem

Tracking frequencies of items in stream without storing all counts.

**Naive approach:** Hash table of all frequencies → O(n) space

**Count-Min Sketch approach:** Multiple hash functions → O(log 1/δ) space, conservative overestimate

### Algorithm Overview

```
1. Create 2D array: depth × width (e.g., 5 × 10000)
2. For each item:
   - Hash with each of depth hash functions
   - Increment counter at that position
3. Query item:
   - Hash with each function
   - Return minimum value
```

### Implementation

```python
from countminsketch import CountMinSketch

# Create sketch: width=10000, depth=5
# Space: 50K counters ≈ 200KB (vs millions for hash table)
cms = CountMinSketch(width=10000, depth=5)

# Add items
for item in stream:
    cms.add(item)

# Query frequency (conservative estimate)
freq = cms.query("frequent_item")
# Returns actual frequency or overestimate (never underestimate)
```

### Neurectomy Application

```python
# Phase 18F: Track most frequent operations in agent logs
from countminsketch import CountMinSketch

operation_freq = CountMinSketch(width=10000, depth=5)

for log_entry in agent_logs:
    operation = log_entry.operation_type
    operation_freq.add(operation)

    # Find heavy hitters
    top_ops = [
        ("compress", operation_freq.query("compress")),
        ("decompress", operation_freq.query("decompress")),
        ("store", operation_freq.query("store")),
        ("retrieve", operation_freq.query("retrieve")),
    ]
    top_ops.sort(key=lambda x: x[1], reverse=True)
    print(f"Top operations: {top_ops[:3]}")
```

### Trade-offs

| Aspect       | Cost                                         |
| ------------ | -------------------------------------------- |
| **Space**    | O(log 1/δ) ≈ 5-10× smaller than exact counts |
| **Time**     | O(log 1/δ) per add, O(log 1/δ) per query     |
| **Error**    | Conservative (never underestimates)          |
| **Accuracy** | ε-approximate with probability 1-δ           |

---

## 3. Bloom Filter - Set Membership Testing

### Problem

Quick "does this item exist?" queries without storing full set.

**Naive approach:** Set lookup → O(log n) time or O(1) hash average

**Bloom Filter approach:** Bit vector → O(k) time (k hash functions), O(n) bits, no false negatives

### Algorithm Overview

```
1. Create bit vector of size m
2. Use k independent hash functions
3. Add item: set bits at positions hash1(item), hash2(item), ...
4. Query item: check if ALL bits at those positions are set
   - If any bit is 0: item definitely NOT in set
   - If all bits are 1: item probably in set (~1% false positive)
```

### Implementation

```python
from bloom_filter2 import BloomFilter

# Create Bloom filter
# max_elements=1,000,000, false_positive_rate=0.01 (1%)
bf = BloomFilter(max_elements=1000000, error_rate=0.01)

# Add items
for item in known_items:
    bf.add(item)

# Query (very fast)
if "query_item" in bf:  # O(k) operations
    # Might be in set (false positive possible)
    # Verify with actual set/database
    if actual_verification("query_item"):
        process("query_item")
```

### Neurectomy Application

```python
# Phase 18F: Fast negative lookups for storage
from bloom_filter2 import BloomFilter

# Track RSUs that definitely don't exist
non_existent_rsus = BloomFilter(max_elements=1000000, error_rate=0.01)

def get_rsu_fast(rsu_id):
    # Quick negative check (1% false positive)
    if rsu_id in non_existent_rsus:
        return None  # Definitely doesn't exist

    # Need to check actual storage
    rsu = storage.retrieve(rsu_id)

    if rsu is None:
        non_existent_rsus.add(rsu_id)  # Remember for next time

    return rsu

# Result: Cache misses detected 99% of time without database query
```

### Trade-offs

| Aspect       | Cost                                         |
| ------------ | -------------------------------------------- |
| **Space**    | O(n) bits ≈ 1 bit per element                |
| **Time**     | O(k) ≈ 10-20 nanoseconds (k hash functions)  |
| **Error**    | Configurable false positive (typically 1-5%) |
| **Accuracy** | Zero false negatives (conservative)          |

---

## 4. MinHash + LSH - Similarity Search

### Problem

Finding similar items in large dataset without pairwise comparison.

**Naive approach:** Compare each pair → O(n²m) time

**MinHash + LSH approach:** Approximate NN search → O(n) expected

### Algorithm Overview

#### MinHash

```
1. For each item's set of elements:
   - Hash all elements with multiple (e.g., 128) hash functions
   - Keep minimum hash value from each function
   - Result: signature of 128 values (compact representation)

2. Jaccard similarity ≈ fraction of matching minimums
```

#### LSH (Locality-Sensitive Hashing)

```
1. Group signatures into bands (e.g., 16 values per band)
2. Hash each band to bucket
3. Items in same bucket are similar
```

### Implementation

```python
from datasketch import MinHash, MinHashLSH

# Create MinHash LSH
lsh = MinHashLSH(num_perm=128, threshold=0.5)

# Add items
for i, item in enumerate(items):
    # Create MinHash signature
    m = MinHash(num_perm=128)
    for element in item.set_of_elements:
        m.update(element.encode('utf8'))

    # Add to LSH
    lsh.insert(f"item_{i}", m)

# Query similar items (fast!)
query_m = MinHash(num_perm=128)
for element in query_item.set_of_elements:
    query_m.update(element.encode('utf8'))

# Find all similar items (O(n) expected instead of O(n²))
similar_items = lsh.query(query_m)
```

### Neurectomy Application

```python
# Phase 18F: Find similar agent tasks for batching
from datasketch import MinHash, MinHashLSH

task_lsh = MinHashLSH(num_perm=128, threshold=0.7)  # 70% similarity

def add_task(task_id, task_features):
    m = MinHash(num_perm=128)
    for feature in task_features:
        m.update(feature.encode('utf8'))

    task_lsh.insert(task_id, m)

def find_similar_tasks(task_id, task_features, num_to_find=10):
    m = MinHash(num_perm=128)
    for feature in task_features:
        m.update(feature.encode('utf8'))

    similar = task_lsh.query(m)  # O(n) expected
    return similar[:num_to_find]

# Batch similar tasks for efficiency
# Find tasks similar to current one, batch execute together
# Result: Better resource utilization, fewer context switches
```

### Trade-offs

| Aspect       | Cost                                         |
| ------------ | -------------------------------------------- |
| **Space**    | O(n) for signatures + buckets                |
| **Time**     | O(n) expected for query (vs O(n²) naive)     |
| **Error**    | Approximate results (configurable threshold) |
| **Accuracy** | Can tune precision/recall trade-off          |

---

## 5. t-Digest - Quantile Approximation

### Problem

Computing quantiles (percentiles) of streaming data without storing all values.

**Naive approach:** Store all values, sort → O(n) space, O(n log n) time

**t-Digest approach:** Approximate quantiles → O(δ) space, O(log δ) time

### Algorithm Overview

```
1. Maintain clusters of values (centroids)
2. Merge nearby clusters when size limit reached
3. Query quantile: interpolate from clusters
```

### Implementation

```python
from tdigest import TDigest

# Create t-digest
td = TDigest()

# Add values (can be streaming)
for value in value_stream:
    td.add(value)

# Query quantiles
p50 = td.percentile(50)  # Median
p99 = td.percentile(99)  # 99th percentile
p999 = td.percentile(99.9)  # 99.9th percentile

# CDF queries
prob = td.cdf(x)  # Probability of value <= x
```

### Neurectomy Application

```python
# Phase 18F: Track inference latency percentiles in real-time
from tdigest import TDigest

inference_latencies = TDigest()

for result in inference_results_stream:
    inference_latencies.add(result.latency_ms)

    # Real-time SLO monitoring
    if inference_latencies.percentile(99) > 150:  # Alert if p99 > 150ms
        alert("High inference latency detected")

    # Reporting
    stats = {
        "p50": inference_latencies.percentile(50),
        "p95": inference_latencies.percentile(95),
        "p99": inference_latencies.percentile(99),
        "p999": inference_latencies.percentile(99.9),
    }
```

### Trade-offs

| Aspect       | Cost                                              |
| ------------ | ------------------------------------------------- |
| **Space**    | O(δ) ≈ 100-200 clusters vs thousands of values    |
| **Time**     | O(log δ) ≈ 7-8 operations                         |
| **Error**    | ε-bounded (configurable accuracy)                 |
| **Accuracy** | Excellent for middle quantiles, good for extremes |

---

## 6. Misra-Gries - Top-K Frequent Items

### Problem

Finding k most frequent items in stream without storing all counts.

**Naive approach:** Hash table of all counts → O(n) space

**Misra-Gries approach:** Exact top-k guarantee → O(1/ε) space

### Algorithm Overview

```
1. Maintain k counters
2. For each item:
   - If item has counter, increment it
   - Else if <k counters, create new counter
   - Else decrement all counters, remove zeros
3. Guarantees: All items with frequency ≥ n/(k+1) are in top-k
```

### Implementation

```python
from misragries import MisraGries

# Track top 10 items with O(1) space
mg = MisraGries(k=10)

# Add items
for item in stream:
    mg.add(item)

# Query top items
top_items = mg.get_top_k()
# Returns: [(item, frequency), ...]
```

### Neurectomy Application

```python
# Phase 18F: Track top bottleneck functions
from misragries import MisraGries

bottleneck_tracker = MisraGries(k=10)

for profile_entry in profiling_data:
    function_name = profile_entry.function
    bottleneck_tracker.add(function_name)

# Get top 10 bottlenecks
top_bottlenecks = bottleneck_tracker.get_top_k()
print("Top bottlenecks:")
for func, count in top_bottlenecks:
    print(f"  {func}: {count} samples")
```

### Trade-offs

| Aspect       | Cost                               |
| ------------ | ---------------------------------- |
| **Space**    | O(1/ε) ≈ k items stored            |
| **Time**     | O(1) amortized per add             |
| **Error**    | Exact guarantee (no approximation) |
| **Accuracy** | 100% for top-k items               |

---

## 7. Probabilistic Counting - Cardinality Estimation Variants

### Alternatives to HyperLogLog

#### Linear Counting (Small cardinality)

```python
from probabilistic_counting import LinearCounter

lc = LinearCounter(size=10000)
for item in stream:
    lc.add(item)

cardinality = lc.cardinality()
# Better accuracy for small cardinalities (<100K)
```

#### LogLog Counting (Medium cardinality)

```python
from probabilistic_counting import LogLogCounter

llc = LogLogCounter(precision=10)  # 1024 registers
for item in stream:
    llc.add(item)

cardinality = llc.cardinality()
# Balanced between space and accuracy
```

---

## 8. Real-World Performance Measurements

### Memory Reduction

```
Algorithm            | Space Reduction | Notes
---------------------|-----------------|---------------------------
HyperLogLog          | 1000-10000×    | Cardinality estimation
Count-Min Sketch     | 100-1000×      | Frequency estimation
Bloom Filter         | 1000-10000×    | Set membership (bits)
t-Digest             | 10-100×        | Quantile approximation
Misra-Gries          | 1000-100000×   | Top-k items
```

### Speed Improvements

```
Operation Type       | Speedup | Example
---------------------|---------|---------------------------
"Not found" queries  | 100×    | Bloom filter lookups
Similarity search    | 100-1000× | MinHash+LSH vs pairwise
Quantile updates     | 1000×   | t-Digest vs full sort
Frequency queries    | 100×    | Count-Min vs hash lookup
```

---

## Integration Guide

### Add Sub-Linear Algorithms to Neurectomy

```python
# 1. Create utilities module
# neurectomy/core/sublinear_algorithms.py

from hyperloglog import HyperLogLog
from countminsketch import CountMinSketch
from bloom_filter2 import BloomFilter
from datasketch import MinHash, MinHashLSH
from tdigest import TDigest

class SubLinearAlgorithms:
    @staticmethod
    def cardinality_estimator(precision=14):
        """Create HyperLogLog cardinality estimator."""
        return HyperLogLog(precision=precision)

    @staticmethod
    def frequency_tracker(width=10000, depth=5):
        """Create Count-Min Sketch for frequency estimation."""
        return CountMinSketch(width=width, depth=depth)

    @staticmethod
    def membership_tester(max_elements=1000000, error_rate=0.01):
        """Create Bloom Filter for set membership."""
        return BloomFilter(max_elements=max_elements, error_rate=error_rate)

    # ... more factories
```

### Use in Components

```python
# neurectomy/sigma_vault/storage.py
from neurectomy.core.sublinear_algorithms import SubLinearAlgorithms

class OptimizedStorage:
    def __init__(self):
        # Track non-existent RSUs with Bloom filter
        self.non_existent = SubLinearAlgorithms.membership_tester()

        # Track access frequency with Count-Min
        self.access_freq = SubLinearAlgorithms.frequency_tracker()

        # Monitor latency percentiles with t-Digest
        self.latency_stats = TDigest()

    def get(self, rsu_id):
        # Fast negative check
        if rsu_id in self.non_existent:
            return None

        # Normal retrieval
        rsu = self._retrieve(rsu_id)

        # Track for future fast rejection
        if rsu is None:
            self.non_existent.add(rsu_id)

        # Track access
        self.access_freq.add(rsu_id)
        return rsu
```

---

## Performance Checklist

- [ ] Identified all streaming operations (candidates for HyperLogLog)
- [ ] Identified frequency tracking needs (Count-Min Sketch)
- [ ] Identified "not found" queries (Bloom Filter)
- [ ] Identified similarity search needs (MinHash+LSH)
- [ ] Identified percentile tracking needs (t-Digest)
- [ ] Identified top-k patterns (Misra-Gries)
- [ ] Implemented sub-linear data structures
- [ ] Measured memory reduction
- [ ] Measured speed improvements
- [ ] Added to benchmark suite
- [ ] Documented in code comments

---

## References

- **HyperLogLog:** "HyperLogLog: The analysis of a near-optimal cardinality estimation algorithm" - Flajolet et al.
- **Count-Min Sketch:** "An Improved Data Stream Summary: The Count-Min Sketch and its Applications" - Cormode & Muthukrishnan
- **Bloom Filters:** "Space/time trade-offs in hash coding with allowable errors" - Bloom
- **MinHash:** "Min-wise independent permutations" - Broder et al.
- **t-Digest:** "Computing Accurate Quantiles Using t-Digests" - Dunning & Ertl
- **Misra-Gries:** "Finding repeated elements" - Misra & Gries
