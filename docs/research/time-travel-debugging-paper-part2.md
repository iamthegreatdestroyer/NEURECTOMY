# Time-Travel Debugging for Autonomous Agents: Deterministic Replay with Counterfactual Branching

## PLDI 2026 / OOPSLA 2026 - Submission Draft

**Part 2 of 3: Methodology**

---

## 6. TTD Framework Architecture

### 6.1 System Overview

The Time-Travel Debugging (TTD) framework consists of four integrated components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TTD Framework Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌─────────────────┐                      ┌─────────────────┐             │
│    │   Agent Code    │                      │   Environment   │             │
│    │   (Policy, RL)  │                      │   (Simulation)  │             │
│    └────────┬────────┘                      └────────┬────────┘             │
│             │                                        │                       │
│             ▼                                        ▼                       │
│    ┌──────────────────────────────────────────────────────────┐             │
│    │              Instrumentation Layer                        │             │
│    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │             │
│    │  │   RNG    │  │  Neural  │  │  System  │  │   I/O    │ │             │
│    │  │  Capture │  │  Net I/O │  │  Calls   │  │  Events  │ │             │
│    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │             │
│    └──────────────────────────┬───────────────────────────────┘             │
│                               │                                              │
│                               ▼                                              │
│    ┌──────────────────────────────────────────────────────────┐             │
│    │                  Recording Engine                         │             │
│    │  ┌──────────────────┐  ┌──────────────────┐              │             │
│    │  │  Event Stream    │  │  Checkpoint      │              │             │
│    │  │  (Non-Det Oracle)│  │  Manager         │              │             │
│    │  └──────────────────┘  └──────────────────┘              │             │
│    └──────────────────────────┬───────────────────────────────┘             │
│                               │                                              │
│                               ▼                                              │
│    ┌──────────────────────────────────────────────────────────┐             │
│    │                   Trace Storage                           │             │
│    │  ┌────────────────────────────────────────────────────┐  │             │
│    │  │  Oracle Log  │  Checkpoints  │  Metadata Index     │  │             │
│    │  └────────────────────────────────────────────────────┘  │             │
│    └──────────────────────────┬───────────────────────────────┘             │
│                               │                                              │
│                               ▼                                              │
│    ┌──────────────────────────────────────────────────────────┐             │
│    │                  Debug Interface                          │             │
│    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │             │
│    │  │ Navigate │  │ Temporal │  │ Counter- │  │ Compare  │ │             │
│    │  │ (← →)    │  │ Breakpts │  │ factual  │  │ Branches │ │             │
│    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │             │
│    └──────────────────────────────────────────────────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Component Responsibilities

**Instrumentation Layer:**

- Intercepts all non-deterministic operations
- Minimal overhead through selective instrumentation
- Language-agnostic through ABI-level interception

**Recording Engine:**

- Serializes events to compact binary format
- Manages checkpoint scheduling
- Handles buffer management and flushing

**Trace Storage:**

- Indexed storage for fast random access
- Compression for space efficiency
- Supports distributed storage for multi-agent systems

**Debug Interface:**

- User-facing API for navigation and queries
- Visual timeline for temporal exploration
- Diff viewer for counterfactual comparison

---

## 7. Deterministic Replay Engine

### 7.1 Recording Architecture

**Algorithm 1: Recording Loop**

```
procedure RECORD(agent, environment, max_steps):
    oracle_log = []
    checkpoints = []

    state = (agent.init_state(), environment.init_state())
    checkpoint(state, 0, checkpoints)

    for t = 0 to max_steps:
        // Capture all non-determinism
        rng_state = capture_rng_state()

        // Execute one step
        obs = environment.observe(state.world)
        action, rng_consumed = agent.act(state.agent, obs)
        next_world, reward = environment.step(state.world, action)
        next_agent = agent.update(state.agent, obs, action, reward)

        // Log non-determinism oracle
        oracle_log.append({
            t: t,
            rng_state: rng_state,
            rng_consumed: rng_consumed,
            system_events: capture_system_events()
        })

        // Adaptive checkpointing
        if should_checkpoint(t, state, oracle_log):
            checkpoint(state, t, checkpoints)

        state = (next_agent, next_world)

    return Trace(oracle_log, checkpoints)
```

### 7.2 Non-Determinism Capture

**7.2.1 Random Number Generators**

Capture RNG state before each operation:

```python
class DeterministicRNG:
    def __init__(self, seed):
        self.state = initialize_state(seed)
        self.log = []

    def random(self):
        self.log.append(('random', copy(self.state)))
        value = generate(self.state)
        self.state = advance(self.state)
        return value

    def replay_random(self, logged_state):
        self.state = logged_state
        return generate(self.state)
```

For numpy/PyTorch, we intercept the global RNG:

```python
# Recording mode
original_random = np.random.random
def recording_random(*args, **kwargs):
    state = np.random.get_state()
    oracle.log('numpy_rng', state)
    return original_random(*args, **kwargs)
np.random.random = recording_random

# Replay mode
def replay_random(*args, **kwargs):
    state = oracle.next('numpy_rng')
    np.random.set_state(state)
    return original_random(*args, **kwargs)
```

**7.2.2 Neural Network Determinism**

GPU operations require special handling:

```python
def ensure_deterministic_nn():
    # PyTorch deterministic mode
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set all seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Environment variable for CUDA
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

For operations with no deterministic alternative, we record outputs:

```python
class RecordedConv2d(nn.Conv2d):
    def forward(self, x):
        if recording:
            output = super().forward(x)
            oracle.log('conv_output', output.cpu().numpy())
            return output
        else:  # Replay
            return torch.from_numpy(oracle.next('conv_output')).to(x.device)
```

**7.2.3 System Calls and Timing**

Intercept time-dependent operations:

```python
class DeterministicTime:
    def __init__(self):
        self.virtual_time = 0
        self.log = []

    def time(self):
        if recording:
            real_time = original_time()
            self.log.append(real_time)
            return real_time
        else:
            return self.log[self.virtual_time++]

    def sleep(self, duration):
        if recording:
            original_sleep(duration)
            self.log.append(('sleep', duration))
        else:
            # In replay, sleep is simulated
            self.virtual_time += duration
```

**7.2.4 Network I/O (Multi-Agent)**

For distributed agents, record message timing:

```python
class RecordedChannel:
    def send(self, msg, dest):
        if recording:
            timestamp = global_clock.time()
            oracle.log('send', (timestamp, self.id, dest, msg))
            original_send(msg, dest)
        else:
            # Replay sends at recorded times
            timestamp, _, _, _ = oracle.next('send')
            wait_until(timestamp)
            original_send(msg, dest)

    def recv(self):
        if recording:
            msg = original_recv()
            timestamp = global_clock.time()
            oracle.log('recv', (timestamp, self.id, msg))
            return msg
        else:
            timestamp, _, msg = oracle.next('recv')
            wait_until(timestamp)
            return msg
```

### 7.3 Checkpoint Management

**Checkpoint Contents:**

```python
@dataclass
class Checkpoint:
    timestamp: int
    agent_state: Dict[str, Any]  # Weights, buffers, optimizer state
    world_state: Dict[str, Any]  # Environment full state
    rng_states: Dict[str, Any]   # All RNG states
    oracle_position: int         # Position in oracle log
    metadata: Dict[str, Any]     # Debug info
```

**Adaptive Checkpoint Scheduling:**

```python
def should_checkpoint(t, state, oracle_log, config):
    # Regular interval
    if t % config.base_interval == 0:
        return True

    # State change threshold
    if state_change_magnitude(state) > config.change_threshold:
        return True

    # High activity (many oracle entries)
    recent_entries = len(oracle_log) - last_checkpoint_oracle_pos
    if recent_entries > config.activity_threshold:
        return True

    # Approaching potential failure
    if failure_probability(state) > config.failure_threshold:
        return True

    return False
```

**Checkpoint Compression:**

```python
def compress_checkpoint(checkpoint, prev_checkpoint):
    if prev_checkpoint is None:
        return full_serialize(checkpoint)

    # Delta encoding for weights
    weight_delta = {}
    for name, param in checkpoint.agent_state['weights'].items():
        prev_param = prev_checkpoint.agent_state['weights'][name]
        delta = param - prev_param
        if sparsity(delta) > 0.9:  # Mostly unchanged
            weight_delta[name] = sparse_encode(delta)
        else:
            weight_delta[name] = param

    # Reference previous for unchanged world state
    world_delta = diff(checkpoint.world_state, prev_checkpoint.world_state)

    return DeltaCheckpoint(
        base_ref=prev_checkpoint.id,
        weight_delta=weight_delta,
        world_delta=world_delta,
        rng_states=checkpoint.rng_states  # Always full
    )
```

### 7.4 Replay Execution

**Algorithm 2: Replay to Target Time**

```
procedure REPLAY_TO(trace, target_time):
    // Find nearest checkpoint before target
    checkpoint = find_nearest_checkpoint(trace.checkpoints, target_time)

    // Restore state from checkpoint
    state = restore_checkpoint(checkpoint)
    oracle = create_oracle_iterator(trace.oracle_log, checkpoint.oracle_position)

    // Replay from checkpoint to target
    for t = checkpoint.timestamp to target_time:
        oracle_entry = oracle.next()

        // Restore RNG states
        restore_rng_states(oracle_entry.rng_states)

        // Execute step (will be deterministic due to RNG)
        obs = environment.observe(state.world)
        action = agent.act(state.agent, obs)
        next_world, reward = environment.step(state.world, action)
        next_agent = agent.update(state.agent, obs, action, reward)

        // Verify determinism (debug mode)
        if DEBUG:
            assert action == oracle_entry.expected_action

        state = (next_agent, next_world)

    return state
```

**Fast-Forward Modes:**

1. **Full Replay:** Execute all steps (1× speed)
2. **Skip Replay:** Jump directly to checkpoints (1000× for sparse checkpoints)
3. **Selective Replay:** Only replay components of interest

```python
def fast_forward(trace, target_time, mode='auto'):
    if mode == 'full':
        return replay_to(trace, target_time)

    if mode == 'skip':
        checkpoint = find_nearest_checkpoint_before(trace, target_time)
        return restore_checkpoint(checkpoint)  # May not be exact

    if mode == 'auto':
        # Estimate replay cost
        checkpoint = find_nearest_checkpoint(trace, target_time)
        replay_steps = target_time - checkpoint.timestamp

        if replay_steps < THRESHOLD:
            return replay_to(trace, target_time)
        else:
            # Warn about imprecision
            return restore_checkpoint(checkpoint)
```

### 7.5 Correctness Proofs

**Theorem 3 (Replay Soundness).** If recording captures all non-determinism, replay produces identical execution:

$$\forall t \leq T: \text{State}_{replay}(t) = \text{State}_{original}(t)$$

_Proof._ By induction on $t$:

**Base case ($t = 0$):** Both start from identical initial state.

**Inductive step:** Assume $\text{State}_{replay}(t) = \text{State}_{original}(t)$.

At step $t+1$:

- Observation: Deterministic function of world state (equal by IH)
- Action: Deterministic given agent state + observation + RNG state (all equal)
- Transition: Deterministic given state + action + RNG (all equal)

Therefore $\text{State}_{replay}(t+1) = \text{State}_{original}(t+1)$. □

**Theorem 4 (Recording Completeness).** The oracle $\mathcal{N}$ captures all information necessary for replay.

_Proof._ We enumerate all non-determinism sources:

1. **RNG:** Captured via state snapshots before each call
2. **Floating-point:** Deterministic mode enforced; non-deterministic ops recorded
3. **System calls:** Time, I/O intercepted and logged
4. **External input:** Recorded at receipt

Any operation not in these categories is deterministic by definition, requiring no capture. □

---

## 8. Temporal Breakpoints

### 8.1 Temporal Predicate Language

Temporal breakpoints use a predicate language over execution traces:

**Syntax:**

```
φ ::= atom                     // Atomic predicate
    | φ ∧ φ | φ ∨ φ | ¬φ      // Boolean connectives
    | ◇φ | □φ                  // Eventually, Always (future)
    | ◇⁻φ | □⁻φ               // Eventually, Always (past)
    | φ S φ                    // Since (past)
    | φ U φ                    // Until (future)
    | @t φ                     // At specific time
    | [t₁, t₂] φ              // In time range

atom ::= state.field op value  // State comparison
       | action == value       // Action equality
       | reward op value       // Reward threshold
       | distance(x, y) op v   // Spatial predicate
       | caused(φ, φ)         // Causal relationship
```

**Semantics:**

$$(\mathcal{T}, t) \models \Diamond\phi \iff \exists t' \geq t: (\mathcal{T}, t') \models \phi$$
$$(\mathcal{T}, t) \models \Box\phi \iff \forall t' \geq t: (\mathcal{T}, t') \models \phi$$
$$(\mathcal{T}, t) \models \Diamond^-\phi \iff \exists t' \leq t: (\mathcal{T}, t') \models \phi$$
$$(\mathcal{T}, t) \models \phi_1 \mathcal{S} \phi_2 \iff \exists t' \leq t: (\mathcal{T}, t') \models \phi_2 \land \forall t'' \in (t', t]: (\mathcal{T}, t'') \models \phi_1$$

### 8.2 Example Breakpoints

**Find first collision:**

```
break when: collision(agent, any)
```

**Find decision that caused failure:**

```
break when: ◇⁻(action == "turn_left") ∧ (□⁻[0, 100] ¬collision) ∧ collision
// Break at turn_left that preceded first collision within 100 steps
```

**Stop when value drops:**

```
break when: Q_value < 0.5 ∧ ◇⁻[0, 10](Q_value > 0.8)
// Break when Q-value drops from >0.8 to <0.5 within 10 steps
```

**Multi-agent coordination failure:**

```
break when: distance(agent_1, agent_2) < threshold
         ∧ ◇⁻(message_sent(agent_1, agent_2))
         ∧ ¬message_received(agent_2, agent_1)
// Break when agents are close but message was lost
```

**Causal chain:**

```
break when: caused(action_t == "grab", reward_t+k < 0)
// Break when grab action causally led to negative reward
```

### 8.3 Breakpoint Evaluation

**Algorithm 3: Forward Breakpoint Evaluation**

```
procedure EVAL_FORWARD(trace, breakpoint_φ, start_t):
    for t = start_t to len(trace):
        state = replay_to(trace, t)

        if evaluate(φ, trace, t):
            return t  // Breakpoint hit

    return None  // No hit

procedure EVALUATE(φ, trace, t):
    match φ:
        case Atom(pred):
            return eval_atom(pred, trace[t])

        case And(φ₁, φ₂):
            return evaluate(φ₁, trace, t) and evaluate(φ₂, trace, t)

        case Eventually(ψ):
            for t' = t to len(trace):
                if evaluate(ψ, trace, t'):
                    return True
            return False

        case EventuallyPast(ψ):
            for t' = t downto 0:
                if evaluate(ψ, trace, t'):
                    return True
            return False

        case Since(φ₁, φ₂):
            for t' = t downto 0:
                if evaluate(φ₂, trace, t'):
                    # Check φ₁ holds in between
                    for t'' = t' + 1 to t:
                        if not evaluate(φ₁, trace, t''):
                            break
                    else:
                        return True
            return False
```

**Optimization: Incremental Evaluation**

For forward simulation, evaluate incrementally:

```python
class IncrementalEvaluator:
    def __init__(self, formula):
        self.formula = formula
        self.state = initialize_formula_state(formula)

    def step(self, trace_entry):
        # Update formula state with new entry
        self.state = update_state(self.state, self.formula, trace_entry)

        # Check if formula is satisfied
        return check_satisfaction(self.state, self.formula)
```

For past operators, maintain sliding window:

```python
class PastOperatorState:
    def __init__(self, window_size):
        self.window = deque(maxlen=window_size)
        self.eventually_past_satisfied = False

    def update(self, entry):
        self.window.append(entry)

        # Update ◇⁻ tracking
        if predicate(entry):
            self.eventually_past_satisfied = True
```

### 8.4 Backward Breakpoints

For predicates over future states, we use backward search:

**Algorithm 4: Backward Breakpoint Evaluation**

```
procedure EVAL_BACKWARD(trace, breakpoint_φ, end_t):
    // φ refers to future: find earliest t where φ holds

    // Precompute future-dependent values
    future_cache = compute_future_values(trace, φ)

    for t = 0 to end_t:
        if evaluate_with_future(φ, trace, t, future_cache):
            return t

    return None

procedure COMPUTE_FUTURE_VALUES(trace, φ):
    cache = {}

    // Backward pass for "eventually" and "until"
    for t = len(trace) - 1 downto 0:
        for subformula in future_subformulas(φ):
            match subformula:
                case Eventually(ψ):
                    cache[(t, subformula)] = (
                        evaluate(ψ, trace, t) or
                        cache.get((t+1, subformula), False)
                    )
                case Until(ψ₁, ψ₂):
                    cache[(t, subformula)] = (
                        evaluate(ψ₂, trace, t) or
                        (evaluate(ψ₁, trace, t) and
                         cache.get((t+1, subformula), False))
                    )

    return cache
```

### 8.5 Causal Breakpoints

For `caused(φ₁, φ₂)` predicates, we use counterfactual reasoning:

**Definition 11 (Causal Influence).** Event $E_1$ at time $t_1$ causes event $E_2$ at time $t_2 > t_1$ if:

1. Both $E_1$ and $E_2$ occurred in actual execution
2. In counterfactual where $E_1$ did not occur, $E_2$ would not occur

**Algorithm 5: Causal Breakpoint Evaluation**

```
procedure EVAL_CAUSED(φ_cause, φ_effect, trace, t):
    // Check if φ_cause holds in past
    cause_times = find_times_where(φ_cause, trace, 0, t)

    // Check if φ_effect holds at or after t
    if not evaluate(φ_effect, trace, t):
        return False

    // For each potential cause, test counterfactual
    for t_cause in cause_times:
        counterfactual = branch(trace, t_cause, negate(φ_cause))

        if not evaluate(φ_effect, counterfactual, t):
            // Effect didn't happen without cause
            return (True, t_cause)

    return False
```

---

## 9. Counterfactual Branching

### 9.1 Branch Operations

**Definition 12 (Branch Point).** A branch point $(t, \Delta)$ specifies:

- Time $t$: When to diverge
- Modification $\Delta$: What to change

**Modification Types:**

```python
@dataclass
class ActionModification:
    """Change the action taken at time t"""
    new_action: Action

@dataclass
class StateModification:
    """Modify agent or world state at time t"""
    state_updates: Dict[str, Any]

@dataclass
class PolicyModification:
    """Replace policy from time t onward"""
    new_policy: Policy

@dataclass
class EnvironmentModification:
    """Modify environment dynamics from time t"""
    env_changes: Dict[str, Any]
```

### 9.2 Branch Execution

**Algorithm 6: Counterfactual Branch**

```
procedure BRANCH(trace, t_branch, modification):
    // Restore state at branch point
    state = replay_to(trace, t_branch)

    // Apply modification
    state = apply_modification(state, modification)

    // Create new oracle for counterfactual
    cf_oracle = create_fresh_oracle(seed=hash(trace, t_branch, modification))

    // Execute counterfactual branch
    cf_trace = [trace[0:t_branch]]  // Share history

    for t = t_branch to trace.end_time:
        oracle_entry = cf_oracle.next()
        restore_rng_states(oracle_entry.rng_states)

        obs = environment.observe(state.world)
        action = modified_policy.act(state.agent, obs, modification)
        next_world, reward = environment.step(state.world, action)
        next_agent = agent.update(state.agent, obs, action, reward)

        cf_trace.append((state.agent, state.world, obs, action, reward))
        state = (next_agent, next_world)

    return CounterfactualTrace(
        branch_point=t_branch,
        modification=modification,
        original=trace,
        counterfactual=cf_trace
    )
```

### 9.3 Counterfactual Comparison

**Difference Analysis:**

```python
def compare_branches(original, counterfactual):
    """Compute differences between original and counterfactual execution."""

    divergence_point = counterfactual.branch_point

    differences = []
    for t in range(divergence_point, min(len(original), len(counterfactual))):
        orig_state = original[t]
        cf_state = counterfactual[t]

        diff = {
            'time': t,
            'state_diff': compute_state_diff(orig_state, cf_state),
            'action_diff': orig_state.action != cf_state.action,
            'reward_diff': orig_state.reward - cf_state.reward,
            'trajectory_divergence': compute_trajectory_distance(
                original[divergence_point:t],
                counterfactual[divergence_point:t]
            )
        }
        differences.append(diff)

    return BranchComparison(
        divergence_point=divergence_point,
        differences=differences,
        outcome_diff=compare_outcomes(original, counterfactual)
    )

def compute_outcome_attribution(original, counterfactuals):
    """Attribute outcome to specific decisions using counterfactual analysis."""

    attributions = {}
    original_outcome = compute_outcome(original)

    for cf in counterfactuals:
        cf_outcome = compute_outcome(cf.counterfactual)

        # Shapley-style attribution
        contribution = original_outcome - cf_outcome
        attributions[cf.modification] = contribution

    return attributions
```

### 9.4 Interactive Counterfactual Exploration

**What-If Analysis Interface:**

```python
class WhatIfExplorer:
    def __init__(self, trace):
        self.trace = trace
        self.branches = {}

    def what_if(self, time, modification):
        """Explore counterfactual: what if we did X at time T?"""
        key = (time, modification)

        if key not in self.branches:
            self.branches[key] = branch(self.trace, time, modification)

        return self.branches[key]

    def compare_outcomes(self, modifications):
        """Compare outcomes of multiple counterfactuals."""
        branches = [self.what_if(*m) for m in modifications]

        return {
            'original_outcome': compute_outcome(self.trace),
            'counterfactual_outcomes': [
                (m, compute_outcome(b)) for m, b in zip(modifications, branches)
            ],
            'best_alternative': max(branches, key=lambda b: compute_outcome(b))
        }

    def find_critical_decisions(self, outcome_threshold):
        """Find decisions where counterfactuals lead to different outcomes."""
        critical = []

        for t in range(len(self.trace)):
            for alt_action in enumerate_alternative_actions(self.trace, t):
                cf = self.what_if(t, ActionModification(alt_action))

                outcome_diff = abs(
                    compute_outcome(cf) - compute_outcome(self.trace)
                )

                if outcome_diff > outcome_threshold:
                    critical.append((t, alt_action, outcome_diff))

        return sorted(critical, key=lambda x: -x[2])
```

### 9.5 Counterfactual Semantics

**Formal Definition:**

Let $\mathcal{M}$ be the execution model, $\mathcal{T}$ be the trace, and $do(X_t = x)$ be an intervention at time $t$.

**Definition 13 (Counterfactual Trace).** The counterfactual trace $\mathcal{T}_{do(X_t = x)}$ is:

$$
\mathcal{T}_{do(X_t = x)}[t'] = \begin{cases}
\mathcal{T}[t'] & \text{if } t' < t \\
x & \text{if } t' = t \text{ and } X \text{ is intervened} \\
f_\mathcal{M}(\mathcal{T}_{do}[t'-1], \mathcal{N}[t']) & \text{if } t' > t
\end{cases}
$$

where $f_\mathcal{M}$ is the transition function and $\mathcal{N}[t']$ is a fresh oracle value (not from original execution).

**Theorem 5 (Counterfactual Consistency).** The counterfactual trace agrees with the original trace before the intervention point:

$$\forall t' < t: \mathcal{T}_{do(X_t = x)}[t'] = \mathcal{T}[t']$$

_Proof._ By construction, the counterfactual trace copies the original for $t' < t$. □

**Theorem 6 (Counterfactual Independence).** Different interventions produce independent counterfactual futures (given different oracle seeds):

$$\mathcal{T}_{do(X_t = x)} \perp \mathcal{T}_{do(X_t = x')} | \mathcal{T}[0:t]$$

_Proof._ Fresh oracle values for $t' > t$ are independent of the intervention. The counterfactual traces share history but diverge independently based on their interventions and oracle values. □

---

## 10. Multi-Agent Extensions

### 10.1 Synchronized Recording

For multi-agent systems, we need global synchronization:

```python
class MultiAgentRecorder:
    def __init__(self, agents, communication_channel):
        self.agents = agents
        self.channel = communication_channel
        self.global_clock = LamportClock()
        self.traces = {a.id: [] for a in agents}
        self.message_log = []

    def record_step(self):
        # Synchronize clocks
        self.global_clock.tick()
        global_time = self.global_clock.time

        # Record each agent
        for agent in self.agents:
            state = capture_agent_state(agent)
            oracle_entry = capture_oracle(agent)
            self.traces[agent.id].append((global_time, state, oracle_entry))

        # Record messages
        messages = self.channel.flush()
        for msg in messages:
            self.message_log.append((
                global_time,
                msg.sender,
                msg.receiver,
                msg.content,
                msg.lamport_time
            ))
```

### 10.2 Coordinated Replay

**Algorithm 7: Multi-Agent Replay**

```
procedure REPLAY_MULTI(traces, message_log, target_time):
    // Initialize all agents
    states = {}
    oracles = {}
    for agent_id, trace in traces:
        checkpoint = find_checkpoint(trace, target_time)
        states[agent_id] = restore(checkpoint)
        oracles[agent_id] = create_oracle(trace, checkpoint.oracle_pos)

    // Replay with message synchronization
    message_queue = PriorityQueue(message_log, key=lambda m: m.time)

    for t = checkpoint_time to target_time:
        // Deliver messages for this timestep
        while message_queue.peek().time == t:
            msg = message_queue.pop()
            deliver_message(states[msg.receiver], msg)

        // Step each agent
        for agent_id in agents:
            oracle_entry = oracles[agent_id].next()
            states[agent_id] = step_agent(states[agent_id], oracle_entry)

    return states
```

### 10.3 Selective Agent Branching

Branch one agent while others follow original trace:

```python
def branch_single_agent(multi_trace, agent_id, t_branch, modification):
    """Branch one agent, others follow original until they interact."""

    # Restore all agents to branch point
    states = replay_multi(multi_trace, t_branch)

    # Apply modification to target agent
    states[agent_id] = apply_modification(states[agent_id], modification)

    # Track which agents have been "contaminated" by counterfactual
    contaminated = {agent_id}

    cf_traces = {a: multi_trace[a][:t_branch] for a in agents}

    for t in range(t_branch, multi_trace.end_time):
        for a in agents:
            if a in contaminated:
                # Execute with counterfactual oracle
                states[a] = step_counterfactual(states[a])
            else:
                # Use original trace
                states[a] = replay_step(multi_trace[a], t)

            # Check for contamination via messages
            for msg in outgoing_messages(states[a]):
                if msg.sender in contaminated:
                    contaminated.add(msg.receiver)

        # Record counterfactual state
        for a in agents:
            cf_traces[a].append(states[a])

    return CounterfactualMultiTrace(
        branch_point=t_branch,
        modified_agent=agent_id,
        modification=modification,
        contaminated_agents=contaminated,
        traces=cf_traces
    )
```

---

## 11. Implementation

### 11.1 Language Support

**Python/PyTorch:**

- Monkey-patch numpy, torch random functions
- Wrap environment step functions
- Inject checkpoint hooks

**C++:**

- Link-time interposition for libc functions
- Compile-time instrumentation for custom RNGs
- Debug symbol integration for state inspection

**Rust:**

- Trait-based instrumentation
- Compile-time determinism verification
- Zero-cost abstractions for recording hooks

### 11.2 Storage Format

**Trace File Structure:**

```
trace.ttd/
├── metadata.json       # Trace info, agent configs
├── checkpoints/
│   ├── 0000000.ckpt   # Full checkpoint at t=0
│   ├── 0001000.ckpt   # Delta checkpoint at t=1000
│   └── ...
├── oracle/
│   ├── rng.bin        # RNG state log (compressed)
│   ├── system.bin     # System call log
│   └── nn_output.bin  # Neural network outputs (when needed)
├── index.bin          # Fast lookup index
└── messages.bin       # Multi-agent message log
```

**Compression:**

- LZ4 for speed (real-time recording)
- ZSTD for archival (post-hoc compression)
- Delta encoding for sequential data
- Sparse encoding for weight updates

### 11.3 Debug Interface

**Command-Line Interface:**

```bash
# Start replay session
ttd replay trace.ttd

# Navigation commands
(ttd) goto 1000              # Jump to timestep 1000
(ttd) back 10                # Go back 10 steps
(ttd) forward 100            # Go forward 100 steps
(ttd) break "reward < 0"     # Set breakpoint
(ttd) continue               # Run until breakpoint

# Inspection
(ttd) inspect agent.policy.weights[0]
(ttd) inspect world.robot_position
(ttd) diff 900 1000          # Compare states at t=900 and t=1000

# Counterfactual
(ttd) branch 500 action="turn_right"
(ttd) compare original branch_1
(ttd) what_if 500 "action='stop'" "action='accelerate'"
```

**Visual Interface:**

- Timeline view with event markers
- State diff visualization
- Branch tree navigator
- Multi-agent synchronization view

---

_Continued in Part 3: Evaluation, Case Studies, Discussion, and Conclusion_

---

## References (Partial - Part 2)

[Lamport, 1978] L. Lamport, "Time, clocks, and the ordering of events in a distributed system," Communications of the ACM, 1978.

[Pearl, 2009] J. Pearl, "Causality: Models, Reasoning, and Inference," Cambridge University Press, 2009.

[Pnueli, 1977] A. Pnueli, "The temporal logic of programs," FOCS, 1977.

---

_End of Part 2_
