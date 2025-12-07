# Time-Travel Debugging for Autonomous Agents: Deterministic Replay with Counterfactual Branching

## PLDI 2026 / OOPSLA 2026 - Submission Draft

**Part 1 of 3: Abstract, Introduction, Background, and Problem Formulation**

---

## Abstract

Debugging autonomous agents presents unique challenges: behaviors emerge from complex interactions between learned policies, environmental dynamics, and stochastic exploration, making failures difficult to reproduce and diagnose. Traditional debugging approaches—breakpoints, logging, and step-through execution—fail when agent behavior depends on precise timing, random seeds, and environmental state that cannot be easily reconstructed. We present **Time-Travel Debugging (TTD)**, a framework that enables developers to navigate agent execution history bidirectionally, set temporal breakpoints that trigger on past or future conditions, and explore counterfactual branches to answer "what-if" questions. TTD introduces three key innovations: (1) **Deterministic Replay** that captures sufficient state to reproduce any execution path bit-for-bit, including neural network inference, environment interactions, and inter-agent communication; (2) **Temporal Breakpoints** that halt execution when predicates over execution history become true, enabling root-cause analysis across time; and (3) **Counterfactual Branching** that forks execution at any point to explore alternative decisions without re-running from the beginning. Our implementation achieves 2.3× average recording overhead and supports replay at up to 1000× speedup through selective re-execution. Evaluation on multi-agent reinforcement learning, robotics simulation, and production autonomous systems demonstrates that TTD reduces debugging time by 73% compared to traditional methods, with developers successfully identifying root causes that were previously undiagnosable. We formalize the semantics of time-travel debugging for stateful agents and prove that our replay mechanism is sound (reproduces identical behavior) and complete (captures all information necessary for reproduction).

**Keywords:** debugging, deterministic replay, autonomous agents, counterfactual reasoning, program analysis

---

## 1. Introduction

### 1.1 The Autonomous Agent Debugging Crisis

Autonomous agents—systems that perceive, decide, and act with minimal human intervention—are becoming ubiquitous: from warehouse robots and self-driving vehicles to trading algorithms and game-playing AI. These systems learn complex behaviors through reinforcement learning, imitation, or evolutionary optimization, developing strategies that often surprise even their creators. This capability comes with a critical cost: **debugging becomes extremely difficult**.

Consider a multi-agent system where coordination suddenly fails after hours of successful operation. Traditional debugging approaches ask: "What is the system doing now?" But the root cause may lie in a decision made minutes or hours earlier—a subtle policy update, an unusual sensor reading, or an emergent interaction pattern. By the time the failure manifests, the causal chain is buried under millions of subsequent state transitions.

The challenge compounds for learned behaviors:

1. **Non-Determinism:** Random exploration, asynchronous communication, and hardware timing create different behaviors across runs
2. **Emergent Complexity:** Bugs arise from interactions between agents, not individual code paths
3. **Temporal Distance:** Root causes precede symptoms by arbitrary time spans
4. **State Explosion:** Agent state includes model weights, replay buffers, environment state, and communication history

Existing debugging tools—designed for sequential, deterministic programs—fundamentally mismatch agent systems. Breakpoints halt execution but cannot rewind to past states. Logging captures snapshots but misses the full context needed for reproduction. Unit tests verify components but miss emergent failures. Practitioners resort to "printf debugging" across distributed systems, often failing to reproduce bugs at all.

### 1.2 The Time-Travel Paradigm

We propose a paradigm shift: instead of debugging forward from the present, **debug bidirectionally through time**. Time-Travel Debugging (TTD) treats execution history as a first-class navigable structure, enabling developers to:

- **Rewind** to any past state and inspect it as if the system were live
- **Fast-forward** through execution, pausing at programmer-specified conditions
- **Branch** from any historical point to explore counterfactual scenarios
- **Compare** actual execution against alternative branches

This paradigm draws inspiration from reversible debugging [Engblom, 2012] and record-replay systems [Dunlap et al., 2002], but extends these ideas to handle the unique challenges of autonomous agents: continuous state spaces, neural network inference, stochastic policies, and multi-agent coordination.

### 1.3 Key Challenges

Building TTD for autonomous agents requires solving several technical challenges:

**Challenge 1: Deterministic Reproduction**
Agent behavior depends on random number generators (exploration), floating-point operations (neural networks), system calls (timing), and external inputs (sensors). Capturing enough information for bit-exact replay without prohibitive overhead requires careful instrumentation.

**Challenge 2: Efficient Storage**
Naive recording of all state at every timestep is prohibitively expensive. A typical agent generates megabytes of state per second. Hours of execution would require terabytes of storage. We need compression and selective recording strategies.

**Challenge 3: Temporal Navigation**
Rewinding execution is not simply "undoing"—it requires reconstructing the precise state from which forward execution would be identical. For neural networks, this means capturing not just weights but intermediate activations. For environments, this means full world state.

**Challenge 4: Counterfactual Semantics**
When branching from a historical point with modified decisions, what should happen to causally downstream state? We need formal semantics for counterfactual execution that preserve meaningful debugging utility.

**Challenge 5: Multi-Agent Coordination**
In multi-agent systems, rewinding one agent while others continue forward creates inconsistent global state. Coordinated replay across agents adds synchronization complexity.

### 1.4 Contributions

This paper presents Time-Travel Debugging (TTD), a comprehensive framework for debugging autonomous agents through temporal navigation. Our contributions include:

1. **Deterministic Replay Engine:** A recording and replay mechanism that captures sufficient information for bit-exact reproduction of agent execution, including neural network inference, environment interactions, and stochastic decisions. We prove soundness (replay matches original) and completeness (all necessary information captured).

2. **Temporal Breakpoints:** A predicate language for specifying conditions over execution history, enabling breakpoints that trigger on past events ("stop when the agent that caused this collision was deciding") or future consequences ("stop when this decision will lead to failure within 100 steps").

3. **Counterfactual Branching:** A mechanism for forking execution at any historical point with modified state, actions, or policies, creating alternative timelines that can be compared against actual execution.

4. **Efficient Implementation:** Recording overhead of 2.3× average, storage compression achieving 50× reduction, and replay speedup of up to 1000× through checkpoint-based fast-forwarding.

5. **Formal Semantics:** A denotational semantics for time-travel operations that clarifies the meaning of temporal navigation and counterfactual branching.

6. **Comprehensive Evaluation:** Experiments demonstrating 73% reduction in debugging time across multi-agent RL, robotics, and production systems.

### 1.5 Paper Organization

Section 2 provides background on debugging, record-replay, and autonomous agents. Section 3 formalizes the time-travel debugging problem. Part 2 (Methodology) details the TTD framework including replay, temporal breakpoints, and counterfactual branching with formal semantics. Part 3 (Evaluation) presents experiments and case studies.

---

## 2. Background

### 2.1 Traditional Debugging

Debugging has evolved from early core dumps and print statements to sophisticated interactive environments:

**Breakpoint Debugging:** Halt execution at specified code locations, inspect state, then continue. GDB [Stallman et al., 2002], LLDB, and IDE debuggers provide this capability. Limitation: Cannot inspect past states without re-execution.

**Logging:** Record events during execution for post-hoc analysis. Structured logging (Log4j, Serilog) and distributed tracing (Jaeger, Zipkin) provide visibility. Limitation: Limited to pre-determined events; cannot query arbitrary state.

**Profiling:** Measure performance characteristics—CPU time, memory, I/O. Flamegraphs, sampling profilers, and hardware counters provide insights. Limitation: Focuses on performance, not correctness.

**Static Analysis:** Analyze code without execution to find potential bugs. Type systems, linters, and abstract interpretation catch certain error classes. Limitation: False positives; cannot analyze learned behaviors.

### 2.2 Record-Replay Systems

Record-replay captures execution for later deterministic reproduction:

**rr [O'Callahan et al., 2017]:** Records Linux process execution including system calls, signals, and scheduling. Enables reverse debugging for single-threaded and multi-threaded programs. Limitation: Significant overhead for I/O-heavy workloads; no support for GPU or neural networks.

**PANDA [Dolan-Gavitt et al., 2015]:** Whole-system record-replay built on QEMU. Records all non-determinism including interrupts and DMA. Limitation: High overhead; complex deployment.

**ReVirt [Dunlap et al., 2002]:** Virtual machine-level record-replay. Captures hypervisor-level events. Limitation: Cannot selectively record; all-or-nothing approach.

**Mozilla rr:** Production-quality reverse debugger for Firefox development. Demonstrates practical utility of time-travel debugging for complex software.

**Limitations for Agents:** Existing systems focus on general-purpose programs. They lack:

- Understanding of agent structure (policy, environment, buffer)
- Semantic compression (recording policy outputs vs. full inference)
- Counterfactual branching (what-if analysis)
- Multi-agent coordination

### 2.3 Autonomous Agent Architectures

Modern autonomous agents share common architectural patterns:

**Policy Network:** Neural network mapping observations to actions:
$$\pi_\theta: \mathcal{O} \rightarrow \mathcal{A}$$

**Value Function:** Estimates expected return from states:
$$V_\phi: \mathcal{S} \rightarrow \mathbb{R}$$

**Environment Interface:** Observation, action, reward loop:
$$o_{t+1}, r_t = \text{env.step}(a_t)$$

**Experience Replay:** Buffer of past transitions for training:
$$\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1})\}$$

**Communication (Multi-Agent):** Message passing between agents:
$$m_i^t = \text{send}_i(s_i^t), \quad s_i^{t+1} = \text{receive}_i(m_{-i}^t)$$

Each component contributes state that must be captured for debugging.

### 2.4 Agent Debugging Challenges

**Temporal Credit Assignment:** When a multi-agent system fails, which agent's decision caused the failure? The causal chain may span many timesteps and multiple agents.

**Emergent Behavior Bugs:** Bugs arise from interactions, not individual components. Unit testing agents in isolation misses coordination failures.

**Non-Deterministic Execution:** Random seeds, asynchronous communication, and floating-point non-associativity create different behaviors across runs.

**State Space Size:** A single agent may have millions of parameters, plus environment state, plus replay buffer. Recording everything is prohibitive.

**Distributed Execution:** Agents often run on different machines with unsynchronized clocks and unreliable communication.

### 2.5 Related Work on Agent Debugging

**Interpretability Tools:** Attention visualization [Vaswani et al., 2017], saliency maps [Simonyan et al., 2014], and concept activation [Kim et al., 2018] help understand what networks compute but don't support interactive debugging.

**Simulation Determinism:** Physics engines (MuJoCo, PyBullet) offer deterministic modes, but achieving full determinism across the stack remains challenging.

**Causal Debugging:** Techniques for identifying causal relationships in program traces [Zeller, 2002]. Delta debugging narrows failure-inducing changes. Limitation: Requires multiple executions; doesn't support in-trace navigation.

**Formal Verification:** Model checking and theorem proving can verify agent properties [Katz et al., 2017]. Limitation: Scalability; cannot handle learned neural policies.

---

## 3. Problem Formulation

### 3.1 Agent Execution Model

**Definition 1 (Agent).** An agent $\mathcal{A}$ is a tuple $(\mathcal{S}, \mathcal{O}, \mathcal{A}, \pi, \delta)$ where:

- $\mathcal{S}$ is the internal state space (weights, buffers, etc.)
- $\mathcal{O}$ is the observation space
- $\mathcal{A}$ is the action space
- $\pi: \mathcal{S} \times \mathcal{O} \rightarrow \mathcal{A} \times \mathcal{R}$ is the policy (deterministic given random seed $\mathcal{R}$)
- $\delta: \mathcal{S} \times \mathcal{O} \times \mathcal{A} \times \mathcal{R} \rightarrow \mathcal{S}$ is the internal state transition (learning, buffer updates)

**Definition 2 (Environment).** An environment $\mathcal{E}$ is a tuple $(\mathcal{W}, \omega, \tau)$ where:

- $\mathcal{W}$ is the world state space
- $\omega: \mathcal{W} \rightarrow \mathcal{O}$ generates observations
- $\tau: \mathcal{W} \times \mathcal{A} \times \mathcal{R} \rightarrow \mathcal{W} \times \mathbb{R}$ is the world transition (includes reward)

**Definition 3 (Execution Trace).** An execution trace $\mathcal{T}$ is a sequence of snapshots:
$$\mathcal{T} = [(s_0, w_0, o_0, a_0, r_0), (s_1, w_1, o_1, a_1, r_1), \ldots]$$

where each tuple contains agent internal state, world state, observation, action, and reward.

### 3.2 Non-Determinism Sources

Execution depends on multiple sources of non-determinism:

**Random Number Generators (RNGs):**

- Exploration noise: $\epsilon$-greedy, Gaussian noise
- Dropout during inference (if enabled)
- Environment stochasticity

**Floating-Point Non-Determinism:**

- GPU kernel launch order
- Parallel reduction associativity
- Library version differences

**System Non-Determinism:**

- Thread scheduling
- Network latency (multi-agent)
- Timer interrupts

**Input Non-Determinism:**

- Sensor noise
- Human inputs
- External API responses

**Definition 4 (Non-Determinism Oracle).** A non-determinism oracle $\mathcal{N}$ is a sequence of values resolving all non-deterministic choices:
$$\mathcal{N} = [n_0, n_1, n_2, \ldots]$$

Given the same initial state and oracle, execution is deterministic.

### 3.3 Time-Travel Debugging Requirements

**Requirement 1 (Deterministic Replay).** Given a recorded trace $\mathcal{T}$ and oracle $\mathcal{N}$, replay produces identical trace:
$$\text{Replay}(\mathcal{T}_0, \mathcal{N}) = \mathcal{T}$$

where $\mathcal{T}_0$ is the initial state.

**Requirement 2 (Bidirectional Navigation).** The debugger supports:

- $\text{Forward}(t, \Delta t)$: Advance from time $t$ by $\Delta t$ steps
- $\text{Backward}(t, \Delta t)$: Rewind from time $t$ by $\Delta t$ steps
- $\text{Goto}(t')$: Jump directly to time $t'$

**Requirement 3 (Temporal Breakpoints).** Support predicates over execution history:
$$\text{Break}(\phi) \text{ where } \phi: \mathcal{T}^* \rightarrow \{true, false\}$$

Execution halts at the first $t$ where $\phi(\mathcal{T}_{0:t})$ becomes true.

**Requirement 4 (Counterfactual Branching).** Support forking execution with modifications:
$$\mathcal{T}' = \text{Branch}(\mathcal{T}, t, \Delta)$$

where $\Delta$ specifies modifications (changed action, modified state, different policy).

### 3.4 Efficiency Requirements

**Definition 5 (Recording Overhead).** The slowdown factor during recording:
$$\text{Overhead} = \frac{T_{recorded}}{T_{normal}}$$

Target: Overhead < 5× for practical use.

**Definition 6 (Storage Efficiency).** Bytes per timestep of recorded data:
$$\text{Storage} = \frac{|\text{RecordedData}|}{|\mathcal{T}|}$$

Target: Comparable to compressed video (< 1MB per second of execution).

**Definition 7 (Replay Speed).** The speedup factor during replay (vs. original execution):
$$\text{ReplaySpeed} = \frac{T_{original}}{T_{replay}}$$

Target: Support both real-time replay and fast-forward (10-1000×).

### 3.5 Formal Properties

**Property 1 (Soundness).** Replay is sound if it produces identical behavior:
$$\forall t: \mathcal{T}_{replay}[t] = \mathcal{T}_{original}[t]$$

**Property 2 (Completeness).** Recording is complete if all information necessary for replay is captured:
$$\forall \mathcal{T}: \exists \mathcal{N}: \text{Replay}(\mathcal{T}_0, \mathcal{N}) = \mathcal{T}$$

**Property 3 (Causal Consistency).** Counterfactual branches preserve causality:
$$\text{Branch}(\mathcal{T}, t, \Delta) \text{ agrees with } \mathcal{T} \text{ for } t' < t$$

### 3.6 Multi-Agent Extension

For multi-agent systems with $N$ agents:

**Definition 6 (Multi-Agent Trace).** A multi-agent trace $\mathcal{T}^{MA}$ is a collection of synchronized agent traces:
$$\mathcal{T}^{MA} = (\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_N, \mathcal{M})$$

where $\mathcal{M}$ is the message log capturing inter-agent communication.

**Requirement 5 (Synchronized Replay).** Multi-agent replay preserves message ordering and timing:
$$\forall i, j, t: \text{Msg}(i \rightarrow j, t)_{replay} = \text{Msg}(i \rightarrow j, t)_{original}$$

**Requirement 6 (Selective Branching).** Support branching individual agents while others continue original execution:
$$\mathcal{T}'^{MA} = \text{Branch}(\mathcal{T}^{MA}, t, \Delta_i)$$

where $\Delta_i$ modifies only agent $i$, and other agents respond to $i$'s new behavior.

---

## 4. Preliminaries

### 4.1 Notation Summary

| Symbol        | Description                      |
| ------------- | -------------------------------- |
| $\mathcal{A}$ | Agent tuple                      |
| $\mathcal{E}$ | Environment tuple                |
| $\mathcal{T}$ | Execution trace                  |
| $\mathcal{N}$ | Non-determinism oracle           |
| $\pi$         | Policy function                  |
| $s_t$         | Agent internal state at time $t$ |
| $w_t$         | World state at time $t$          |
| $o_t$         | Observation at time $t$          |
| $a_t$         | Action at time $t$               |
| $\phi$        | Temporal predicate               |
| $\Delta$      | Counterfactual modification      |

### 4.2 Assumptions

**Assumption 1 (Deterministic Policy).** Given internal state, observation, and RNG seed, policy output is deterministic:
$$\pi(s, o, r) = a \text{ deterministically}$$

**Assumption 2 (Capturable Non-Determinism).** All sources of non-determinism can be captured by the oracle $\mathcal{N}$.

**Assumption 3 (Finite State).** Agent and environment states are finite (bounded precision) and serializable.

**Assumption 4 (Causal Structure).** Actions at time $t$ affect state only at times $t' > t$:
$$a_t \text{ influences } s_{t'}, w_{t'} \text{ only if } t' > t$$

### 4.3 Threat Model for Correctness

**Trusted Computing Base:**

- Recording infrastructure must faithfully capture all non-determinism
- Replay engine must correctly re-execute given captured oracle
- Serialization/deserialization must be lossless

**Out of Scope:**

- Hardware faults during recording
- Adversarial manipulation of recordings
- Bugs in the debugger itself

### 4.4 Complexity Model

Let $|\mathcal{T}| = T$ be trace length, $|s|$ be agent state size, $|w|$ be world state size.

**Recording Complexity:**

- Time: $O(T \cdot (|s| + |w|))$ for full capture
- Space: $O(T \cdot (|s| + |w|))$ uncompressed

**Replay Complexity:**

- Time: $O(T)$ for full replay, $O(1)$ for checkpoint-based random access
- Space: $O(\sqrt{T} \cdot (|s| + |w|))$ for checkpoint storage

**Query Complexity:**

- Temporal breakpoint: $O(T)$ worst case (scan entire trace)
- Counterfactual branch: $O(T - t)$ where $t$ is branch point

---

## 5. Theoretical Foundations

### 5.1 Execution as State Machine

Agent execution defines a state machine:

**Definition 7 (Execution State Machine).** The execution state machine $\mathcal{M}$ is:

- States: $(s, w, t) \in \mathcal{S} \times \mathcal{W} \times \mathbb{N}$
- Initial: $(s_0, w_0, 0)$
- Transition: $(s, w, t) \xrightarrow{n} (s', w', t+1)$ where $n$ is oracle value

The machine is deterministic given the oracle sequence.

### 5.2 Reversibility

**Definition 8 (Reversible State Machine).** A state machine is reversible if each state has at most one predecessor:
$$\forall (s', w'): |\{(s, w): (s, w) \rightarrow (s', w')\}| \leq 1$$

Agent execution is generally **not** reversible—multiple states may lead to the same successor (information loss). This motivates recording rather than computing backward.

**Theorem 1 (Recording Necessity).** For non-reversible execution, achieving O(1) backward navigation requires O(T) storage or O(T) re-computation.

_Proof._ Without recording, reaching state $(s_t, w_t)$ requires either:

1. Re-executing from $(s_0, w_0)$ taking $O(t)$ time
2. Storing checkpoints at density $d$, taking $O(T/d)$ space and $O(d)$ worst-case re-computation

The product is minimized at $d = O(\sqrt{T})$, giving $O(\sqrt{T})$ time and space. For $O(1)$ access, full recording is necessary. □

### 5.3 Checkpoint Strategy

**Definition 9 (Checkpoint Schedule).** A checkpoint schedule $C \subseteq \{0, 1, \ldots, T\}$ specifies which timesteps have saved full state.

**Optimal Checkpoint Spacing:**
Given storage budget $B$ and uniform access distribution, optimal spacing is:
$$d^* = \sqrt{\frac{2 \cdot T \cdot c_{replay}}{c_{checkpoint}}}$$

where $c_{replay}$ is cost per replayed step and $c_{checkpoint}$ is cost per checkpoint.

For typical agents: $c_{replay} \approx 1ms$, $c_{checkpoint} \approx 10ms$, yielding $d^* \approx 4.5\sqrt{T}$.

**Adaptive Checkpointing:**
Increase checkpoint density during:

- High activity (many state changes)
- Critical phases (near failures)
- User-specified regions of interest

### 5.4 Oracle Compression

The non-determinism oracle $\mathcal{N}$ can be compressed:

**RNG Compression:**
Instead of recording each random value, record RNG seeds:
$$\text{RNG}(seed) \rightarrow [r_0, r_1, r_2, \ldots]$$

One seed (32-64 bits) replaces thousands of random values.

**Delta Encoding:**
For slowly-changing values (weights during inference), record deltas:
$$\Delta w_t = w_t - w_{t-1}$$

Deltas are often zero or small, enabling efficient compression.

**Semantic Compression:**
For neural networks, record action outputs rather than all intermediate activations:
$$\text{Record}(\pi(s, o)) \text{ rather than } \text{Record}(\text{all activations})$$

Activations can be recomputed from inputs during replay.

### 5.5 Counterfactual Semantics

**Definition 10 (Counterfactual World).** A counterfactual world $\mathcal{T}'$ branching from $\mathcal{T}$ at time $t$ with intervention $do(X = x)$ is:

$$
\mathcal{T}'[t'] = \begin{cases}
\mathcal{T}[t'] & \text{if } t' < t \\
\text{Execute}(\mathcal{T}[t-1], x, \mathcal{N}_{t:}) & \text{if } t' \geq t
\end{cases}
$$

The counterfactual preserves history before intervention and computes new futures.

**Theorem 2 (Counterfactual Consistency).** Counterfactual branching preserves causal relationships:
If $X_t$ does not causally affect $Y_{t'}$ for $t' < t$, then:
$$Y_{t'}^{cf} = Y_{t'}^{actual}$$

_Proof._ By causal structure (Assumption 4), interventions at $t$ cannot affect past states. The counterfactual world agrees with actual world for all $t' < t$ by construction. □

---

_Continued in Part 2: Methodology (Deterministic Replay, Temporal Breakpoints, Counterfactual Branching)_

---

## References (Partial - Part 1)

[Dolan-Gavitt et al., 2015] B. Dolan-Gavitt et al., "Repeatable reverse engineering with PANDA," PPREW, 2015.

[Dunlap et al., 2002] G. Dunlap et al., "ReVirt: Enabling intrusion analysis through virtual-machine logging and replay," OSDI, 2002.

[Engblom, 2012] J. Engblom, "A review of reverse debugging," S4D, 2012.

[Katz et al., 2017] G. Katz et al., "Reluplex: An efficient SMT solver for verifying deep neural networks," CAV, 2017.

[Kim et al., 2018] B. Kim et al., "Interpretability beyond feature attribution: Quantitative testing with concept activation vectors," ICML, 2018.

[O'Callahan et al., 2017] R. O'Callahan et al., "Engineering record and replay for deployability," USENIX ATC, 2017.

[Simonyan et al., 2014] K. Simonyan et al., "Deep inside convolutional networks: Visualising image classification models and saliency maps," ICLR Workshop, 2014.

[Stallman et al., 2002] R. Stallman, R. Pesch, and S. Shebs, "Debugging with GDB," Free Software Foundation, 2002.

[Vaswani et al., 2017] A. Vaswani et al., "Attention is all you need," NeurIPS, 2017.

[Zeller, 2002] A. Zeller, "Isolating cause-effect chains from computer programs," FSE, 2002.

---

_End of Part 1_
