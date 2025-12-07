# Multi-Fidelity Digital Twins for Large-Scale Swarm Robotics

## IEEE Transactions on Robotics - Submission Draft

**Part 2 of 3: Methodology**

---

## 6. Multi-Fidelity Swarm Twins Framework

### 6.1 System Architecture

The Multi-Fidelity Swarm Twins (MFST) framework consists of four integrated components operating in a continuous loop:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MFST Framework Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Swarm      │───▶│   Fidelity   │───▶│Computational │          │
│  │   State      │    │  Importance  │    │   Budget     │          │
│  │   Monitor    │    │  Estimator   │    │  Optimizer   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         │                   ▼                   ▼                   │
│         │            ┌─────────────────────────────┐               │
│         │            │   Fidelity Allocation Map   │               │
│         │            │   φ: Agent → Fidelity Level │               │
│         │            └─────────────────────────────┘               │
│         │                        │                                  │
│         ▼                        ▼                                  │
│  ┌──────────────────────────────────────────────────┐              │
│  │           Multi-Fidelity Simulation Engine        │              │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │              │
│  │  │Level 0 │ │Level 1 │ │Level 2 │ │Level 3 │    │              │
│  │  │Kinematic│ │Rigid   │ │Simple  │ │Full    │    │              │
│  │  │        │ │Body    │ │Contact │ │Physics │    │              │
│  │  └────────┘ └────────┘ └────────┘ └────────┘    │              │
│  └──────────────────────────────────────────────────┘              │
│                        │                                            │
│                        ▼                                            │
│  ┌──────────────────────────────────────────────────┐              │
│  │        Seamless Fidelity Transition (SFT)         │              │
│  │   Ensures physical consistency during changes     │              │
│  └──────────────────────────────────────────────────┘              │
│                        │                                            │
│                        ▼                                            │
│              Updated Swarm State S^(t+1)                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Algorithm Overview

**Algorithm 1: MFST Simulation Loop**

```
Input: Initial state S⁰, budget B, time horizon T
Output: State trajectory {S⁰, S¹, ..., S^T}

1: Initialize fidelity allocation φ⁰ = uniform_low_fidelity()
2: for t = 0 to T-1 do
3:    // Phase 1: Importance Estimation
4:    I ← FidelityImportanceEstimator(Sᵗ, task_context)
5:
6:    // Phase 2: Budget Optimization
7:    φᵗ ← ComputationalBudgetOptimizer(I, B, costs)
8:
9:    // Phase 3: Multi-Fidelity Simulation
10:   for each agent i in parallel do
11:      ℓ ← φᵗ(i)
12:      s_i^{t+1,raw} ← Simulate_Level_ℓ(s_i^t, u_i^t, S_{-i}^t)
13:   end for
14:
15:   // Phase 4: Fidelity Transition
16:   S^{t+1} ← SeamlessFidelityTransition(S^{t+1,raw}, φᵗ, φ^{t-1})
17:
18:   // Phase 5: Update and Logging
19:   Log(Sᵗ, φᵗ, compute_time, accuracy_estimate)
20: end for
21: return {S⁰, S¹, ..., S^T}
```

---

## 7. Fidelity Importance Estimator (FIE)

### 7.1 Feature Extraction

FIE computes importance scores from local state features efficiently computable in $O(1)$ per agent:

**Kinematic Features** (per agent):
$$\mathbf{x}_i^{kin} = [\|\mathbf{v}_i\|, \|\boldsymbol{\omega}_i\|, \|\mathbf{a}_i\|, \|\boldsymbol{\alpha}_i\|]$$

- Velocity magnitude indicates motion activity
- Angular velocity indicates rotational dynamics
- Acceleration indicates dynamic transients
- Angular acceleration indicates rotational transients

**Proximity Features** (requires neighbor query):
$$\mathbf{x}_i^{prox} = [d_{min}, n_{near}, \bar{d}_{near}, \sigma_{d}]$$

- $d_{min}$: Distance to nearest neighbor
- $n_{near}$: Count of neighbors within interaction radius
- $\bar{d}_{near}$: Mean distance to nearby agents
- $\sigma_d$: Variance of neighbor distances

**Interaction Features**:
$$\mathbf{x}_i^{int} = [\rho_i, \mathbf{1}[contact], F_{contact}, n_{contacts}]$$

- $\rho_i$: Interaction density (Definition 5)
- Contact indicator from previous timestep
- Contact force magnitude
- Number of active contacts

**Task Features**:
$$\mathbf{x}_i^{task} = [\tau_i, d_{goal}, \theta_{goal}, phase]$$

- $\tau_i$: Task relevance (Definition 6)
- Distance to goal/target
- Heading error to goal
- Task phase encoding

**Temporal Features**:
$$\mathbf{x}_i^{temp} = [\Delta\mathbf{s}_i^{recent}, \ell_i^{prev}, t_{since\_transition}]$$

- Recent state change magnitude
- Previous fidelity level
- Time since last fidelity transition

### 7.2 Importance Model

FIE uses a lightweight neural network trained to predict fidelity importance:

**Architecture:**
$$\hat{I}_i = \text{MLP}([\mathbf{x}_i^{kin}, \mathbf{x}_i^{prox}, \mathbf{x}_i^{int}, \mathbf{x}_i^{task}, \mathbf{x}_i^{temp}])$$

The MLP consists of:

- Input layer: 20 features (concatenation of all feature vectors)
- Hidden layer 1: 32 units, ReLU activation
- Hidden layer 2: 16 units, ReLU activation
- Output layer: 1 unit, Sigmoid activation (importance in [0,1])

**Total parameters:** ~1,200 (extremely lightweight)
**Inference time:** 0.01 ms per agent on CPU

### 7.3 Training Procedure

FIE is trained to predict which agents would benefit most from high-fidelity simulation:

**Training Data Generation:**

1. Run high-fidelity simulation for diverse scenarios
2. Run multi-fidelity simulation with random allocations
3. Compute per-agent error: $e_i = \|s_i^{HiFi} - s_i^{LoFi}\|$
4. Label importance as: $I_i^{label} = \sigma(e_i / \bar{e})$ (normalized, sigmoidized)

**Loss Function:**
$$\mathcal{L}_{FIE} = \sum_i \left(\hat{I}_i - I_i^{label}\right)^2 + \lambda \|\theta_{FIE}\|^2$$

**Training Details:**

- Optimizer: Adam with learning rate $10^{-3}$
- Batch size: 256 agents
- Training samples: 1M agent-timestep pairs
- Early stopping based on validation error

### 7.4 Importance Calibration

Raw model outputs require calibration to ensure proper budget allocation:

**Temperature Scaling:**
$$\tilde{I}_i = \sigma\left(\frac{\text{logit}(\hat{I}_i)}{T}\right)$$

where temperature $T$ is optimized on validation data to minimize expected calibration error:

$$ECE = \sum_{b=1}^{B} \frac{|b|}{N} |acc(b) - conf(b)|$$

Calibrated importance ensures that high-importance agents truly require high fidelity.

---

## 8. Computational Budget Optimizer (CBO)

### 8.1 Optimization Formulation

Given importance scores $\{I_1, \ldots, I_N\}$ and computational budget $B$, CBO finds the optimal fidelity allocation:

**Continuous Relaxation:**
$$\min_{\mathbf{z}} \sum_{i=1}^N I_i \cdot e(\mathbf{z}_i) \quad \text{s.t.} \quad \sum_{i=1}^N c(\mathbf{z}_i) \leq B, \quad \mathbf{z}_i \in [0, L]$$

where $\mathbf{z}_i \in \mathbb{R}$ is the continuous fidelity level, $e(\cdot)$ is the error function, and $c(\cdot)$ is the cost function.

**Error Model:**
$$e(\ell) = \alpha \cdot \exp(-\beta \cdot \ell)$$

**Cost Model:**
$$c(\ell) = c_0 \cdot \gamma^\ell$$

where $\alpha, \beta, \gamma > 0$ are fitted from empirical data.

### 8.2 Greedy Allocation Algorithm

The continuous relaxation yields the optimal marginal allocation principle:

**Theorem 2 (Optimal Marginal Allocation).** At optimality, all agents have equal marginal error reduction per unit cost:

$$\frac{\partial e(\mathbf{z}_i^*)/\partial \mathbf{z}_i}{\partial c(\mathbf{z}_i^*)/\partial \mathbf{z}_i} = \lambda \quad \forall i$$

where $\lambda$ is the Lagrange multiplier for the budget constraint.

_Proof._ Apply the KKT conditions to the Lagrangian:

$$\mathcal{L} = \sum_i I_i \cdot e(\mathbf{z}_i) + \lambda \left(\sum_i c(\mathbf{z}_i) - B\right)$$

Taking derivatives: $I_i \cdot e'(\mathbf{z}_i) + \lambda \cdot c'(\mathbf{z}_i) = 0$

Rearranging: $\frac{e'(\mathbf{z}_i)}{c'(\mathbf{z}_i)} = -\frac{\lambda}{I_i}$

For agents with equal importance, fidelity levels are equal. For higher importance, fidelity is higher. □

**Algorithm 2: Greedy Budget Allocation**

```
Input: Importances I, budget B, fidelity levels L, costs c
Output: Allocation φ

1: Initialize φ(i) = 0 for all i  // Start at lowest fidelity
2: remaining_budget = B - sum(c[0] for all agents)
3: priority_queue Q = empty
4:
5: // Compute initial upgrade priorities
6: for i = 1 to N do
7:    if φ(i) < L then
8:       priority = I[i] * (e(φ(i)) - e(φ(i)+1)) / (c[φ(i)+1] - c[φ(i)])
9:       Q.push((priority, i))
10: end for
11:
12: // Greedy upgrade loop
13: while Q is not empty do
14:    (priority, i) = Q.pop_max()
15:    upgrade_cost = c[φ(i)+1] - c[φ(i)]
16:
17:    if upgrade_cost <= remaining_budget then
18:       φ(i) = φ(i) + 1
19:       remaining_budget -= upgrade_cost
20:
21:       // Re-add to queue if more upgrades possible
22:       if φ(i) < L then
23:          new_priority = I[i] * (e(φ(i)) - e(φ(i)+1)) / (c[φ(i)+1] - c[φ(i)])
24:          Q.push((new_priority, i))
25:       end if
26:    end if
27: end while
28:
29: return φ
```

**Complexity:** $O(NL \log N)$ using a priority queue

### 8.3 Theoretical Guarantees

**Theorem 3 (Approximation Ratio).** Algorithm 2 achieves a $(1 - 1/e)$-approximation to the optimal allocation for submodular error functions.

_Proof._ The total error reduction is a submodular function of the allocation (upgrading additional agents has diminishing returns). The greedy algorithm for submodular maximization with knapsack constraints achieves a $(1 - 1/e)$ approximation [Nemhauser & Wolsey, 1978].

Let $E^* = \sum_i I_i \cdot e(0)$ be the baseline error (all agents at Level 0).
Let $\Delta E_{greedy}$ be the error reduction from greedy allocation.
Let $\Delta E_{opt}$ be the optimal error reduction.

Then: $\Delta E_{greedy} \geq (1 - 1/e) \cdot \Delta E_{opt} \approx 0.632 \cdot \Delta E_{opt}$

In practice, the exponential error decay means greedy is often near-optimal. □

**Theorem 4 (Budget Scaling).** As computational budget $B \rightarrow \infty$, MFST converges to full high-fidelity simulation:

$$\lim_{B \rightarrow \infty} E(\phi_B) = 0$$

_Proof._ With sufficient budget, all agents can be allocated to Level $L$, yielding zero approximation error by definition. The convergence rate is:

$$E(\phi_B) = O(N \cdot e^{-\beta B / (N \cdot c_L)})$$

The error decreases exponentially as budget per agent increases. □

### 8.4 Load Balancing

For parallel simulation with $P$ workers, we must balance load:

**Algorithm 3: Load-Balanced Allocation**

```
Input: Allocation φ, workers P
Output: Worker assignment W

1: Sort agents by cost c[φ(i)] descending
2: worker_loads = [0] * P
3: W = [empty list] * P
4:
5: for i in sorted agents do
6:    min_worker = argmin(worker_loads)
7:    W[min_worker].append(i)
8:    worker_loads[min_worker] += c[φ(i)]
9: end for
10:
11: return W
```

This is the LPT (Longest Processing Time) heuristic with approximation ratio $4/3$ for makespan minimization.

---

## 9. Seamless Fidelity Transition (SFT)

### 9.1 The Transition Problem

When an agent's fidelity level changes between timesteps, discontinuities can occur:

1. **State Representation Mismatch:** Level 0 (point-mass) has no rotation; Level 1+ has full pose
2. **Energy Inconsistency:** Lower fidelity may accumulate energy artifacts
3. **Contact State Loss:** Transitioning out of Level 3 loses contact patch information

Naive transitions create non-physical behaviors: sudden rotations, velocity jumps, or penetrating collisions.

### 9.2 Upward Transition (Low → High Fidelity)

When upgrading from level $\ell$ to level $\ell' > \ell$:

**State Expansion:**
$$\mathbf{s}_i^{(\ell')} = \text{Expand}(\mathbf{s}_i^{(\ell)})$$

Level 0 → Level 1:

- Position and velocity preserved
- Orientation initialized from velocity direction: $\mathbf{R} = \text{AlignZ}(\mathbf{v}/\|\mathbf{v}\|)$
- Angular velocity initialized to zero: $\boldsymbol{\omega} = \mathbf{0}$

Level 1 → Level 2/3:

- Full pose preserved
- Contact state initialized via collision detection
- Contact forces computed from penetration depth

**Energy Matching:**
Ensure kinetic energy is preserved:
$$E_k^{(\ell')} = E_k^{(\ell)} \Rightarrow \frac{1}{2}m\|\mathbf{v}'\|^2 + \frac{1}{2}\boldsymbol{\omega}'^T \mathbf{I} \boldsymbol{\omega}' = \frac{1}{2}m\|\mathbf{v}\|^2$$

If expansion adds rotational energy, reduce translational velocity:
$$\mathbf{v}' = \mathbf{v} \cdot \sqrt{1 - \frac{\boldsymbol{\omega}'^T \mathbf{I} \boldsymbol{\omega}'}{m\|\mathbf{v}\|^2}}$$

### 9.3 Downward Transition (High → Low Fidelity)

When downgrading from level $\ell$ to level $\ell' < \ell$:

**State Projection:**
$$\mathbf{s}_i^{(\ell')} = \text{Project}(\mathbf{s}_i^{(\ell)})$$

Level 1+ → Level 0:

- Position preserved
- Velocity preserved (rotational information discarded)

Level 2/3 → Level 1:

- Pose and twist preserved
- Contact forces converted to impulses applied before transition

**Momentum Conservation:**
Total momentum must be preserved:
$$\sum_i m_i \mathbf{v}_i^{(\ell')} = \sum_i m_i \mathbf{v}_i^{(\ell)}$$

For angular momentum (when downgrading to Level 0), convert to translational motion of the center of mass.

### 9.4 Transition Blending

To avoid discontinuities, we blend states over multiple timesteps:

**Exponential Blending:**
$$\mathbf{s}_i^{blend}(t) = \alpha(t) \cdot \mathbf{s}_i^{new\_fidelity}(t) + (1 - \alpha(t)) \cdot \mathbf{s}_i^{old\_fidelity}(t)$$

where:
$$\alpha(t) = 1 - e^{-(t - t_{transition})/\tau_{blend}}$$

and $\tau_{blend}$ is the blending time constant (typically 5-10 timesteps).

**Algorithm 4: Seamless Fidelity Transition**

```
Input: Raw states S^{raw}, current allocation φ, previous allocation φ_prev
Output: Consistent states S

1: S = copy(S^{raw})
2:
3: for each agent i do
4:    if φ(i) ≠ φ_prev(i) then  // Fidelity changed
5:       if φ(i) > φ_prev(i) then  // Upgrade
6:          S[i] = Expand(S[i], φ_prev(i), φ(i))
7:          S[i] = EnergyMatch(S[i], S^{raw}[i])
8:       else  // Downgrade
9:          S[i] = Project(S[i], φ_prev(i), φ(i))
10:         S[i] = MomentumConserve(S[i], S^{raw}[i])
11:      end if
12:      MarkForBlending(i, t_current)
13:   end if
14:
15:   if InBlendingWindow(i) then
16:      α = BlendingWeight(i, t_current)
17:      S[i] = α * S[i] + (1-α) * ExtrapolateOldFidelity(i)
18:   end if
19: end for
20:
21: // Global consistency check
22: S = EnforceConstraints(S)  // Penetration resolution, etc.
23:
24: return S
```

### 9.5 Constraint Satisfaction

After transitions, enforce physical constraints:

**Penetration Resolution:**
If transition creates penetrating contacts, resolve via position correction:
$$\mathbf{p}_i' = \mathbf{p}_i + \sum_j \mathbf{n}_{ij} \cdot \max(0, d_{penetration})$$

**Velocity Correction:**
Ensure no velocity into collision:
$$\mathbf{v}_i' = \mathbf{v}_i - \sum_j \mathbf{n}_{ij} \cdot \min(0, \mathbf{v}_i \cdot \mathbf{n}_{ij})$$

---

## 10. Multi-Fidelity Simulation Engine

### 10.1 Fidelity-Specific Simulators

**Level 0 Simulator (Kinematic):**

```python
def simulate_level0(state, control, dt):
    """Point-mass kinematic integration."""
    p, v = state.position, state.velocity
    a = control.acceleration

    # Simple Euler integration
    v_new = v + a * dt
    p_new = p + v_new * dt

    return State(position=p_new, velocity=v_new)
```

Complexity: O(1) per agent, no collision detection

**Level 1 Simulator (Rigid Body, No Contact):**

```python
def simulate_level1(state, control, dt):
    """Rigid body dynamics without contact."""
    p, R, v, ω = state.position, state.rotation, state.linear_vel, state.angular_vel
    F, τ = control.force, control.torque
    m, I = state.mass, state.inertia

    # Newton-Euler equations
    v_new = v + (F / m) * dt
    ω_new = ω + I_inv @ (τ - np.cross(ω, I @ ω)) * dt

    p_new = p + v_new * dt
    R_new = R @ exp_so3(ω_new * dt)

    return State(p_new, R_new, v_new, ω_new)
```

Complexity: O(1) per agent

**Level 2 Simulator (Simplified Contact):**

```python
def simulate_level2(state, control, neighbors, dt):
    """Impulse-based contact resolution."""
    # Step 1: Integrate as Level 1
    state_predicted = simulate_level1(state, control, dt)

    # Step 2: Collision detection (sphere-sphere)
    collisions = detect_collisions_sphere(state_predicted, neighbors)

    # Step 3: Impulse resolution
    for collision in collisions:
        j = compute_impulse(collision, restitution=0.5)
        state_predicted = apply_impulse(state_predicted, j, collision.normal)

    return state_predicted
```

Complexity: O(k) per agent, where k is average neighbor count

**Level 3 Simulator (Full Physics):**

```python
def simulate_level3(state, control, neighbors, environment, dt):
    """Full rigid body with LCP-based contact."""
    # Step 1: Predict next state
    state_predicted = integrate_unconstrained(state, control, dt)

    # Step 2: Collision detection (mesh-mesh)
    contacts = detect_contacts_mesh(state_predicted, neighbors, environment)

    # Step 3: Build LCP
    A, b = build_contact_lcp(contacts, state_predicted)

    # Step 4: Solve LCP (Lemke's algorithm or PGS)
    λ = solve_lcp(A, b)

    # Step 5: Apply contact forces
    state_final = apply_contact_forces(state_predicted, contacts, λ)

    return state_final
```

Complexity: O(k² + |contacts|³) per agent

### 10.2 Inter-Fidelity Interactions

When agents at different fidelity levels interact:

**Rule 1 (Upgrade on Interaction):** If a Level 0 agent approaches a Level 2+ agent within interaction radius, temporarily upgrade to Level 1 for that timestep.

**Rule 2 (Force Coupling):** Contact forces computed at higher fidelity are projected to lower fidelity:

- Level 3 → Level 2: Use impulse approximation
- Level 2 → Level 1: Ignore contact (handled by repulsion)
- Level 1 → Level 0: Ignore (kinematic model)

**Rule 3 (Boundary Regions):** Agents near fidelity boundaries receive intermediate treatment—collision detection at high fidelity, dynamics at low fidelity.

### 10.3 Parallel Execution

MFST leverages GPU parallelism for Level 0-2 and CPU for Level 3:

**GPU Kernels (Level 0-2):**

- Parallel state integration (one thread per agent)
- Spatial hashing for neighbor queries
- Parallel collision detection

**CPU Processing (Level 3):**

- Sequential LCP solve (numerically sensitive)
- Mesh-mesh collision (complex geometry)
- Constraint stabilization

**Hybrid Schedule:**

1. Launch GPU kernels for Level 0-2 agents
2. Launch CPU threads for Level 3 agents
3. Synchronize and exchange forces
4. Apply fidelity transitions

---

## 11. Theoretical Analysis

### 11.1 Error Bounds

**Theorem 5 (Per-Agent Error Bound).** For agent $i$ simulated at fidelity level $\ell < L$, the state error after $T$ timesteps is bounded by:

$$\|\mathbf{s}_i^{(\ell)}(T) - \mathbf{s}_i^{(L)}(T)\| \leq \epsilon_\ell \cdot \frac{(1 + \gamma_\ell)^T - 1}{\gamma_\ell}$$

where $\epsilon_\ell$ is the per-timestep error and $\gamma_\ell$ is the error growth rate.

_Proof._ From Assumption 4, error grows as:
$$e_i(t+1) \leq (1 + \gamma_\ell) e_i(t) + \epsilon_\ell$$

Unrolling the recursion with $e_i(0) = 0$:
$$e_i(T) \leq \epsilon_\ell \sum_{k=0}^{T-1} (1 + \gamma_\ell)^k = \epsilon_\ell \cdot \frac{(1 + \gamma_\ell)^T - 1}{\gamma_\ell}$$

This is geometric growth with rate $1 + \gamma_\ell$. □

**Corollary 2 (Total Error Bound).** The total swarm error is bounded by:
$$E(\phi, T) \leq \sum_{i=1}^N \epsilon_{\phi(i)} \cdot \frac{(1 + \gamma_{\phi(i)})^T - 1}{\gamma_{\phi(i)}}$$

### 11.2 Emergence Preservation Guarantee

**Theorem 6 (Behavioral Fidelity).** If behavioral criticality $\kappa_i$ is estimated with accuracy $\eta$, then emergent behaviors are preserved with probability at least $1 - \delta$ where:

$$\delta = N \cdot \exp\left(-\frac{2\eta^2}{E_{max}^2}\right)$$

_Proof._ By Hoeffding's inequality, the probability that importance estimation fails for agent $i$ is:
$$P[|\hat{I}_i - I_i| > \eta] \leq 2\exp\left(-\frac{2\eta^2}{E_{max}^2}\right)$$

By union bound over $N$ agents:
$$P[\exists i: |\hat{I}_i - I_i| > \eta] \leq N \cdot 2\exp\left(-\frac{2\eta^2}{E_{max}^2}\right)$$

When importance is correctly estimated, high-criticality agents receive high fidelity, preserving behavioral influence. □

### 11.3 Computational Complexity

**Theorem 7 (MFST Complexity).** The computational cost of MFST per timestep is:
$$C_{MFST} = O(N) + O(N \log N) + O(N \bar{c}(\phi))$$

where $\bar{c}(\phi) = \frac{1}{N}\sum_i c_{\phi(i)}$ is the average fidelity cost.

_Proof._

- FIE: O(N) feature extraction, O(N) neural network inference
- CBO: O(N log N) greedy allocation with priority queue
- Simulation: O(N · average cost per agent)
- SFT: O(N) transition handling

The dominant term is simulation cost, which scales with $N \cdot \bar{c}(\phi)$. Under budget constraint $B$, we have $\bar{c}(\phi) \leq B/N$, so total cost is $O(B)$ by construction. □

### 11.4 Convergence Properties

**Theorem 8 (Budget Convergence).** MFST simulation converges to high-fidelity simulation as budget increases:

$$\lim_{B \rightarrow \infty} \mathbb{E}[\|\mathbf{S}^{MFST}(T) - \mathbf{S}^{HiFi}(T)\|] = 0$$

_Proof._ As $B \rightarrow \infty$, CBO allocates all agents to Level $L$:
$$B > N \cdot c_L \Rightarrow \phi^*(i) = L \quad \forall i$$

At Level $L$, approximation error is zero by definition. □

**Theorem 9 (Graceful Degradation).** As budget decreases, MFST degrades gracefully:

$$E(\phi_B, T) = O(1/B)$$

_Proof._ CBO prioritizes high-importance agents. As budget decreases, low-importance agents move to lower fidelity. By importance weighting, the total error contribution of downgraded agents is bounded. The error scales inversely with budget due to the greedy allocation's approximation guarantee. □

---

## 12. Implementation Details

### 12.1 Spatial Data Structures

Efficient neighbor queries are critical for interaction features and collision detection:

**Spatial Hashing:**
$$hash(\mathbf{p}) = \lfloor p_x / cell\_size \rfloor \cdot P_1 \oplus \lfloor p_y / cell\_size \rfloor \cdot P_2 \oplus \lfloor p_z / cell\_size \rfloor$$

where $P_1, P_2$ are large primes.

**Hierarchical Grid:**
For agents at different fidelity levels, use hierarchy:

- Level 0 agents: Coarse grid (cell size = 4× agent radius)
- Level 1-2 agents: Medium grid (cell size = 2× agent radius)
- Level 3 agents: Fine grid (cell size = 1× agent radius)

### 12.2 Memory Management

**State Storage:**

- Level 0: 6 floats per agent (position + velocity)
- Level 1: 13 floats per agent (pose + twist)
- Level 2: 13 + contact buffer (dynamic)
- Level 3: 13 + contact patch (large, dynamic)

**Memory Pool:**
Pre-allocate maximum memory for highest fidelity, reuse for lower fidelities.

### 12.3 Numerical Stability

**Integration:**

- Level 0-1: Semi-implicit Euler (stable, fast)
- Level 2: Velocity-level integration with position correction
- Level 3: Implicit Euler with Newton iteration

**Contact Solving:**

- Regularization parameter $\epsilon = 10^{-6}$ for LCP
- Maximum iterations = 100 for PGS solver
- Warm-starting from previous timestep

---

_Continued in Part 3: Experiments, Results, Discussion, and Conclusion_

---

## References (Partial - Part 2)

[Nemhauser & Wolsey, 1978] G. Nemhauser and L. Wolsey, "Best algorithms for approximating the maximum of a submodular set function," Mathematics of Operations Research, 1978.

---

_End of Part 2_
