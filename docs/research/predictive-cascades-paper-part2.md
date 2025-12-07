# Predictive Failure Cascade Analysis: Part 2 - Methodology

_[Continuation of predictive-cascades-paper-part1.md]_

---

## 5. PFCA Framework Architecture

### 5.1 System Overview

The Predictive Failure Cascade Analysis (PFCA) framework consists of four integrated components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PREDICTIVE FAILURE CASCADE ANALYSIS (PFCA)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   STRUCTURAL    │     │    TEMPORAL     │     │    CAUSAL       │       │
│  │   VULNERABILITY │ ──→ │    PREDICTOR    │ ──→ │   INTERVENTION  │       │
│  │   ENCODER (SVE) │     │    (TPP)        │     │   PLANNER (CIP) │       │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘       │
│           │                       │                       │                 │
│           ▼                       ▼                       ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    EARLY WARNING GENERATOR (EWG)                     │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │   │
│  │  │ Vulnerability │  │ Time-to-      │  │ Intervention  │            │   │
│  │  │ Rankings      │  │ Cascade       │  │ Recommend.    │            │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     CONTINUOUS MONITORING LOOP                       │   │
│  │   Network State → Update Predictions → Alert if Threshold → Act     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Data Flow:**

1. **Input:** Network topology $G$, node states $\{x_i(t)\}$, historical cascades
2. **SVE:** Encodes structural vulnerability into node embeddings
3. **TPP:** Predicts failure timing using temporal point process
4. **CIP:** Identifies optimal intervention targets using causal analysis
5. **EWG:** Generates actionable early warnings

### 5.2 Structural Vulnerability Encoder (SVE)

The SVE learns cascade-relevant node representations through a specialized graph neural network.

#### 5.2.1 Architecture

**Input Features:**
For each node $i$, we construct feature vector $x_i^{(0)}$:

$$x_i^{(0)} = [\underbrace{L_i, C_i, L_i/C_i}_{\text{load/capacity}}, \underbrace{d_i, B_i, k_i}_{\text{centrality}}, \underbrace{\sigma_{hist,i}}_{\text{history}}, \underbrace{x_i^{ext}}_{\text{external}}]$$

where:

- $L_i, C_i$: Current load and capacity
- $d_i$: Degree
- $B_i$: Betweenness centrality
- $k_i$: k-core number
- $\sigma_{hist,i}$: Historical failure frequency
- $x_i^{ext}$: External state features (domain-specific)

**Cascade-Aware Message Passing:**

Standard GNNs aggregate neighbor information uniformly. We design a cascade-aware message function that models load redistribution:

$$m_{i \leftarrow j}^{(l)} = \text{MLP}^{(l)}\left(h_j^{(l)}, e_{ij}, \frac{w_{ij}}{\sum_{k \in \mathcal{N}(j)} w_{jk}}\right)$$

The third argument models load fraction transferred from $j$ to $i$ if $j$ fails.

**Multi-Scale Aggregation:**

Cascades propagate at multiple scales. We aggregate at different neighborhood depths:

$$h_i^{(l+1)} = \text{UPDATE}^{(l)}\left(h_i^{(l)}, \bigoplus_{r=1}^{R} \alpha_r \cdot \text{AGG}^{(l,r)}\left(\{m_{i \leftarrow j}^{(l)} : j \in \mathcal{N}^r(i)\}\right)\right)$$

where $\mathcal{N}^r(i)$ is the $r$-hop neighborhood and $\alpha_r$ are learnable weights.

**Attention Mechanism:**

Not all neighbors are equally important for cascade risk. We use attention:

$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(a^T [W h_i \| W h_j \| e_{ij}]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(a^T [W h_i \| W h_k \| e_{ik}]\right)\right)}$$

Attention weights are interpretable—high $\alpha_{ij}$ means $j$ is a cascade threat to $i$.

#### 5.2.2 Vulnerability Prediction Head

The final node embeddings $h_i^{(L)}$ feed into a vulnerability prediction head:

$$r_i = \sigma\left(\text{MLP}_{vuln}(h_i^{(L)})\right) \in [0, 1]$$

**Training Objective:**
We train on historical cascades with binary cross-entropy:

$$\mathcal{L}_{SVE} = -\sum_{i \in V} \left[\mathbb{1}[i \in \mathcal{C}_{init}] \log r_i + \mathbb{1}[i \notin \mathcal{C}_{init}] \log(1 - r_i)\right]$$

where $\mathcal{C}_{init}$ is the set of cascade initiation nodes.

#### 5.2.3 Graph-Level Vulnerability

Beyond node scores, we predict system-wide cascade risk:

$$R_{global} = \text{ReadOut}\left(\{h_i^{(L)}\}_{i \in V}\right) = \sigma\left(\text{MLP}_{global}\left(\frac{1}{N}\sum_i h_i^{(L)} \| \max_i h_i^{(L)}\right)\right)$$

### 5.3 Temporal Point Process Predictor (TPP)

While SVE identifies vulnerable nodes, TPP predicts when failures will occur.

#### 5.3.1 Neural Marked Temporal Point Process

We model the failure sequence as a marked temporal point process:

**Event Representation:**
Each failure is an event $(t_k, v_k, m_k)$ where:

- $t_k$: Failure time
- $v_k$: Failed node ID (mark)
- $m_k$: Failure magnitude (optional)

**Conditional Intensity:**
The intensity for node $i$ to fail at time $t$ given history $\mathcal{H}_t$:

$$\lambda_i(t | \mathcal{H}_t) = f_\theta\left(h_i, \text{TemporalEnc}(\mathcal{H}_t)\right)$$

**History Encoding:**

We encode the event history using a Transformer:

$$\text{TemporalEnc}(\mathcal{H}_t) = \text{Transformer}\left(\{(e_{t_k}, e_{v_k})\}_{t_k < t}\right)$$

where $e_{t_k}$ is positional encoding and $e_{v_k}$ is node embedding from SVE.

**Self-Exciting Dynamics:**

Failures trigger subsequent failures (Hawkes-like):

$$\lambda_i(t) = \underbrace{\mu_i}_{\text{base rate}} + \underbrace{\sum_{t_k < t} \phi_\theta(i, v_k, t - t_k)}_{\text{triggered component}}$$

The triggering kernel $\phi_\theta$ is parameterized by a neural network that takes:

- Target node $i$
- Previous failed node $v_k$
- Time elapsed $t - t_k$

$$\phi_\theta(i, j, \Delta t) = \text{MLP}\left(h_i \| h_j \| A_{ij} \| \exp(-\Delta t / \tau)\right)$$

#### 5.3.2 Cascade Sequence Prediction

Given current state, we predict future failure sequence:

**Sampling Algorithm:**

```python
def sample_cascade(model, G, x, max_time, max_failures):
    """Sample cascade trajectory from TPP model."""
    history = []
    t = 0

    while t < max_time and len(history) < max_failures:
        # Compute intensities for all nodes
        intensities = model.compute_intensities(G, x, history, t)

        # Total intensity
        lambda_total = sum(intensities)

        # Sample inter-event time (exponential)
        dt = -log(random()) / lambda_total
        t += dt

        if t >= max_time:
            break

        # Sample which node fails (categorical)
        probs = intensities / lambda_total
        failed_node = categorical(probs)

        history.append((t, failed_node))

        # Update node states (load redistribution)
        x = update_states(x, G, failed_node)

    return history
```

**Monte Carlo Cascade Size Estimation:**

$$\hat{S} = \frac{1}{M} \sum_{m=1}^{M} |\text{sample\_cascade}^{(m)}(\cdot)|$$

#### 5.3.3 Time-to-Cascade Prediction

We predict time until cascade reaches critical size:

$$\hat{\tau}_{cascade} = \mathbb{E}[t : S(t) > S_{critical} | \mathcal{H}_t]$$

Estimated via Monte Carlo:

$$\hat{\tau}_{cascade} \approx \text{median}\left(\{t^{(m)}_{S > S_{crit}}\}_{m=1}^{M}\right)$$

**Training Objective:**

Negative log-likelihood of observed cascades:

$$\mathcal{L}_{TPP} = -\sum_{\mathcal{C}} \left[\sum_{(t_k, v_k) \in \mathcal{C}} \log \lambda_{v_k}(t_k) - \int_0^{T} \sum_i \lambda_i(t) dt\right]$$

The integral is approximated via Monte Carlo or quadrature.

### 5.4 Causal Intervention Planner (CIP)

TPP predicts what will happen; CIP determines how to prevent it.

#### 5.4.1 Structural Causal Model

We formalize the cascade generation process as an SCM:

**Variables:**

- $X_i$: Node state (load, health)
- $\sigma_i$: Failure indicator
- $\mathcal{C}$: Cascade (set of failures)
- $I$: Intervention set

**Structural Equations:**
$$X_i = f_{load}(X_{Pa(i)}, U_i^{load})$$
$$\sigma_i = \mathbb{1}[X_i > C_i \wedge U_i^{fail} < q_i(X)]$$
$$\mathcal{C} = g(\{\sigma_i\}_{i \in V})$$

**Intervention:**
Reinforcing node $i$ sets:
$$do(C_i = \gamma \cdot C_i) \quad \text{or} \quad do(\sigma_i = 0)$$

#### 5.4.2 Counterfactual Cascade Simulation

To evaluate intervention $I$, we simulate the counterfactual cascade:

```python
def counterfactual_cascade(model, G, x, I, intervention_type='reinforce'):
    """Simulate cascade under intervention on nodes I."""

    # Apply intervention
    if intervention_type == 'reinforce':
        for i in I:
            G.capacity[i] *= gamma  # Increase capacity
    elif intervention_type == 'isolate':
        for i in I:
            G.remove_node(i)  # Remove from network

    # Simulate cascade under intervention
    cascade_I = model.simulate_cascade(G, x)

    return cascade_I
```

**Causal Effect:**

$$\text{CE}(I) = \mathbb{E}[S | \text{no intervention}] - \mathbb{E}[S | do(\text{reinforce } I)]$$

Estimated via Monte Carlo counterfactual simulations.

#### 5.4.3 Optimal Intervention Selection

We seek intervention set $I^*$ that maximizes cascade reduction:

**Optimization Problem:**
$$I^* = \arg\max_{I \subseteq V, |I| \leq k} \text{CE}(I)$$

subject to budget constraint $|I| \leq k$.

**Greedy Approximation:**

Cascade reduction is approximately submodular—greedy achieves $(1 - 1/e)$ approximation:

```python
def greedy_intervention_selection(model, G, x, budget_k):
    """Greedy selection of intervention set."""
    I = set()

    for _ in range(budget_k):
        best_node = None
        best_gain = 0

        for v in V - I:
            # Evaluate marginal gain of adding v
            gain = cascade_reduction(I | {v}) - cascade_reduction(I)

            if gain > best_gain:
                best_gain = gain
                best_node = v

        I.add(best_node)

    return I
```

**Acceleration with GNN:**

Rather than simulating each candidate, we train a GNN to predict intervention effectiveness:

$$\hat{\text{CE}}(v | I) = \text{MLP}_{interv}\left(h_v \| \text{ReadOut}(\{h_i\}_{i \in I})\right)$$

This provides 100× speedup over simulation-based selection.

#### 5.4.4 Intervention Types

PFCA supports multiple intervention modalities:

| Intervention   | Action                                 | Cost Model                 |
| -------------- | -------------------------------------- | -------------------------- |
| **Reinforce**  | Increase capacity $C_i \to \gamma C_i$ | $\propto (\gamma - 1) C_i$ |
| **Shed Load**  | Decrease load $L_i \to \alpha L_i$     | $\propto (1 - \alpha) L_i$ |
| **Isolate**    | Remove from network                    | $\propto d_i$ (degree)     |
| **Redundancy** | Add backup node                        | $\propto C_i$              |
| **Reroute**    | Modify edge weights                    | $\propto \Delta W$         |

The CIP optimizes intervention type jointly with target selection.

### 5.5 Early Warning Generator (EWG)

The EWG synthesizes outputs from SVE, TPP, and CIP into actionable alerts.

#### 5.5.1 Alert Levels

Alerts are tiered by severity:

| Level      | Condition                   | Lead Time | Action                     |
| ---------- | --------------------------- | --------- | -------------------------- |
| **GREEN**  | $R_{global} < 0.3$          | N/A       | Monitor                    |
| **YELLOW** | $0.3 \leq R_{global} < 0.6$ | >24h      | Prepare interventions      |
| **ORANGE** | $0.6 \leq R_{global} < 0.8$ | 4-24h     | Execute preventive actions |
| **RED**    | $R_{global} \geq 0.8$       | <4h       | Emergency intervention     |

#### 5.5.2 Alert Generation

```python
class EarlyWarningGenerator:
    def generate_alert(self, sve_output, tpp_output, cip_output):
        """Generate early warning alert."""

        alert = Alert()

        # Global risk level
        alert.risk_level = self.classify_risk(sve_output.R_global)

        # Top vulnerable nodes
        alert.vulnerable_nodes = sve_output.top_k_vulnerable(k=10)

        # Predicted cascade timeline
        alert.time_to_cascade = tpp_output.expected_time_to_critical
        alert.predicted_cascade_size = tpp_output.expected_size

        # Recommended interventions
        alert.interventions = cip_output.optimal_interventions
        alert.expected_reduction = cip_output.predicted_effectiveness

        # Confidence intervals
        alert.confidence = self.compute_confidence(sve_output, tpp_output)

        return alert
```

#### 5.5.3 Adaptive Thresholds

Alert thresholds adapt based on:

1. **Historical false positive rate:** Increase threshold if too many false alarms
2. **Intervention cost:** Higher cost → more conservative alerts
3. **Cascade severity:** More severe → more sensitive alerts

$$\theta_{alert} = \theta_0 + \lambda_{FP} \cdot \text{FPR}_{recent} - \lambda_{severity} \cdot \mathbb{E}[S | \text{cascade}]$$

### 5.6 Training Procedure

#### 5.6.1 Data Collection

**Historical Cascades:**
Collect from real systems (anonymized power grid, financial networks) or simulated.

**Synthetic Data Generation:**

```python
def generate_cascade_data(G, cascade_model, n_samples):
    """Generate synthetic cascade dataset."""
    cascades = []

    for _ in range(n_samples):
        # Random initial failure
        init_node = random_choice(V, p=vulnerability_prior)

        # Simulate cascade
        cascade = cascade_model.simulate(G, init_node)

        # Record (network state before, cascade)
        cascades.append({
            'network_state': G.get_state(),
            'init_node': init_node,
            'cascade': cascade
        })

    return cascades
```

#### 5.6.2 Multi-Task Learning

We train SVE, TPP, and CIP jointly:

$$\mathcal{L}_{total} = \underbrace{\mathcal{L}_{SVE}}_{\text{vulnerability}} + \beta_1 \underbrace{\mathcal{L}_{TPP}}_{\text{timing}} + \beta_2 \underbrace{\mathcal{L}_{CIP}}_{\text{intervention}}$$

**Intervention Loss:**

$$\mathcal{L}_{CIP} = \sum_I \left(\hat{CE}(I) - CE_{sim}(I)\right)^2$$

where $CE_{sim}$ is the simulated ground truth causal effect.

#### 5.6.3 Training Algorithm

```python
def train_pfca(model, train_data, epochs, lr):
    """Train PFCA model end-to-end."""
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in train_data:
            # Forward pass
            sve_out = model.sve(batch.G, batch.x)
            tpp_out = model.tpp(batch.G, batch.x, batch.cascade)
            cip_out = model.cip(batch.G, batch.x, sve_out)

            # Compute losses
            loss_sve = cross_entropy(sve_out.r, batch.init_mask)
            loss_tpp = nll_tpp(tpp_out.intensities, batch.cascade)
            loss_cip = mse(cip_out.predicted_ce, batch.true_ce)

            loss_total = loss_sve + beta1 * loss_tpp + beta2 * loss_cip

            # Backward pass
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        # Evaluate
        val_metrics = evaluate(model, val_data)
        log(epoch, val_metrics)
```

### 5.7 Computational Complexity

| Component       | Time Complexity        | Space Complexity |
| --------------- | ---------------------- | ---------------- | ---------------- | ------------------- |
| SVE (L layers)  | $O(L \cdot             | E                | \cdot d)$        | $O(N \cdot d)$      |
| TPP (M samples) | $O(M \cdot T \cdot N)$ | $O(N +           | \mathcal{H}      | )$                  |
| CIP (k budget)  | $O(k \cdot N \cdot M)$ | $O(N \cdot k)$   |
| Total           | $O(L                   | E                | d + kNM + MT N)$ | $O(Nd + N \cdot k)$ |

For typical values ($N=10^4$, $|E|=10^5$, $L=4$, $d=64$, $M=100$, $k=10$):

- Training: ~1 hour on single GPU
- Inference: ~100ms per network state update

---

_[Continued in Part 3: Experiments, Results, Discussion, Conclusion]_
