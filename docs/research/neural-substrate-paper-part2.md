# Neural Substrate Mapping: Part 2 - Methodology

_[Continuation of neural-substrate-paper-part1.md]_

---

## 4. Neural Substrate Mapping Framework

### 4.1 Framework Overview

Neural Substrate Mapping (NSM) provides a principled approach to designing brain-inspired agent architectures. The framework operates at three levels:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NEURAL SUBSTRATE MAPPING (NSM)                       │
├─────────────────────────────────────────────────────────────────────────┤
│  Level 1: FUNCTION MAPPING                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │  Cognitive  │ →  │   Brain     │ →  │   Spiking   │                 │
│  │  Function   │    │   Region    │    │   Circuit   │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Level 2: ARCHITECTURE INSTANTIATION                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        Multi-Region Network                        │ │
│  │  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     │ │
│  │  │   PFC   │ ←→  │  Hipp.  │     │  Basal  │ ←→  │  Cere.  │     │ │
│  │  │ (plan)  │     │ (memory)│     │ Ganglia │     │ (motor) │     │ │
│  │  └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘     │ │
│  │       └───────────────┴───────────────┴───────────────┘           │ │
│  │                    Inter-Region Connectivity                       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│  Level 3: LEARNING DYNAMICS                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │    STDP     │    │  Three-     │    │   Sleep     │                 │
│  │  (local)    │ +  │  Factor     │ +  │  Consol.    │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Function-to-Region Mapping

We define a mapping $\mathcal{M}: \mathcal{F} \to \mathcal{R}$ from cognitive functions to brain regions:

| Cognitive Function  | Brain Region      | Computational Primitive   |
| ------------------- | ----------------- | ------------------------- |
| Working memory      | Prefrontal Cortex | Attractor dynamics        |
| Planning/sequencing | Prefrontal Cortex | Sequence generation       |
| Episodic memory     | Hippocampus       | Pattern storage/retrieval |
| Spatial navigation  | Hippocampus       | Place cells, grid cells   |
| Action selection    | Basal Ganglia     | Winner-take-all           |
| Habit formation     | Basal Ganglia     | Stimulus-response         |
| Motor prediction    | Cerebellum        | Forward model             |
| Error correction    | Cerebellum        | Supervised learning       |
| Sensory processing  | Thalamus          | Gating, routing           |
| Feature extraction  | Sensory cortex    | Hierarchical coding       |

### 4.3 Multi-Region Architecture

#### 4.3.1 Prefrontal Cortex (PFC) Module

The PFC module maintains task-relevant information and generates action plans through attractor dynamics.

**Architecture:**

- Excitatory neurons: $N_E = 800$ (80%)
- Inhibitory neurons: $N_I = 200$ (20%)
- Recurrent connectivity: $p_{rec} = 0.15$
- External inputs: Sensory cortex, hippocampus
- Outputs: Basal ganglia, motor cortex

**Connectivity:**
$$w_{ij}^{EE} \sim \mathcal{N}(J_{EE}/\sqrt{N_E}, \sigma^2)$$
$$w_{ij}^{EI}, w_{ij}^{IE}, w_{ij}^{II} \text{ scaled for E/I balance}$$

**Attractor Dynamics:**
Working memory is implemented via bump attractors. The network has multiple stable states corresponding to remembered items:

$$\tau \frac{d\mathbf{r}}{dt} = -\mathbf{r} + f(\mathbf{W}\mathbf{r} + \mathbf{I}_{ext})$$

where $f$ is a threshold-linear activation. Attractors are carved by Hebbian learning:

$$\Delta w_{ij} = \eta (r_i - \bar{r}_i)(r_j - \bar{r}_j)$$

**Persistent Activity:**
During delays, recurrent excitation maintains activity patterns without external input. The number of items that can be maintained is:

$$N_{items} \approx \frac{N_E}{N_{neurons/item}} \approx \frac{800}{200} = 4$$

consistent with working memory capacity limits (Miller, 1956; Cowan, 2001).

#### 4.3.2 Hippocampus Module

The hippocampus module enables rapid storage and retrieval of episodic memories.

**Architecture:**

- Dentate Gyrus (DG): $N_{DG} = 1000$ (pattern separation)
- CA3: $N_{CA3} = 500$ (autoassociative memory)
- CA1: $N_{CA1} = 500$ (output encoding)
- Entorhinal Cortex (EC): $N_{EC} = 300$ (input/output interface)

**Tri-Synaptic Pathway:**

```
EC → DG → CA3 → CA1 → EC
 ↓         ↑
 └─────────┘ (direct path)
```

**Pattern Separation (DG):**
Sparse coding with ~1-2% activation:

$$a_i^{DG} = \text{top-}k\left(\sum_j w_{ij}^{EC \to DG} \cdot s_j^{EC}\right)$$

where $k = 0.01 \cdot N_{DG}$.

**Autoassociative Memory (CA3):**
Recurrent connectivity with STDP enables pattern completion:

$$\Delta w_{ij}^{CA3} = \eta_{CA3} \cdot (s_i^{post} - \bar{s}) \cdot s_j^{pre}$$

Capacity (Hopfield-like): $C_{CA3} \approx 0.14 \cdot N_{CA3} \approx 70$ patterns.

**Memory Retrieval:**
Given partial cue $\mathbf{x}_{partial}$, CA3 pattern-completes to full memory $\mathbf{x}_{full}$ through attractor dynamics (~50-100ms).

**Place Cells and Grid Cells:**
For spatial tasks, we implement:

- **Place cells** in CA1: Fire when agent at specific location
- **Grid cells** in EC: Fire at regular hexagonal grid locations

$$\text{PlaceCell}_i(x,y) = \exp\left(-\frac{(x-x_i)^2 + (y-y_i)^2}{2\sigma_{place}^2}\right)$$

$$\text{GridCell}_i(x,y) = \sum_{k=1}^{3} \cos\left(\mathbf{k}_i^{(k)} \cdot (x,y)\right)$$

where $\mathbf{k}_i^{(k)}$ are wave vectors at 60° angles.

#### 4.3.3 Basal Ganglia Module

The basal ganglia implements action selection through competition and reinforcement learning.

**Architecture:**

- Striatum: $N_{striatum} = 600$ (D1: 300, D2: 300)
- Globus Pallidus external (GPe): $N_{GPe} = 100$
- Globus Pallidus internal (GPi): $N_{GPi} = 100$
- Subthalamic Nucleus (STN): $N_{STN} = 100$
- Substantia Nigra pars compacta (SNc): Dopamine source

**Direct and Indirect Pathways:**

```
       Cortex
          ↓
      Striatum
      ↙      ↘
    D1         D2
    ↓           ↓
   GPi  ←───  GPe
    ↓           ↓
 Thalamus ←── STN
    ↓
  Cortex
```

**Direct Pathway (Go):**
Cortex → D1 Striatum → GPi (inhibit) → Thalamus (disinhibit) → Action

**Indirect Pathway (NoGo):**
Cortex → D2 Striatum → GPe (inhibit) → STN (disinhibit) → GPi (excite) → Thalamus (inhibit) → No Action

**Dopamine Modulation:**
$$\Delta w^{D1} = \alpha \cdot \delta \cdot e_{ij}$$
$$\Delta w^{D2} = -\alpha \cdot \delta \cdot e_{ij}$$

where $\delta = r - V(s)$ is the reward prediction error (TD error), and $e_{ij}$ is the eligibility trace.

**Action Selection:**
Winner-take-all in GPi selects single action:
$$a^* = \arg\max_a \left(-\text{GPi}_a\right)$$

#### 4.3.4 Cerebellum Module

The cerebellum predicts motor outcomes and corrects errors.

**Architecture:**

- Granule Cells (GC): $N_{GC} = 2000$ (sparse coding)
- Purkinje Cells (PC): $N_{PC} = 200$
- Deep Cerebellar Nuclei (DCN): $N_{DCN} = 50$
- Inferior Olive (IO): Error signal source

**Microcircuit:**

```
Mossy Fibers → Granule Cells → Purkinje Cells → DCN → Motor Output
                      ↑
              Climbing Fibers (IO)
                 (Error Signal)
```

**Expansion Coding (Granule Cells):**
Mossy fiber inputs ($d_{MF} \approx 4$) are expanded to high-dimensional sparse code:

$$\mathbf{g} = \sigma\left(\mathbf{W}^{GC}\mathbf{m} - \theta_{GC}\right)$$

with sparsity ~2% (Marr-Albus theory).

**Supervised Learning (Purkinje Cells):**
Climbing fibers convey error signals that modify parallel fiber → Purkinje cell synapses:

$$\Delta w_{ij}^{PF \to PC} = -\eta_{PC} \cdot g_j \cdot CF_i$$

where $CF_i$ is the climbing fiber input (0 or 1).

**Forward Model:**
The cerebellum learns to predict sensory consequences of motor commands:

$$\hat{s}_{t+1} = f_{CB}(s_t, a_t)$$

Prediction errors drive learning and are used for online correction.

### 4.4 Inter-Region Connectivity

Regions are connected following known neuroanatomy:

**Connectivity Matrix $\Omega$:**

|           | PFC | Hipp | BG  | Cereb | Thal | Cortex |
| --------- | --- | ---- | --- | ----- | ---- | ------ |
| **PFC**   | R   | B    | →   | ←     | B    | B      |
| **Hipp**  | B   | R    | →   | ←     | ←    | B      |
| **BG**    | ←   | ←    | R   | ←     | →    | ←      |
| **Cereb** | →   | →    | →   | R     | →    | →      |
| **Thal**  | B   | →    | ←   | ←     | R    | B      |

Legend: R = recurrent, B = bidirectional, → = feedforward, ← = feedback

**Connection Strengths:**
Inter-region weights are initialized weaker than intra-region:
$$w_{inter} = 0.3 \cdot w_{intra}$$

Delays model axonal conduction (~5-20ms for cortical connections).

### 4.5 Biologically Plausible Learning

#### 4.5.1 Three-Factor Hebbian Learning

Standard STDP lacks credit assignment for delayed rewards. We use three-factor rules:

$$\Delta w_{ij} = \eta \cdot M(t) \cdot \text{STDP}(\Delta t) \cdot e_{ij}(t)$$

**Factor 1: STDP**

$$
\text{STDP}(\Delta t) = \begin{cases}
A_+ e^{-\Delta t / \tau_+} & \text{if } \Delta t > 0 \\
-A_- e^{\Delta t / \tau_-} & \text{if } \Delta t < 0
\end{cases}
$$

**Factor 2: Eligibility Trace**
$$\frac{de_{ij}}{dt} = -\frac{e_{ij}}{\tau_e} + \text{STDP}(\Delta t) \cdot \delta(t - t_{spike})$$

Traces decay with $\tau_e \approx 1s$, bridging the gap between spikes and rewards.

**Factor 3: Neuromodulation**
$$M(t) = M_{DA}(t) + M_{ACh}(t) + M_{NE}(t)$$

- **Dopamine ($M_{DA}$):** Reward prediction error, gates plasticity in striatum
- **Acetylcholine ($M_{ACh}$):** Attention, enhances learning in cortex
- **Norepinephrine ($M_{NE}$):** Arousal, global gain modulation

#### 4.5.2 Region-Specific Learning Rules

**PFC Learning:**
$$\Delta w_{ij}^{PFC} = \eta_{PFC} \cdot M_{DA} \cdot e_{ij} + \eta_{slow} \cdot \text{STDP}(\Delta t)$$

Combines fast reward-modulated learning with slow Hebbian consolidation.

**Hippocampus Learning:**
$$\Delta w_{ij}^{Hipp} = \eta_{Hipp} \cdot (s_i - \bar{s})(s_j - \bar{s}) + \eta_{replay} \cdot \text{STDP}_{replay}$$

Pure Hebbian for rapid one-shot learning, enhanced by offline replay.

**Basal Ganglia Learning:**
$$\Delta w_{ij}^{D1} = \alpha \cdot [\delta]^+ \cdot e_{ij}$$
$$\Delta w_{ij}^{D2} = \alpha \cdot [\delta]^- \cdot e_{ij}$$

D1/D2 segregation implements actor-critic with positive/negative prediction errors.

**Cerebellum Learning:**
$$\Delta w_{ij}^{PC} = -\eta_{PC} \cdot g_j \cdot (CF_i - \bar{CF}_i)$$

Supervised learning with climbing fiber error signals.

#### 4.5.3 Sleep Consolidation

Memory consolidation occurs during simulated "sleep" phases:

**Slow-Wave Sleep (SWS):**

- Hippocampal sharp-wave ripples trigger memory replay
- Cortical slow oscillations modulate plasticity
- Memories transfer from hippocampus to cortex

$$\text{Replay: } \mathbf{h}_t^{replay} \sim p(\mathbf{h} | \text{recent memories})$$

**REM Sleep:**

- Random activation patterns
- Synaptic homeostasis (downscaling)
- Creative recombination

$$w_{ij} \leftarrow (1 - \lambda) \cdot w_{ij}$$

**Consolidation Algorithm:**

```
procedure SLEEP_CONSOLIDATION(duration):
    for t in range(duration):
        if SWS_phase(t):
            memory = sample_hippocampus()
            replay(memory)
            update_cortex_weights(memory)
        elif REM_phase(t):
            generate_random_activity()
            homeostatic_scaling()
```

### 4.6 Neuromorphic Implementation

#### 4.6.1 Hardware Mapping

NSM is designed for efficient deployment on neuromorphic hardware:

**Intel Loihi 2 Mapping:**

- 1 neuron core = 128 neurons, 128K synapses
- PFC: 8 cores (1024 neurons)
- Hippocampus: 18 cores (2300 neurons)
- Basal Ganglia: 8 cores (1000 neurons)
- Cerebellum: 18 cores (2250 neurons)
- Total: ~52 cores

**Spike Routing:**
Loihi's spike routing network handles inter-region communication with programmable delays.

**On-Chip Learning:**
Three-factor learning implemented using:

- Trace variables (eligibility)
- Reward signals (neuromodulation)
- Local STDP (spike timing)

#### 4.6.2 Energy Efficiency

**Theoretical Analysis:**
Energy per synaptic operation on Loihi 2: ~10 pJ
Energy per MAC on GPU (A100): ~100 pJ

For our architecture:

- Synaptic operations per timestep: ~$N_{syn} \cdot \bar{r} \cdot \Delta t$ ≈ 100K
- Energy per timestep: ~1 μJ
- Compare to ANN equivalent: ~100 μJ

**Expected Improvement:** 10-100× energy reduction on neuromorphic hardware.

### 4.7 Theoretical Analysis

#### 4.7.1 Theorem 1: Convergence of Three-Factor Learning

**Theorem 1 (Three-Factor Learning Convergence):**
_Under three-factor learning with eligibility traces, the expected weight update converges to the policy gradient:_

$$\mathbb{E}[\Delta w_{ij}] = \eta \cdot \frac{\partial J(\theta)}{\partial w_{ij}} + O(\epsilon)$$

_where $\epsilon$ depends on trace decay rate $\tau_e$ and reward delay distribution._

**Proof Sketch:**

1. **Eligibility Trace as Credit Assignment:**
   The eligibility trace $e_{ij}(t)$ maintains a decaying memory of STDP events:
   $$e_{ij}(t) = \int_0^t \text{STDP}(\Delta t') e^{-(t-t')/\tau_e} dt'$$

2. **Expectation under Reward:**
   Taking expectation over trajectories:
   $$\mathbb{E}[\Delta w_{ij}] = \mathbb{E}\left[\sum_t M(t) \cdot e_{ij}(t)\right]$$

3. **Connection to Policy Gradient:**
   The STDP rule approximates $\frac{\partial \log \pi(a|s)}{\partial w_{ij}}$ for spiking policies:
   $$\text{STDP}(\Delta t) \propto \frac{\partial}{\partial w_{ij}} \log P(s_i^{post} | s_j^{pre}, w_{ij})$$

4. **Policy Gradient Theorem:**
   Combining with reward modulation:
   $$\mathbb{E}[\Delta w_{ij}] \propto \mathbb{E}\left[\sum_t r_t \cdot \sum_{t'<t} e^{-(t-t')/\tau_e} \frac{\partial \log \pi}{\partial w_{ij}}\right]$$
   $$= \mathbb{E}\left[\sum_t r_t \cdot \frac{\partial \log \pi}{\partial w_{ij}} \cdot \text{discount}\right] + O(\epsilon)$$

The error $\epsilon$ decreases with longer trace decay $\tau_e$ but introduces variance. ∎

#### 4.7.2 Theorem 2: Hippocampal Memory Capacity

**Theorem 2 (Pattern Storage Capacity):**
_The hippocampal CA3 network with $N$ neurons and sparsity $a$ can store_

$$C = \frac{N \cdot a(1-a)}{2 \ln(1/a)}$$

_patterns with probability of correct retrieval $\geq 1 - \delta$._

**Proof Sketch:**

Following Amit et al. (1987) for sparse Hopfield networks:

1. **Signal-to-Noise Ratio:**
   For pattern $\mu$, the input to neuron $i$ when pattern $\mu$ is presented:
   $$h_i^\mu = \sum_j w_{ij} \xi_j^\mu = \underbrace{(a-a^2) N}_{\text{signal}} + \underbrace{\sum_{\nu \neq \mu} m^\nu \cdot O(a^2 N)}_{\text{noise}}$$

2. **Capacity Scaling:**
   For successful retrieval, signal > noise threshold:
   $$C \cdot a^2 N < a(1-a) N / \sqrt{2\ln(1/\delta)}$$

3. **Solving for C:**
   $$C < \frac{(1-a)}{a \cdot \sqrt{2\ln(1/\delta)}} \approx \frac{N \cdot a(1-a)}{2\ln(1/a)}$$

For our parameters ($N_{CA3} = 500$, $a = 0.02$):
$$C \approx \frac{500 \cdot 0.02 \cdot 0.98}{2 \ln(50)} \approx \frac{9.8}{7.8} \approx 63 \text{ patterns}$$ ∎

#### 4.7.3 Theorem 3: Cerebellar Forward Model Convergence

**Theorem 3 (Forward Model Learning):**
_The cerebellar forward model with expansion coding and supervised learning converges to the true dynamics with error:_

$$\|\hat{f}_{CB} - f^*\|_2 \leq O\left(\frac{1}{\sqrt{N_{GC}}} + \epsilon_{IO}\right)$$

_where $N_{GC}$ is the number of granule cells and $\epsilon_{IO}$ is the error signal noise._

**Proof Sketch:**

1. **Universal Approximation:**
   With random expansion to $N_{GC}$ dimensions and sparsity $a$, any smooth function can be approximated (Rahimi & Recht, 2007):
   $$\|\hat{f} - f^*\|_2 \leq O(1/\sqrt{N_{GC}})$$

2. **Gradient Descent Convergence:**
   Climbing fiber signals provide gradient:
   $$CF \propto \nabla_{PC} \|s_{t+1} - \hat{s}_{t+1}\|^2$$
3. **Convergence Rate:**
   With step size $\eta_{PC}$, loss decreases as:
   $$\mathcal{L}_{t+1} \leq \mathcal{L}_t - \eta_{PC} \|\nabla \mathcal{L}\|^2 + O(\eta_{PC}^2)$$

4. **Noise Floor:**
   Error signal noise $\epsilon_{IO}$ sets minimum achievable error. ∎

---

_[Continued in Part 3: Experiments, Results, Discussion, Conclusion]_
