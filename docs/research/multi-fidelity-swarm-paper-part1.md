# Multi-Fidelity Digital Twins for Large-Scale Swarm Robotics: Adaptive Fidelity Allocation Under Computational Constraints

## IEEE Transactions on Robotics - Submission Draft

**Part 1 of 3: Abstract, Introduction, Related Work, and Problem Formulation**

---

## Abstract

Simulating large-scale swarm robotics systems presents a fundamental computational challenge: high-fidelity physics simulation provides accurate predictions but scales poorly with swarm size, while low-fidelity approximations sacrifice critical emergent behaviors. We present **Multi-Fidelity Swarm Twins (MFST)**, a framework that dynamically allocates computational resources across heterogeneous fidelity levels based on importance-weighted sampling and predictive uncertainty quantification. MFST maintains a hierarchy of simulation models—from coarse particle-based approximations to full rigid-body dynamics with contact mechanics—and automatically routes each agent to the appropriate fidelity level based on its current behavioral criticality, interaction density, and task relevance. Our approach introduces three key innovations: (1) a **Fidelity Importance Estimator** that predicts which agents require high-fidelity simulation, (2) a **Computational Budget Optimizer** that allocates fixed compute across agents to minimize total prediction error, and (3) a **Seamless Fidelity Transition** mechanism that maintains physical consistency during fidelity changes. Extensive experiments on swarm foraging (1,000 agents), formation control (500 robots), and collective construction (200 manipulators) demonstrate that MFST achieves 94.7% of full high-fidelity accuracy while reducing computational cost by 23.8×. Real-time simulation of 10,000-agent swarms becomes feasible on commodity hardware, enabling rapid policy iteration and hardware-in-the-loop testing. We provide theoretical guarantees on approximation error bounds and prove that MFST converges to high-fidelity simulation as computational budget increases.

**Keywords:** swarm robotics, digital twins, multi-fidelity simulation, computational resource allocation, physics simulation

---

## 1. Introduction

### 1.1 The Swarm Simulation Bottleneck

Swarm robotics has emerged as a transformative paradigm for applications requiring robustness, scalability, and emergent intelligence—from warehouse automation and precision agriculture to search-and-rescue and space exploration. The fundamental premise is compelling: many simple robots, coordinated through local interactions, can accomplish complex tasks that exceed the capabilities of individual sophisticated machines. However, developing and validating swarm behaviors faces a critical bottleneck: **simulation scalability**.

Consider a swarm foraging scenario with 1,000 autonomous agents. High-fidelity simulation—including rigid-body dynamics, contact mechanics, sensor noise, actuator dynamics, and environmental interactions—requires approximately 10ms per timestep per agent on modern hardware. For real-time factor-1 simulation at 100Hz, this demands 10,000 agent-timesteps per 10ms wall-clock time, requiring 1,000× real-time speedup that exceeds current computational capabilities. The situation worsens for contact-rich manipulation or aerodynamic swarms (drones) where per-agent costs increase by 10-100×.

Practitioners face an uncomfortable tradeoff:

1. **Reduce fidelity globally**: Use simplified dynamics (point-mass, kinematic) for all agents, losing critical emergent behaviors that depend on physical interactions
2. **Reduce swarm size**: Simulate smaller swarms, sacrificing the very scalability that makes swarm robotics valuable
3. **Reduce time horizon**: Run shorter simulations, missing long-term behavioral patterns and rare failure modes
4. **Increase hardware**: Deploy expensive GPU clusters, making rapid iteration impractical for academic labs and startups

All options compromise the fundamental goal: understanding how large-scale swarms behave in realistic conditions.

### 1.2 The Multi-Fidelity Insight

Our key insight is that **not all agents require high-fidelity simulation at all times**. In a foraging swarm, an agent in open space following a gradient requires only kinematic simulation, while an agent negotiating a narrow passage with other robots requires full contact dynamics. In formation control, agents at steady-state need minimal computation, while agents executing maneuvers near obstacles require detailed physics. This heterogeneity creates an opportunity: allocate computational resources where they matter most.

This insight connects to two established principles:

1. **Importance Sampling**: Sample computational effort in proportion to contribution to overall accuracy
2. **Multi-Fidelity Modeling**: Combine cheap low-fidelity models with expensive high-fidelity models to achieve accuracy at reduced cost

We synthesize these principles into a unified framework for swarm digital twins that dynamically balances fidelity across agents based on their behavioral context.

### 1.3 Contributions

This paper presents Multi-Fidelity Swarm Twins (MFST), a framework that enables efficient simulation of large-scale swarms through adaptive fidelity allocation. Our contributions include:

1. **Fidelity Importance Estimator (FIE)**: A learned model that predicts which agents require high-fidelity simulation based on local state features, interaction patterns, and task context. FIE operates at 0.01ms per agent, enabling real-time fidelity decisions.

2. **Computational Budget Optimizer (CBO)**: An optimization algorithm that allocates fixed computational budget across agents to minimize total prediction error. CBO provides theoretical guarantees on approximation quality as a function of budget.

3. **Seamless Fidelity Transition (SFT)**: A mechanism for transitioning agents between fidelity levels without discontinuities or energy artifacts. SFT ensures physical consistency through momentum matching and constraint satisfaction.

4. **Multi-Fidelity Hierarchy**: A principled hierarchy of simulation models from Level 0 (point-mass) to Level 3 (full rigid-body with contact), with clear cost-accuracy tradeoffs at each level.

5. **Theoretical Analysis**: Formal guarantees on approximation error bounds, convergence properties, and computational complexity.

6. **Comprehensive Evaluation**: Extensive experiments demonstrating 23.8× speedup at 94.7% accuracy across foraging, formation control, and collective construction tasks.

### 1.4 Paper Organization

Section 2 reviews related work in multi-fidelity simulation, swarm robotics, and digital twins. Section 3 formalizes the multi-fidelity swarm simulation problem. Part 2 (Methodology) details the MFST framework including FIE, CBO, and SFT components with theoretical analysis. Part 3 (Experiments) presents comprehensive evaluation and ablation studies.

---

## 2. Related Work

### 2.1 Multi-Fidelity Modeling

Multi-fidelity methods combine models of varying accuracy and cost to achieve better accuracy per computational dollar. The foundational work on co-kriging [Kennedy & O'Hagan, 2000] established how low-fidelity model outputs can be corrected using sparse high-fidelity samples. Multi-fidelity Monte Carlo [Peherstorfer et al., 2018] extended these ideas to uncertainty quantification, achieving variance reduction through optimal sample allocation across fidelity levels.

In computational mechanics, multi-fidelity approaches are well-established. Eldred [2009] developed strategies for combining surrogate models with high-fidelity simulations in design optimization. More recently, multi-fidelity neural networks [Meng & Karniadakis, 2020] learn mappings between fidelity levels using deep learning, enabling more flexible correction functions.

However, existing multi-fidelity methods typically assume:

- **Spatial uniformity**: All spatial regions use the same fidelity
- **Temporal consistency**: Fidelity levels remain fixed during simulation
- **Independent samples**: Samples at different fidelities are independent

Swarm simulation violates all three assumptions: agents require different fidelities, fidelity requirements change dynamically, and agent states are highly coupled through interactions. Our work extends multi-fidelity principles to handle these unique challenges.

### 2.2 Physics Simulation for Robotics

Robot simulation has advanced significantly with platforms like MuJoCo [Todorov et al., 2012], PyBullet [Coumans & Bai, 2016], and Isaac Gym [Makoviychuk et al., 2021]. These simulators provide high-fidelity rigid-body dynamics with contact, enabling sim-to-real transfer for manipulation and locomotion.

For multi-robot simulation, Gazebo [Koenig & Howard, 2004] remains the standard for ROS integration, though it struggles with large robot counts. ARGoS [Pinciroli et al., 2012] specifically targets swarm simulation with simplified physics for scalability. More recent work on GPU-accelerated simulation [Freeman et al., 2021] enables thousands of parallel environments but focuses on single-robot policy training rather than swarm coordination.

The key gap is bridging high-fidelity single-robot simulation with large-scale swarm requirements. Existing approaches force a choice; we enable both through adaptive fidelity.

### 2.3 Swarm Robotics and Emergent Behavior

Swarm robotics draws inspiration from biological collectives—ant colonies, bird flocks, fish schools—that exhibit emergent intelligence through local interactions [Brambilla et al., 2013]. Key behaviors include:

- **Collective motion**: Coordinated movement patterns from local alignment and repulsion rules [Reynolds, 1987; Vicsek et al., 1995]
- **Task allocation**: Dynamic division of labor through response thresholds [Bonabeau et al., 1996]
- **Collective construction**: Building structures through stigmergic coordination [Werfel et al., 2014]
- **Foraging**: Efficient resource exploitation through recruitment and pheromone trails [Ducatelle et al., 2011]

These behaviors depend critically on physical interactions—collision avoidance, force transmission, environmental sensing—that low-fidelity simulations often fail to capture. Our framework preserves emergence by applying high fidelity precisely where physical interactions are most consequential.

### 2.4 Digital Twins

Digital twins—virtual replicas synchronized with physical systems—have transformed manufacturing, infrastructure, and aerospace [Glaessgen & Stargel, 2012]. In robotics, digital twins enable predictive maintenance, mission planning, and operator training [Schluse et al., 2018].

For swarm systems, digital twins face unique challenges:

- **Scale**: Thousands of synchronized agents
- **Emergence**: Collective behaviors not predictable from individual models
- **Adaptation**: Swarms reconfigure dynamically

Recent work on swarm digital twins [Dorigo et al., 2021] emphasizes the need for scalable, accurate simulation but lacks concrete solutions for the fidelity-computation tradeoff. We directly address this gap.

### 2.5 Adaptive Simulation

Adaptive methods that vary computational effort based on context appear across scientific computing. Adaptive mesh refinement [Berger & Colella, 1989] places computational nodes where error is highest. Level-of-detail rendering [Luebke et al., 2003] reduces geometric complexity for distant objects. Adaptive time-stepping [Hairer et al., 1993] varies integration step size based on local dynamics.

In robotics simulation, adaptive contact resolution [Macklin et al., 2014] focuses computation on active contacts. Subspace dynamics [Barbič & Zhao, 2011] reduces deformable body DOFs based on deformation magnitude. These ideas inspire our approach but focus on single objects; we extend adaptivity to agent-level fidelity in multi-agent systems.

---

## 3. Problem Formulation

### 3.1 Swarm Simulation Model

Consider a swarm of $N$ agents with state $\mathbf{s}_i \in \mathcal{S}$ for agent $i \in \{1, \ldots, N\}$. The global swarm state is $\mathbf{S} = (\mathbf{s}_1, \ldots, \mathbf{s}_N) \in \mathcal{S}^N$. Agent dynamics follow:

$$\mathbf{s}_i^{t+1} = f_i(\mathbf{s}_i^t, \mathbf{u}_i^t, \mathbf{S}_{-i}^t, \mathbf{e}^t)$$

where $\mathbf{u}_i^t$ is agent $i$'s control input, $\mathbf{S}_{-i}^t$ denotes other agents' states (for interaction), and $\mathbf{e}^t$ represents environmental state (obstacles, terrain, resources).

The dynamics function $f_i$ encapsulates physics simulation, which can be computed at different fidelity levels $\ell \in \{0, 1, \ldots, L\}$:

$$f_i^{(\ell)}: \mathcal{S} \times \mathcal{U} \times \mathcal{S}^{N-1} \times \mathcal{E} \rightarrow \mathcal{S}$$

Higher fidelity levels provide more accurate dynamics but at greater computational cost.

### 3.2 Fidelity Hierarchy

We define a hierarchy of fidelity levels with increasing accuracy and cost:

**Level 0 - Point Mass (Kinematic)**

- State: Position and velocity $\mathbf{s}_i = (\mathbf{p}_i, \mathbf{v}_i) \in \mathbb{R}^6$
- Dynamics: $\mathbf{v}_i^{t+1} = \mathbf{v}_i^t + \mathbf{u}_i^t \Delta t$, $\mathbf{p}_i^{t+1} = \mathbf{p}_i^t + \mathbf{v}_i^{t+1} \Delta t$
- Cost: $c_0 = 0.001$ ms
- Captures: Gross motion, spatial distribution
- Ignores: Rotation, collisions, physical interactions

**Level 1 - Rigid Body (No Contact)**

- State: Pose and twist $\mathbf{s}_i = (\mathbf{p}_i, \mathbf{R}_i, \mathbf{v}_i, \boldsymbol{\omega}_i) \in SE(3) \times \mathbb{R}^6$
- Dynamics: Rigid body equations $\mathbf{M}\dot{\boldsymbol{\nu}} = \mathbf{F}_{ext}$
- Cost: $c_1 = 0.05$ ms
- Captures: Rotation dynamics, inertial effects
- Ignores: Contact, friction, deformation

**Level 2 - Rigid Body with Simplified Contact**

- State: Full pose plus contact mode
- Dynamics: Impulse-based contact resolution [Mirtich & Canny, 1995]
- Cost: $c_2 = 0.5$ ms
- Captures: Collision response, coarse friction
- Ignores: Contact patch geometry, surface deformation

**Level 3 - Full Physics (Contact + Friction)**

- State: Full pose plus contact patch state
- Dynamics: LCP-based contact [Stewart & Trinkle, 1996] or convex optimization [Anitescu, 2006]
- Cost: $c_3 = 5.0$ ms
- Captures: All relevant physics
- Limitations: Computational cost, numerical stiffness

### 3.3 The Fidelity Allocation Problem

**Definition 1 (Fidelity Allocation).** A fidelity allocation is a mapping $\phi: \{1, \ldots, N\} \rightarrow \{0, 1, \ldots, L\}$ assigning each agent to a fidelity level.

**Definition 2 (Computational Budget).** The computational budget $B$ is the total time available for simulating one timestep:

$$C(\phi) = \sum_{i=1}^N c_{\phi(i)} \leq B$$

**Definition 3 (Approximation Error).** The approximation error for allocation $\phi$ is the expected deviation from full high-fidelity simulation:

$$E(\phi) = \mathbb{E}\left[\sum_{i=1}^N \left\| f_i^{(L)}(\mathbf{s}_i, \cdot) - f_i^{(\phi(i))}(\mathbf{s}_i, \cdot) \right\|^2 \right]$$

**Problem 1 (Optimal Fidelity Allocation).** Find the allocation $\phi^*$ that minimizes approximation error subject to computational budget:

$$\phi^* = \arg\min_\phi E(\phi) \quad \text{s.t.} \quad C(\phi) \leq B$$

This is a combinatorial optimization problem with $(L+1)^N$ possible allocations—intractable for large swarms. We develop efficient approximation algorithms with provable guarantees.

### 3.4 Behavioral Criticality

Not all agents contribute equally to overall swarm behavior. We define behavioral criticality to capture each agent's importance:

**Definition 4 (Behavioral Criticality).** The behavioral criticality $\kappa_i \in [0, 1]$ of agent $i$ measures its influence on global swarm objectives:

$$\kappa_i = \frac{\partial \mathcal{J}(\mathbf{S})}{\partial \mathbf{s}_i} \cdot \Delta \mathbf{s}_i^{max}$$

where $\mathcal{J}$ is the swarm objective (foraging efficiency, formation error, etc.) and $\Delta \mathbf{s}_i^{max}$ is the maximum possible state change.

Agents with high criticality require accurate simulation because their state errors propagate to global objectives. Agents with low criticality can tolerate approximation with minimal impact.

### 3.5 Interaction Density

Agent interactions drive emergent behavior and require high fidelity for accurate capture:

**Definition 5 (Interaction Density).** The interaction density $\rho_i$ measures the intensity of agent $i$'s physical interactions:

$$\rho_i = \sum_{j \neq i} \frac{\mathbf{1}[\|\mathbf{p}_i - \mathbf{p}_j\| < r_{int}]}{V_{int}} \cdot \left(1 - \frac{\|\mathbf{p}_i - \mathbf{p}_j\|}{r_{int}}\right)$$

where $r_{int}$ is the interaction radius and $V_{int}$ is the interaction volume.

High interaction density indicates crowded regions where contact, collision avoidance, and coordination require detailed physics.

### 3.6 Task Relevance

Current task context affects fidelity requirements:

**Definition 6 (Task Relevance).** The task relevance $\tau_i$ indicates whether agent $i$ is executing a task phase requiring high fidelity:

$$
\tau_i = \begin{cases}
1 & \text{if agent } i \text{ in critical task phase (manipulation, docking)} \\
\alpha \in (0,1) & \text{if agent } i \text{ in moderate phase (navigation near obstacles)} \\
0 & \text{if agent } i \text{ in low-criticality phase (free space motion)}
\end{cases}
$$

### 3.7 Importance-Weighted Fidelity

Combining these factors yields the fidelity importance:

**Definition 7 (Fidelity Importance).** The fidelity importance $I_i$ for agent $i$ is:

$$I_i = w_\kappa \kappa_i + w_\rho \rho_i + w_\tau \tau_i$$

where weights $w_\kappa, w_\rho, w_\tau \geq 0$ are task-specific hyperparameters.

Agents with high importance receive high fidelity allocation; agents with low importance receive low fidelity, conserving computational budget.

### 3.8 Problem Complexity

**Theorem 1 (NP-Hardness).** The optimal fidelity allocation problem (Problem 1) is NP-hard.

_Proof Sketch._ We reduce from the Knapsack problem. Given items with values and weights, construct agents where fidelity levels correspond to taking/not-taking items, costs are weights, and error reduction is value. An optimal fidelity allocation solves the underlying Knapsack instance.

**Corollary 1.** No polynomial-time algorithm achieves optimal allocation unless P = NP.

This motivates our development of efficient approximation algorithms in Part 2.

---

## 4. Preliminaries

### 4.1 Notation Summary

| Symbol         | Description                        |
| -------------- | ---------------------------------- |
| $N$            | Number of agents in swarm          |
| $\mathbf{s}_i$ | State of agent $i$                 |
| $\mathbf{S}$   | Global swarm state                 |
| $f_i^{(\ell)}$ | Dynamics at fidelity level $\ell$  |
| $c_\ell$       | Computational cost of level $\ell$ |
| $\phi$         | Fidelity allocation function       |
| $B$            | Computational budget               |
| $I_i$          | Fidelity importance of agent $i$   |
| $\kappa_i$     | Behavioral criticality             |
| $\rho_i$       | Interaction density                |
| $\tau_i$       | Task relevance                     |

### 4.2 Assumptions

**Assumption 1 (Monotonic Fidelity).** Higher fidelity levels are strictly more accurate:
$$\ell' > \ell \Rightarrow E_i^{(\ell')} \leq E_i^{(\ell)}$$

**Assumption 2 (Monotonic Cost).** Higher fidelity levels are strictly more expensive:
$$\ell' > \ell \Rightarrow c_{\ell'} > c_\ell$$

**Assumption 3 (Lipschitz Dynamics).** Dynamics are Lipschitz continuous in state:
$$\|f_i^{(\ell)}(\mathbf{s}) - f_i^{(\ell)}(\mathbf{s}')\| \leq L_\ell \|\mathbf{s} - \mathbf{s}'\|$$

**Assumption 4 (Bounded Error Growth).** Approximation errors grow at most linearly per timestep:
$$E_i^{(\ell)}(t+1) \leq (1 + \gamma_\ell) E_i^{(\ell)}(t) + \epsilon_\ell$$

### 4.3 Computational Model

We assume a parallel simulation architecture where agents can be simulated concurrently up to hardware limits. The computational budget $B$ represents wall-clock time for one simulation timestep, with parallelism factored into effective costs.

For $P$ parallel workers:
$$C_{eff}(\phi) = \max_{p \in \{1,\ldots,P\}} \sum_{i: \text{agent } i \text{ assigned to worker } p} c_{\phi(i)}$$

Load balancing across workers is addressed in our implementation (Part 2).

---

## 5. Theoretical Foundations

### 5.1 Error Decomposition

Total simulation error decomposes into three sources:

$$E_{total} = E_{fidelity} + E_{transition} + E_{coupling}$$

1. **Fidelity Error** $E_{fidelity}$: Error from using lower fidelity for individual agents
2. **Transition Error** $E_{transition}$: Error from changing fidelity levels
3. **Coupling Error** $E_{coupling}$: Error from inconsistent fidelity across interacting agents

Our framework addresses all three: FIE minimizes fidelity error through intelligent allocation, SFT minimizes transition error through consistent handoffs, and CBO considers coupling in its optimization.

### 5.2 Fidelity-Accuracy Tradeoff

**Lemma 1 (Cost-Error Tradeoff).** For a single agent, the relationship between computational cost and approximation error follows:

$$E_i^{(\ell)} \approx \alpha_i \cdot c_\ell^{-\beta}$$

where $\alpha_i$ is agent-specific and $\beta > 0$ is the fidelity scaling exponent.

_Proof._ Physics simulation accuracy scales with integration order and collision detection resolution, both of which have polynomial cost scaling.

This motivates Pareto-optimal allocation: distribute budget to equalize marginal error reduction across agents.

### 5.3 Emergence Preservation

**Definition 8 (Emergence Fidelity).** A multi-fidelity simulation preserves emergence if collective behaviors remain statistically indistinguishable from high-fidelity simulation:

$$\mathbb{P}[B(\mathbf{S}^{MFST}) = b] \approx \mathbb{P}[B(\mathbf{S}^{HiFi}) = b] \quad \forall b \in \mathcal{B}$$

where $B$ is an emergent behavior classifier and $\mathcal{B}$ is the set of possible collective behaviors.

Our theoretical analysis (Part 2) proves emergence preservation under appropriate importance allocation.

---

_Continued in Part 2: Methodology (FIE, CBO, SFT algorithms with theoretical analysis)_

---

## References (Partial - Part 1)

[Anitescu, 2006] M. Anitescu, "Optimization-based simulation of nonsmooth rigid multibody dynamics," Mathematical Programming, 2006.

[Barbič & Zhao, 2011] J. Barbič and Y. Zhao, "Real-time large-deformation substructuring," ACM TOG, 2011.

[Berger & Colella, 1989] M. Berger and P. Colella, "Local adaptive mesh refinement for shock hydrodynamics," J. Comp. Phys., 1989.

[Bonabeau et al., 1996] E. Bonabeau et al., "Quantitative study of the fixed threshold model for the regulation of division of labour in insect societies," Proc. Royal Soc. B, 1996.

[Brambilla et al., 2013] M. Brambilla et al., "Swarm robotics: A review from the swarm engineering perspective," Swarm Intelligence, 2013.

[Coumans & Bai, 2016] E. Coumans and Y. Bai, "PyBullet, a Python module for physics simulation," 2016.

[Dorigo et al., 2021] M. Dorigo et al., "Swarm robotics: Past, present, and future," Proc. IEEE, 2021.

[Ducatelle et al., 2011] F. Ducatelle et al., "Self-organized cooperation between robotic swarms," Swarm Intelligence, 2011.

[Eldred, 2009] M. Eldred, "Recent advances in non-intrusive polynomial chaos and stochastic collocation methods for uncertainty analysis and design," AIAA, 2009.

[Freeman et al., 2021] C. Freeman et al., "Brax—A differentiable physics engine for large scale rigid body simulation," arXiv, 2021.

[Glaessgen & Stargel, 2012] E. Glaessgen and D. Stargel, "The digital twin paradigm for future NASA and US Air Force vehicles," AIAA, 2012.

[Hairer et al., 1993] E. Hairer et al., "Solving Ordinary Differential Equations I," Springer, 1993.

[Kennedy & O'Hagan, 2000] M. Kennedy and A. O'Hagan, "Predicting the output of a complex computer code when fast approximations are available," Biometrika, 2000.

[Koenig & Howard, 2004] N. Koenig and A. Howard, "Design and use paradigms for Gazebo, an open-source multi-robot simulator," IROS, 2004.

[Luebke et al., 2003] D. Luebke et al., "Level of Detail for 3D Graphics," Morgan Kaufmann, 2003.

[Macklin et al., 2014] M. Macklin et al., "Unified particle physics for real-time applications," ACM TOG, 2014.

[Makoviychuk et al., 2021] V. Makoviychuk et al., "Isaac Gym: High performance GPU-based physics simulation for robot learning," NeurIPS, 2021.

[Meng & Karniadakis, 2020] X. Meng and G. Karniadakis, "A composite neural network that learns from multi-fidelity data," J. Comp. Phys., 2020.

[Mirtich & Canny, 1995] B. Mirtich and J. Canny, "Impulse-based simulation of rigid bodies," I3D, 1995.

[Peherstorfer et al., 2018] B. Peherstorfer et al., "Survey of multifidelity methods in uncertainty propagation, inference, and optimization," SIAM Review, 2018.

[Pinciroli et al., 2012] C. Pinciroli et al., "ARGoS: A modular, parallel, multi-engine simulator for multi-robot systems," Swarm Intelligence, 2012.

[Reynolds, 1987] C. Reynolds, "Flocks, herds and schools: A distributed behavioral model," SIGGRAPH, 1987.

[Schluse et al., 2018] M. Schluse et al., "Experimentable digital twins—Streamlining simulation-based systems engineering for Industry 4.0," IEEE Trans. Industrial Informatics, 2018.

[Stewart & Trinkle, 1996] D. Stewart and J. Trinkle, "An implicit time-stepping scheme for rigid body dynamics with inelastic collisions and Coulomb friction," IJRR, 1996.

[Todorov et al., 2012] E. Todorov et al., "MuJoCo: A physics engine for model-based control," IROS, 2012.

[Vicsek et al., 1995] T. Vicsek et al., "Novel type of phase transition in a system of self-driven particles," Physical Review Letters, 1995.

[Werfel et al., 2014] J. Werfel et al., "Designing collective behavior in a termite-inspired robot construction team," Science, 2014.

---

_End of Part 1_
