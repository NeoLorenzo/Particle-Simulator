<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyGame-6E7072?style=for-the-badge&logo=pygame&logoColor=white" alt="Pygame"/>
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Powered%20by-Numba-blueviolet?style=for-the-badge" alt="Numba"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License: MIT"/>
</p>

<h1 align="center">Particle Simulator</h1>

<p align="center">
  <img src="./assets/simulation_demo.gif" width="80%">
</p>

<p align="center">
  <strong><a href="https://NeoLorenzo.github.io/Particle-Simulator/timelapse.html" target="_blank">Click Here to Watch the Full High-Resolution Timelapse Video</a></strong>
</p>

<p align="center">
  <a href="#core-features">Core Features</a> •
  <a href="#physics--simulation-model">Physics Model</a> •
  <a href="#computational-architecture--optimizations">Architecture</a> •
  <a href="#how-to-run-the-simulation">How to Run</a>
</p>

---

> This project is a high-performance 2D particle simulator engineered in Python to model complex astrophysical phenomena. The simulation universe is populated by particles governed by first-principles physics, including gravitational attraction, inelastic collisions, and thermodynamic evolution. Its architecture is meticulously designed for physical realism, computational efficiency, and numerical stability, enabling the emergence of sophisticated behaviors such as orbital mechanics, accretion disk formation, and stellar explosions.

> The simulator's exceptional performance is achieved through a sophisticated hybrid computational paradigm, combining Numba-JIT compilation of core physics kernels with vectorized NumPy operations. Visualization is rendered via Pygame, featuring a physically-inspired, temperature-based color gradient and a post-processing bloom effect for enhanced visual fidelity.

---

## Core Features

| Feature | Description |
| :--- | :--- |
| **N-Body Gravitational Simulation** | Employs a highly-optimized Barnes-Hut algorithm to simulate long-range gravitational forces with O(n log n) complexity, enabling large-scale structure formation. |
| **First-Principles Thermodynamics** | Implements a multi-stage, energy-conserving collision model that correctly transforms kinetic and potential energy into thermal energy, adhering to the laws of thermodynamics. |
| **Emergent Thermodynamic Phenomena** | Models thermal conduction, radiative cooling, and energetic particle explosions, allowing for the study of complex, system-wide thermodynamic behaviors. |
| **High-Performance Hybrid Architecture** | Utilizes a dual spatial-partitioning scheme (QuadTree for gravity, Uniform Grid for collisions) and JIT-compiles the most demanding computational loops to native machine code. |
| **Symplectic Numerical Integration** | Built upon the Velocity Verlet integration method, a symplectic integrator that ensures superior long-term energy and momentum conservation. |
| **Configurable & Deterministic** | Simulation parameters are externalized to a <kbd>config.json</kbd> file, and all stochastic processes are governed by a single master seed for fully reproducible scientific experiments. |

---

## Physics & Simulation Model

The simulation loop is rigorously structured to maintain physical fidelity. Each discrete time step advances the system by first integrating continuous forces (gravity) and subsequently resolving discrete, instantaneous events (collisions and thermodynamic exchanges).

<details>
<summary><strong>1. Numerical Integration: Velocity Verlet</strong></summary>
<br>
The temporal evolution of the particle system is computed using the <strong>Velocity Verlet</strong> method. This symplectic integrator is selected for its exceptional energy and momentum conservation properties over extended simulation runs, a critical feature for maintaining the stability of orbital systems.

The update sequence per tick is as follows:
<ul>
    <li><strong>Update Position:</strong> <code>p(t+dt) = p(t) + v(t)dt + 0.5a(t)dt²</code></li>
    <li><strong>Rebuild Spatial Structures:</strong> The QuadTree and spatial grid are reconstructed based on the new particle positions.</li>
    <li><strong>Calculate New Forces:</strong> New gravitational accelerations <code>a(t+dt)</code> are computed using the updated spatial hierarchies.</li>
    <li><strong>Update Velocity:</strong> <code>v(t+dt) = v(t) + 0.5 * (a(t) + a(t+dt))dt</code></li>
</ul>
</details>

<details>
<summary><strong>2. Gravitational Interaction: Barnes-Hut Approximation</strong></summary>
<br>
Gravitational forces are modeled using a <strong>Barnes-Hut N-body simulation</strong>, an elegant approximation that reduces the computational complexity from O(n²) to O(n log n).
<ul>
    <li><strong>Hierarchical Partitioning:</strong> The simulation space is recursively subdivided into a QuadTree data structure.</li>
    <li><strong>Center of Mass Calculation:</strong> The aggregate mass and center of mass are computed for each node in the tree in a single post-order traversal.</li>
    <li><strong>Multipole Expansion:</strong> To calculate the net force on a particle, the tree is traversed. If a node is sufficiently distant (governed by the <code>barnes_hut_theta</code> parameter, analogous to a multipole acceptance criterion), its entire mass is treated as a single point source. Otherwise, the algorithm descends to a deeper level of the hierarchy.</li>
    <li><strong>Gravitational Softening:</strong> To prevent numerical instability and singularities from near-infinite forces between close particles, a <code>softening_factor</code> is introduced to the denominator of the force equation:
    <blockquote>
    F = G * (m₁*m₂) / (r² + s)
    </blockquote>
    </li>
</ul>
</details>

<details>
<summary><strong>3. Collision Dynamics & Energy Conservation</strong></summary>
<br>
Collisions are modeled as discrete, inelastic events that strictly conserve the total energy of an interacting pair by transforming it between kinetic, potential, and thermal forms.

The resolution process within the <code>_resolve_collision_jit</code> kernel is:
<ol>
    <li><strong>Overlap Resolution:</strong> Spatially overlapping particles are repositioned along their normal vector, preserving the pair's center of mass.</li>
    <li><strong>Potential Energy Accounting:</strong> This repositioning alters the inter-particle distance, changing their mutual gravitational potential energy. This <code>pe_change</code> is precisely calculated.</li>
    <li><strong>Inelastic Impulse:</strong> Velocities are updated based on the <code>coefficient_of_restitution</code>, which models the kinetic energy dissipated during the collision.</li>
    <li><strong>First Law of Thermodynamics:</strong> The net thermal energy (heat) generated is calculated by balancing the system's energy budget, directly enforcing the law of conservation of energy:
    <blockquote>
    <code>heat_generated = kinetic_energy_lost - potential_energy_change</code>
    </blockquote>
    This resulting thermal energy is then distributed between the particles, raising their internal temperatures.</li>
</ol>
</details>

<details>
<summary><strong>4. Advanced Thermodynamic Modeling</strong></summary>
<br>
The simulation incorporates several thermodynamic processes that contribute to emergent, system-wide behaviors:
<ul>
    <li><strong>Thermal Conduction:</strong> Particles in physical contact exchange thermal energy at a rate proportional to their temperature differential, an abstraction of Fourier's law of heat conduction. This process is accelerated by the spatial grid.</li>
    <li><strong>Radiative Cooling:</strong> The entire system slowly loses energy via a <code>thermal_damping_factor</code>, which models black-body radiation into the vacuum of space. This prevents runaway temperature escalation and allows the system to approach thermal equilibrium.</li>
    <li><strong>Explosive Events:</strong> Particles exceeding a critical temperature threshold undergo a catastrophic explosion. Their entire thermal energy is converted into kinetic energy and imparted as a shockwave to their neighbors, simulating phenomena like supernovae and removing the source particle from the system.</li>
</ul>
</details>

---

## Computational Architecture & Optimizations

<details>
<summary><strong>1. Hybrid Spatial Partitioning</strong></summary>
<br>
The simulation employs a sophisticated, dual-pronged strategy for spatial partitioning, leveraging the optimal data structure for each physical interaction domain:
<ul>
    <li><strong>QuadTree (Barnes-Hut):</strong> Ideal for the hierarchical, far-field approximations required for gravity. The entire tree is constructed and flattened into contiguous NumPy arrays for direct consumption by the Numba-JIT kernels.</li>
    <li><strong>Uniform Spatial Grid:</strong> Utilized for broad-phase collision and heat transfer detection. This structure is optimal for identifying spatially local neighbors for short-range interactions, reducing the complexity of these checks from O(n²) to nearly O(n).</li>
</ul>
</details>

<details>
<summary><strong>2. Just-in-Time (JIT) Compilation with Numba</strong></summary>
<br>
The most computationally intensive kernels of the simulation are written in a restricted, high-performance subset of Python and compiled to optimized machine code at runtime using <strong>Numba</strong>. This "zero-overhead" approach applies to:
<ul>
    <li>The entire Barnes-Hut gravity calculation, including tree traversal.</li>
    <li>The QuadTree construction, center-of-mass calculation, and flattening routines.</li>
    <li>The spatial grid traversal and all pairwise collision and heat transfer physics.</li>
</ul>
</details>

<details>
<summary><strong>3. Vectorization and Memory Management</strong></summary>
<br>
All particle data is stored as <strong>NumPy</strong> arrays (Structure of Arrays), enabling vectorized operations that delegate computations to highly optimized, low-level C and Fortran libraries. To eliminate runtime overhead, memory for the QuadTree nodes is pre-allocated at the start of each frame, avoiding costly dynamic memory allocation within the simulation loop.
</details>

---

## Visualization

- **Rendering Engine**
  - The simulation state is visualized in real-time using **Pygame**.
- **Thermodynamic Coloring**
  - Particle color is mapped directly to its temperature, following a physically-inspired black-body radiation spectrum. The gradient progresses from black through purple, blue, green, yellow, red, and finally to white-hot, providing intuitive visual feedback on the system's energy distribution.
- **Bloom Post-Processing**
  - A post-processing bloom effect is applied to high-temperature particles, creating a luminous glow that enhances the visual representation of energetic events and dense, hot clusters.

---

## How to Run the Simulation

### 1. Prerequisites
A working installation of Python 3.x is required. The necessary libraries can be installed via pip:
```bash
pip install pygame numpy numba
```

### 2. Configuration
Simulation behavior is controlled by two primary files:
- <kbd>constants.py</kbd>: Defines static application values like screen resolution and rendering constants.
- <kbd>config.json</kbd>: Controls the tunable parameters of the scientific experiment.

| Parameter | Description |
| :--- | :--- |
| <kbd>particle_count</kbd> | The initial number of particles in the simulation. |
| <kbd>gravity_constant</kbd> | The universal gravitational constant, controlling the strength of gravity. |
| <kbd>softening_factor</kbd> | A numerical stability parameter to prevent singularities at close range. |
| <kbd>barnes_hut_theta</kbd> | The multipole acceptance criterion for the Barnes-Hut algorithm. Lower values increase accuracy at the cost of performance. |
| <kbd>coefficient_of_restitution</kbd> | The elasticity of collisions (0.0 = perfectly inelastic, 1.0 = perfectly elastic). |
| <kbd>thermal_damping_factor</kbd> | The rate of system-wide energy loss due to radiative cooling. |
| <kbd>explosion_efficiency</kbd> | The fraction of a particle's thermal energy converted to kinetic energy during an explosion. |
| <kbd>grid_cell_size_multiplier</kbd>| A tuning parameter for the collision detection grid's cell size. |

### 3. Execution
To run the simulation, execute the main script from your terminal:
```bash
python main.py
```
