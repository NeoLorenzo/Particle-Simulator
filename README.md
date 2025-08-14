# Particle Simulator

> This project is a high-performance 2D particle simulator in Python that models a universe of particles governed by gravitational attraction and inelastic collisions. The architecture is designed for physical realism, computational efficiency, and verifiable energy conservation, enabling the emergence of complex behaviors like orbital mechanics, accretion, and thermodynamic equilibrium.

> The simulation's performance is achieved through a hybrid optimization strategy, Numba-JIT compilation of core physics loops, and vectorized NumPy operations. Visualization is handled by Pygame, featuring temperature-based coloring and a bloom effect for visual clarity.

---

## Core Features

| Feature | Description |
| :--- | :--- |
| **N-Body Gravitational Simulation** | Utilizes a Barnes-Hut approximation to efficiently calculate long-range gravitational forces with O(n log n) complexity. |
| **Energy-Conserving Collision Physics** | Implements a multi-stage inelastic collision model that correctly transforms kinetic and potential energy into thermal energy, adhering to the first law of thermodynamics. |
| **High-Performance Architecture** | Employs a hybrid spatial partitioning scheme (QuadTree for gravity, Uniform Grid for collisions) and JIT-compiles the hottest computational loops to machine code with Numba. |
| **Stable Numerical Integration** | Built on the Velocity Verlet integration method to ensure long-term stability and minimize energy drift. |
| **Configurable & Deterministic** | Simulation parameters are externalized to a <kbd>config.json</kbd> file, and all random processes are controlled by a single master seed for reproducible results. |

---

## Physics & Simulation Model

The simulation loop is carefully structured to maintain physical accuracy. Each time step advances by first integrating continuous forces (gravity) and then resolving discrete events (collisions).

<details>
<summary><strong>1. Numerical Integration: Velocity Verlet</strong></summary>

<br>
The state of the particles is advanced using the **Velocity Verlet** method, chosen for its excellent energy conservation properties over long simulation runs compared to simpler integrators.

The update sequence per tick is:
1.  **Update Position:** `p(t+dt) = p(t) + v(t)dt + 0.5a(t)dt²`
2.  **Update Spatial Structures:** The QuadTree and spatial grid are rebuilt based on the new positions.
3.  **Calculate New Forces:** New gravitational accelerations `a(t+dt)` are computed using the updated spatial structures.
4.  **Update Velocity:** `v(t+dt) = v(t) + 0.5 * (a(t) + a(t+dt))dt`
<br>

</details>

<details>
<summary><strong>2. Gravitational Interaction: Barnes-Hut Approximation</strong></summary>

<br>
Gravitational forces are modeled using a Barnes-Hut N-body simulation. This algorithm avoids the O(n²) complexity of direct summation by treating distant clusters of particles as a single center of mass.

*   **QuadTree Partitioning:** The simulation space is recursively divided into a QuadTree.
*   **Center of Mass Calculation:** The total mass and center of mass are computed for each node in the tree.
*   **Force Calculation:** To calculate the force on a particle, the tree is traversed. If a node is sufficiently far away (determined by the `barnes_hut_theta` parameter), its entire mass is used in a single force calculation. Otherwise, the algorithm traverses deeper into the tree.
*   **Softening Factor:** To prevent numerical instability from near-infinite forces between close particles, a `softening_factor` is added to the distance term: `F = G * (m₁*m₂) / (r² + s)`.
<br>

</details>

<details>
<summary><strong>3. Collision Dynamics & Thermodynamics</strong></summary>

<br>
Collisions are modeled as discrete, inelastic events that strictly conserve the total energy of the interacting pair by converting it between kinetic, potential, and thermal forms.

The resolution process within `_resolve_collision_jit` is:
1.  **Overlap Resolution:** Overlapping particles are repositioned along their normal vector to conserve the pair's center of mass.
2.  **Potential Energy Accounting:** This repositioning alters the distance between particles, changing their mutual gravitational potential energy. This `pe_change` is calculated and accounted for.
3.  **Inelastic Velocity Update:** Velocities are updated based on the `coefficient_of_restitution`, resulting in a loss of kinetic energy for the system.
4.  **Energy Transformation:** The net heat generated is calculated by balancing the energy accounts:
    ```
    heat_generated = ke_lost - pe_change
    ```
    The energy cost of pushing particles apart (potential energy gain) is subtracted from the kinetic energy lost. This net energy is then distributed as thermal energy, raising the particles' temperatures.
<br>

</details>

<details>
<summary><strong>4. System-Wide Energy Conservation</strong></summary>

<br>
To counteract minor numerical errors inherent in discrete simulations, a global energy correction mechanism is in place. The total system energy (Kinetic + Potential + Thermal) is tracked, and any drift is accumulated. Periodically, this accumulated error is injected back into (or removed from) the system's total thermal energy, ensuring that the simulation remains energy-neutral over extended periods.
<br>

</details>

---

## Computational Architecture & Optimizations

<details>
<summary><strong>1. Hybrid Spatial Partitioning</strong></summary>

<br>
The simulation employs a deliberate, dual-pronged strategy for spatial partitioning, using the optimal data structure for each type of physical interaction:
*   **QuadTree (Barnes-Hut):** Ideal for the hierarchical approximations needed for long-range gravity. The entire tree construction and mass calculation process is JIT-compiled with Numba for maximum efficiency.
*   **Uniform Spatial Grid:** Used for broad-phase collision detection. This structure is optimal for identifying spatially local neighbors for short-range interactions, reducing the complexity of collision checks from O(n²) to nearly O(n).
<br>

</details>

<details>
<summary><strong>2. Just-in-Time (JIT) Compilation</strong></summary>

<br>
The most computationally intensive parts of the simulation are written in a restricted subset of Python and compiled to optimized machine code at runtime using **Numba**. This includes:
*   The entire Barnes-Hut gravity calculation loop.
*   The QuadTree construction and center-of-mass calculations.
*   The spatial grid traversal and pairwise collision resolution logic.
<br>

</details>

<details>
<summary><strong>3. Vectorization</strong></summary>

<br>
Particle data is stored in **NumPy** arrays, enabling vectorized operations (Structure of Arrays) for integration steps, boundary checks, and property calculations. This avoids slow Python loops and leverages optimized, low-level library code.
<br>

</details>

---

## Visualization

*   **Rendering:** The simulation is visualized using **Pygame**.
*   **Thermodynamic Coloring:** Particle colors are determined by their temperature, mapping a physical property to a visual one. The gradient ranges from red (cool) through yellow to white (hot).
*   **Bloom Effect:** A post-processing bloom effect is applied to bright, hot particles to create a glow, enhancing visual feedback on the system's energy distribution.

---

## How to Run the Simulation

### 1. Prerequisites
*   Python 3.x
*   Pygame
*   NumPy
*   Numba

### 2. Configuration
Simulation behavior is controlled by two files:
*   <kbd>constants.py</kbd>: Defines static application values like screen resolution and framerate.
*   <kbd>config.json</kbd>: Controls the parameters of the simulation experiment.

| Parameter | Description |
| :--- | :--- |
| <kbd>particle_count</kbd> | The number of particles to simulate. |
| <kbd>gravity_constant</kbd> | The strength of the gravitational force. |
| <kbd>softening_factor</kbd> | Prevents extreme forces at close range to maintain stability. |
| <kbd>barnes_hut_theta</kbd> | Controls Barnes-Hut accuracy. Lower is more accurate but slower. |
| <kbd>coefficient_of_restitution</kbd> | The "bounciness" of collisions (0.0 = inelastic, 1.0 = elastic). |
| <kbd>grid_cell_size_multiplier</kbd>| Tunes the spatial grid cell size for collision detection performance. |

### 3. Execution
Run the main script from your terminal:
```bash
python main.py
```