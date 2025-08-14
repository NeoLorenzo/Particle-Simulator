# Particle Simulator

This project is a high-performance 2D particle simulator in Python that models a universe of particles governed by gravitational attraction and inelastic collisions. The architecture is designed for physical realism, computational efficiency, and verifiable energy conservation, enabling the emergence of complex behaviors like orbital mechanics, accretion, and thermodynamic equilibrium.

The simulation's performance is achieved through a hybrid optimization strategy, Numba-JIT compilation of core physics loops, and vectorized NumPy operations. Visualization is handled by Pygame, featuring temperature-based coloring and a bloom effect for visual clarity.

---

## Core Features

*   **N-Body Gravitational Simulation:** Utilizes a Barnes-Hut approximation to efficiently calculate long-range gravitational forces with O(n log n) complexity.
*   **Energy-Conserving Collision Physics:** Implements a multi-stage inelastic collision model that correctly transforms kinetic and potential energy into thermal energy, adhering to the first law of thermodynamics.
*   **High-Performance Architecture:** Employs a hybrid spatial partitioning scheme (QuadTree for gravity, Uniform Grid for collisions) and JIT-compiles the hottest computational loops to machine code with Numba.
*   **Stable Numerical Integration:** Built on the Velocity Verlet integration method to ensure long-term stability and minimize energy drift.
*   **Configurable & Deterministic:** Simulation parameters are externalized to a `config.json` file, and all random processes are controlled by a single master seed for reproducible results.

---

## Physics & Simulation Model

The simulation loop is carefully structured to maintain physical accuracy. Each time step advances by first integrating continuous forces (gravity) and then resolving discrete events (collisions).

### 1. Numerical Integration: Velocity Verlet

The state of the particles is advanced using the **Velocity Verlet** method, chosen for its excellent energy conservation properties over long simulation runs compared to simpler integrators.

> The update sequence per tick is:
> 1.  **Update Position:** `p(t+dt) = p(t) + v(t)dt + 0.5a(t)dt²`
> 2.  **Update Spatial Structures:** The QuadTree and spatial grid are rebuilt based on the new positions.
> 3.  **Calculate New Forces:** New gravitational accelerations `a(t+dt)` are computed using the updated spatial structures.
> 4.  **Update Velocity:** `v(t+dt) = v(t) + 0.5 * (a(t) + a(t+dt))dt`

### 2. Gravitational Interaction: Barnes-Hut Approximation

Gravitational forces are modeled using a Barnes-Hut N-body simulation. This algorithm avoids the O(n²) complexity of direct summation by treating distant clusters of particles as a single center of mass.

*   **QuadTree Partitioning:** The simulation space is recursively divided into a QuadTree.
*   **Center of Mass Calculation:** The total mass and center of mass are computed for each node in the tree.
*   **Force Calculation:** To calculate the force on a particle, the tree is traversed. If a node is sufficiently far away (determined by the `barnes_hut_theta` parameter), its entire mass is used in a single force calculation. Otherwise, the algorithm traverses deeper into the tree.
*   **Softening Factor:** To prevent numerical instability from near-infinite forces between close particles, a `softening_factor` is added to the distance term: `F = G * (m1*m2) / (r² + s)`.

### 3. Collision Dynamics & Thermodynamics

Collisions are modeled as discrete, inelastic events that strictly conserve the total energy of the interacting pair by converting it between kinetic, potential, and thermal forms.

> The core principle of energy transformation is:
>
> `heat_generated = ke_lost - pe_change`
>
> The energy cost of pushing particles apart (potential energy gain) is subtracted from the kinetic energy lost during the inelastic impact. This net energy is then distributed as thermal energy, raising the particles' temperatures.

### 4. System-Wide Energy Conservation

To counteract minor numerical errors inherent in discrete simulations, a global energy correction mechanism is in place. The total system energy (Kinetic + Potential + Thermal) is tracked, and any drift is accumulated. Periodically, this accumulated error is injected back into (or removed from) the system's total thermal energy, ensuring that the simulation remains energy-neutral over extended periods.

---

## Computational Architecture & Optimizations

### 1. Hybrid Spatial Partitioning

The simulation employs a deliberate, dual-pronged strategy for spatial partitioning, using the optimal data structure for each type of physical interaction:
*   **QuadTree (Barnes-Hut):** Ideal for the hierarchical approximations needed for long-range gravity. The entire tree construction and mass calculation process is JIT-compiled with Numba for maximum efficiency.
*   **Uniform Spatial Grid:** Used for broad-phase collision detection. This structure is optimal for identifying spatially local neighbors for short-range interactions, reducing the complexity of collision checks from O(n²) to nearly O(n).

### 2. Just-in-Time (JIT) Compilation

The most computationally intensive parts of the simulation are written in a restricted subset of Python and compiled to optimized machine code at runtime using **Numba**. This includes the entire Barnes-Hut gravity calculation, QuadTree construction, and the collision resolution logic.

### 3. Vectorization

Particle data is stored in **NumPy** arrays, enabling vectorized operations (Structure of Arrays) for integration steps, boundary checks, and property calculations. This avoids slow Python loops and leverages optimized, low-level library code.

---

## Visualization

*   **Rendering:** The simulation is visualized using Pygame.
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
*   `constants.py`: Defines static application values like screen resolution and framerate.
*   `config.json`: Controls the parameters of the simulation experiment.

| Parameter | Description |
| :--- | :--- |
| `particle_count` | The number of particles to simulate. |
| `gravity_constant` | The strength of the gravitational force. |
| `softening_factor` | Prevents extreme forces at close range to maintain stability. |
| `barnes_hut_theta` | Controls Barnes-Hut accuracy. Lower is more accurate but slower. |
| `coefficient_of_restitution` | The "bounciness" of collisions (0.0 = inelastic, 1.0 = elastic). |
| `grid_cell_size_multiplier`| Tunes the spatial grid cell size for collision detection performance. |

### 3. Execution
Run the main script from your terminal:
```bash
python main.py
```