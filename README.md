# Particle Simulator

This project is a high-performance 2D particle simulator written in Python. It models the behavior of a large number of particles interacting through a localized approximation of gravity and direct, inelastic collisions. The simulation is built with a focus on physical realism, computational efficiency, and long-term energy conservation.

It leverages NumPy for vectorized calculations, Numba for just-in-time (JIT) compilation of performance-critical physics loops, and Pygame for visualization.

## Core Physics & Simulation Logic

The simulation is built on a precise sequence of operations designed to model physical laws accurately while maintaining stability. The core logic is orchestrated by the Velocity Verlet integration method, which dictates the order of calculations for each discrete time step.

### 1. Numerical Integration (Velocity Verlet)

To update particle states over time, the simulation employs the **Velocity Verlet** integration method. This technique is chosen over simpler methods like Euler integration for its superior energy conservation over long periods, which is critical for simulation stability.

The update sequence in each tick is as follows:

1.  **Update Positions (First Half):** Calculate the new positions of particles based on their current velocity and acceleration from the *previous* tick: `p(t+dt) = p(t) + v(t)dt + 0.5a(t)dt²`.
2.  **Calculate New Forces:** Determine the new gravitational forces (and thus new accelerations, `a(t+dt)`) acting on the particles in their newly calculated positions.
3.  **Update Velocities:** Calculate the final velocities for the current tick using an average of the old and new accelerations: `v(t+dt) = v(t) + 0.5 * (a(t) + a(t+dt))dt`.

After this continuous force integration, discrete events like collisions are handled.

### 2. Gravitational Interaction (Barnes-Hut Approximation)

The simulation approximates the gravitational N-body problem using the **Barnes-Hut algorithm**, a scientifically-grounded abstraction that reduces the computational complexity from O(n²) to O(n log n). This allows for the efficient simulation of long-range gravitational forces between thousands of particles.

The force between any two particles is based on Newton's Law of Universal Gravitation, with a key modification for numerical stability:

**F = G * (m1 * m2) / (r² + s)**

Where:
*   **F** is the magnitude of the gravitational force.
*   **G** is the `gravity_constant` from `config.json`.
*   **m1** and **m2** are the masses of the two particles.
*   **r²** is the square of the distance between them.
*   **s** is a `softening_factor` from `config.json`. This prevents the force from becoming infinite when particles get extremely close, avoiding numerical instability and the "slingshot" effect.

The Barnes-Hut algorithm works as follows:
1.  A **QuadTree** is constructed each tick, spatially partitioning all particles into a hierarchy of nodes.
2.  The center of mass and total mass are calculated for each node in the tree.
3.  To calculate the net force on a given particle, the tree is traversed from the root.
4.  For each node, the ratio of the node's width to the distance from the particle (`width / dist`) is calculated.
5.  If this ratio is below a certain threshold (`barnes_hut_theta` in `config.json`), the entire cluster of particles within that node is treated as a single, massive body located at the node's center of mass. A single gravitational force calculation is performed.
6.  If the ratio is above the threshold, the node is too close to be approximated, and the algorithm recursively traverses its children nodes.
7.  If a leaf node is reached, direct force calculations are performed with the individual particles inside it.

### 3. Inelastic Collisions and Energy Transformation

Particle collisions are complex events that transform energy between potential, kinetic, and thermal forms. The simulation models this using a multi-step process within the `_resolve_collision_jit` function. Collisions are **inelastic**, governed by the `coefficient_of_restitution` in `config.json`.

The collision resolution process is as follows:

1.  **Overlap Resolution:** When two particles are found to be overlapping, they are first pushed apart along the collision normal until their edges are just touching. This positional correction is weighted by mass to conserve the center of mass of the pair.
2.  **Potential Energy Accounting:** Pushing the particles apart changes their distance, which in turn changes the gravitational potential energy of the pair. This `pe_change` is carefully calculated. If particles are pushed apart, potential energy increases (becomes less negative).
3.  **Inelastic Velocity Update:** The particle velocities are then updated. The component of their relative velocity along the collision normal is reflected and scaled by the `coefficient_of_restitution`. A value of 1.0 would be perfectly elastic, while a value of 0.0 would be perfectly inelastic.
4.  **Kinetic to Thermal Energy Conversion:** Because the collision is inelastic (restitution < 1.0), kinetic energy is not conserved. The total kinetic energy of the pair before and after the velocity update is measured, and the difference is calculated as `ke_lost`.
5.  **Net Heat Generation:** The core principle of energy conservation within a collision is `heat_generated = ke_lost - pe_change`. The energy "cost" of pushing particles apart (the potential energy gain) is paid for by the kinetic energy lost during the inelastic impact. This net energy change is converted into thermal energy and distributed between the two colliding particles, increasing their temperature.

### 4. System-Wide Energy Conservation

While the collision model is carefully designed to conserve energy and the Velocity Verlet integrator is stable, the discrete nature of the simulation (and the approximation of using a spatial grid) can lead to small numerical errors that cause the total system energy to drift over time.

The simulation actively corrects for this drift:
1.  The total energy of the system (Kinetic + Potential + Thermal) is measured before and after each physics tick.
2.  The difference (`total_delta_this_tick`) represents the numerical integration error for that step.
3.  This error is accumulated over 100 ticks.
4.  Every 100 ticks, a `thermal_correction` is applied. The exact amount of energy that was artificially gained or lost due to numerical error is subtracted from or added to the system's total thermal energy, ensuring the simulation's total energy remains constant over long periods.

## Computational Optimizations for Performance

A naive O(n²) implementation would be too slow. This simulation uses several techniques to achieve high performance.

### 1. Barnes-Hut Algorithm for Gravity

The primary optimization for the N-body gravity calculation is the **Barnes-Hut algorithm**, implemented using a **QuadTree**. By treating distant particle clusters as single points of mass, it reduces the complexity of calculating gravitational forces from O(n²) to O(n log n), enabling the simulation of long-range interactions that would otherwise be computationally prohibitive.

### 2. Spatial Grid for Broad-Phase Collision Detection

For discrete collisions, which are short-range interactions, the simulation space is partitioned into a uniform grid. The cell size is determined by a heuristic based on the maximum particle radius and a `grid_cell_size_multiplier` in the config. When checking for collisions, a particle only interacts with other particles in its own grid cell and its 8 immediate neighbors. This "broad-phase" check drastically reduces the number of pairs that need to be considered for a direct collision check, lowering the complexity from O(n²) to approximately O(n\*k), where *k* is the average number of particles in the local neighborhood.

### 2. Just-in-Time (JIT) Compilation with Numba

The most intensive calculations—the loops that compute gravity and resolve collisions for pairs of particles—are implemented in dedicated functions decorated with `@numba.jit(nopython=True)`. Numba compiles these Python functions into highly optimized machine code at runtime, bypassing the Python interpreter's overhead and yielding performance comparable to C or Fortran.

### 3. Vectorization with NumPy

Particle data (positions, velocities, masses, etc.) is stored in NumPy arrays. This allows for vectorized operations that apply to all particles simultaneously, which is significantly faster than iterating in Python loops. This is used for tasks like the Velocity Verlet integration, boundary collisions, and calculating particle colors from their temperatures.

## How to Run the Simulation

1.  **Prerequisites:**
    *   Python 3.x
    *   Pygame
    *   NumPy
    *   Numba

2.  **Configuration:**
    *   `constants.py`: For application-level settings like screen resolution (`WIDTH`, `HEIGHT`) and target framerate (`FPS`).
    *   `config.json`: To change simulation-specific parameters. Key parameters include:
        *   `particle_count`: The number of particles to simulate.
        *   `gravity_constant`: The strength of gravity.
        *   `softening_factor`: Prevents extreme forces at close range.
        *   `barnes_hut_theta`: Controls the accuracy of the Barnes-Hut approximation. A lower value (e.g., 0.3) is more accurate but slower. A higher value (e.g., 0.8) is faster but less accurate.
        *   `coefficient_of_restitution`: Controls the "bounciness" of collisions (0.0 to 1.0).
        *   `grid_cell_size_multiplier`: Adjusts the size of the spatial grid cells for collision detection performance tuning.

3.  **Execution:**
    *   Run the main script from your terminal:
        ```bash
        python main.py
        ```