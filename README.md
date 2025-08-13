# Particle Simulator

This project is a high-performance 2D particle simulator written in Python. It models the behavior of a large number of particles interacting through gravity and direct collisions. The simulation is built with a focus on physical realism, computational efficiency, and modular design.

It leverages NumPy for vectorized calculations, Numba for just-in-time (JIT) compilation of performance-critical physics loops, and Pygame for visualization.

## Core Physics Principles

The simulation is grounded in several key physics principles to achieve realistic emergent behavior.

### 1. Gravitational Interaction (N-Body Simulation)

At its core, the simulator solves the gravitational N-body problem. In principle, every particle in the system exerts a gravitational force on every other particle. The force between any two particles is calculated using Newton's Law of Universal Gravitation:

**F = G * (m1 * m2) / r²**

Where:
- **F** is the magnitude of the gravitational force.
- **G** is the gravitational constant (a configurable parameter).
- **m1** and **m2** are the masses of the two particles.
- **r²** is the square of the distance between them.

From this force, we derive the acceleration (**a = F/m**) for each particle, which dictates its movement. A "softening factor" is added to the distance calculation to prevent the force from becoming infinite when particles get extremely close, which is a common technique in N-body simulations to avoid numerical instability.

### 2. Collision Physics and Heat Transfer

The simulation implements elastic collisions between particles. When two particles' distance is less than the sum of their radii, a collision is detected. The resolution involves two main parts:

*   **Elastic Collision Response:** The simulation calculates the change in velocity for each colliding particle to conserve both momentum and kinetic energy. This ensures that particles bounce off each other realistically.
*   **Conductive Heat Transfer:** During a collision, thermal energy is exchanged between the particles. This process, known as thermal conduction, depends on the temperature difference between the colliding particles. The amount of heat transferred is governed by a `heat_transfer_coefficient`, and the temperature of each particle changes based on its mass and the amount of heat it gains or loses.

### 3. Numerical Integration (Velocity Verlet)

To update the positions and velocities of particles over discrete time steps, the simulation employs the **Velocity Verlet** integration method. This technique is superior to simpler methods like Euler integration because it offers better energy conservation over long periods, which is crucial for the stability of a physics-based simulation.

The process for each time step is:
1.  **Update Positions:** Calculate the new positions of particles based on their current velocities and accelerations.
2.  **Calculate New Forces:** Determine the new forces (and thus new accelerations) acting on the particles in their new positions.
3.  **Update Velocities:** Calculate the new velocities using an average of the old and new accelerations.

This method is time-reversible and symplectic, meaning it preserves key properties of the physical system being modeled, leading to more stable and accurate long-term simulations.

## Computational Optimizations for Performance

A naive implementation that calculates interactions between every pair of particles would have a computational complexity of O(n²), making it too slow for a large number of particles. This simulation uses several advanced techniques to achieve high performance.

### 1. Spatial Grid for Broad-Phase Collision Detection

To avoid the O(n²) complexity, the simulation space is partitioned into a uniform grid. Each particle is placed into a grid cell based on its position. When checking for gravity and collisions for a given particle, the search is limited to particles in the same cell and its immediate neighbors. This drastically reduces the number of pairs that need to be checked, changing the complexity to be closer to O(n*k), where *k* is the average number of particles in the neighboring cells.

### 2. Just-in-Time (JIT) Compilation with Numba

The most computationally intensive parts of the simulation—the loops that calculate gravity and resolve collisions for pairs of particles—are written as separate functions and compiled to highly optimized machine code at runtime using the Numba library.

By applying the `@numba.jit(nopython=True)` decorator, these Python functions are transformed to a much faster implementation that operates directly on NumPy array data, bypassing the overhead of the Python interpreter. This provides a massive speedup, making it feasible to simulate thousands of particles in real-time.

### 3. Vectorization with NumPy

NumPy is used extensively to store particle properties (positions, velocities, masses, etc.) in contiguous arrays. Operations that can be applied to all particles at once, such as updating positions from velocities, handling boundary collisions, or calculating colors from temperatures, are performed using vectorized NumPy functions. This is significantly faster than iterating through particles in a standard Python loop.

## How to Run the Simulation

1.  **Prerequisites:**
    *   Python 3.x
    *   Pygame
    *   NumPy
    *   Numba

2.  **Configuration:**
    *   Modify `constants.py` for application-level settings like screen resolution and FPS.
    *   Modify `config.json` to change simulation-specific parameters like the number of particles, gravity strength, and particle properties.

3.  **Execution:**
    *   Run the main script from your terminal:
        ```bash
        python main.py
        ```