# particle_system.py

import numpy as np
import pygame
import logging
import numba

logger = logging.getLogger("particle_sim")

# --- JIT-Compiled Physics Functions ---
# These functions are compiled to machine code by Numba for maximum performance.
# They are deliberately kept outside the ParticleSystem class and operate only on
# NumPy arrays and simple scalar values, as required by Numba's nopython mode.

@numba.jit(nopython=True)
def _calculate_gravity_for_pair_jit(i, j, positions, masses, accelerations, g_const, softening):
    """
    Numba-accelerated gravity calculation for a single pair of particles.
    Modifies the 'accelerations' array in place.
    """
    pos_i, pos_j = positions[i], positions[j]
    mass_i, mass_j = masses[i][0], masses[j][0] # Access scalar from (1,) array

    diff = pos_j - pos_i
    dist_sq = np.sum(diff**2) + softening

    # Calculate force magnitude: F = G * (m1*m2) / r^2
    force_magnitude = (g_const * mass_i * mass_j) / dist_sq

    # Calculate force vector
    dist = np.sqrt(dist_sq)
    force_vector = force_magnitude * (diff / dist)

    # Apply acceleration (a = F/m) to both particles
    accelerations[i] += force_vector / mass_i
    accelerations[j] -= force_vector / mass_j

@numba.jit(nopython=True)
def _resolve_collision_jit(i, j, positions, velocities, masses, radii, temperatures, heat_coeff, restitution):
    """
    Numba-accelerated inelastic collision and energy conversion for a single pair.
    Modifies positions, velocities, and temperatures in place.
    """
    pos_i, pos_j = positions[i], positions[j]
    rad_i, rad_j = radii[i][0], radii[j][0]

    distance_vec = pos_j - pos_i
    distance_sq = np.sum(distance_vec**2)
    min_distance = rad_i + rad_j

    if distance_sq < min_distance**2 and distance_sq > 0:
        distance = np.sqrt(distance_sq)
        m_i, m_j = masses[i][0], masses[j][0]

        # --- Store pre-collision state for energy calculation ---
        v_i_before = velocities[i].copy()
        v_j_before = velocities[j].copy()
        ke_before = 0.5 * m_i * np.sum(v_i_before**2) + 0.5 * m_j * np.sum(v_j_before**2)

        # 1. Resolve overlap
        overlap = min_distance - distance
        correction = 0.5 * overlap * (distance_vec / distance)
        positions[i] -= correction
        positions[j] += correction

        # 2. Inelastic Collision Response
        normal = distance_vec / distance
        v_rel = v_j_before - v_i_before
        v_rel_dot_normal = v_rel[0] * normal[0] + v_rel[1] * normal[1]

        # Impulse formula now includes the coefficient of restitution
        impulse_j = (-(1 + restitution) * m_i * m_j * v_rel_dot_normal) / (m_i + m_j)

        velocities[i] -= (impulse_j / m_i) * normal
        velocities[j] += (impulse_j / m_j) * normal

        # 3. Convert Lost Kinetic Energy to Thermal Energy
        ke_after = 0.5 * m_i * np.sum(velocities[i]**2) + 0.5 * m_j * np.sum(velocities[j]**2)
        ke_lost = ke_before - ke_after

        # The lost kinetic energy is converted to heat.
        # We distribute this new heat between the two particles.
        # A simple 50/50 split is a reasonable starting abstraction.
        if ke_lost > 0:
            heat_generated = ke_lost * heat_coeff # Use coeff to scale conversion
            temperatures[i][0] += (heat_generated * 0.5) / m_i
            temperatures[j][0] += (heat_generated * 0.5) / m_j


class ParticleSystem:
    """
    Manages the state and physics of all particles in the simulation using
    vectorized NumPy operations for high performance.

    Data Contract:
    - Inputs:
        - num_particles (int): The number of particles to simulate.
        - config (dict): The 'simulation' section of the config file.
        - rng (np.random.Generator): The master seeded random number generator.
        - bounds (tuple): The (width, height) of the simulation area.
    - Outputs: None. This class modifies its internal state.
    - Side Effects: Manages the lifecycle of all particle data.
    - Invariants: The number of particles is constant throughout the simulation.
      All internal arrays must maintain the same length (num_particles).
    """
    def __init__(self, num_particles: int, config: dict, rng: np.random.Generator, bounds: tuple):
        self.num_particles = num_particles
        self.config = config
        self.bounds = np.array(bounds)
        self.restitution = config['coefficient_of_restitution']

        # --- Initialize properties using NumPy arrays (Structure of Arrays) ---
        self.positions = rng.random((num_particles, 2)) * self.bounds
        self.velocities = np.zeros((num_particles, 2), dtype=float)
        self.accelerations = np.zeros((num_particles, 2), dtype=float)
        self.masses = rng.uniform(config['min_mass'], config['max_mass'], (num_particles, 1))
        self.temperatures = rng.uniform(config['min_temp'], config['max_temp'], (num_particles, 1))
        self.radii = np.sqrt(self.masses).astype(int)

        # --- Spatial Grid Optimization ---
        # Heuristic: Cell size should be at least as large as the largest possible particle's diameter
        # to ensure any two colliding particles can be in adjacent cells.
        max_radius = np.sqrt(config['max_mass']).astype(int)
        self.cell_size = max_radius * 2
        if self.cell_size == 0: # Avoid division by zero if masses are tiny
            self.cell_size = 10 # A reasonable default
        self.grid_width = int(np.ceil(self.bounds[0] / self.cell_size))
        self.grid_height = int(np.ceil(self.bounds[1] / self.cell_size))
        self.grid = [[] for _ in range(self.grid_width * self.grid_height)]

        logger.info(f"ParticleSystem created for {num_particles} particles.")
        logger.info(f"Spatial grid initialized with cell size {self.cell_size} ({self.grid_width}x{self.grid_height} cells).")

    def _get_colors(self):
        """
        Vectorized color calculation based on temperature.
        Maps temperature from a min/max range to a Red -> Yellow -> White gradient.
        """
        # Normalize temperature to a 0-1 range
        norm_temp = (self.temperatures - self.config['min_temp']) / (self.config['max_temp'] - self.config['min_temp'])
        norm_temp = np.clip(norm_temp, 0, 1)

        # Create color array
        colors = np.zeros((self.num_particles, 3), dtype=int)

        # Red to Yellow (norm_temp < 0.5)
        mask1 = norm_temp < 0.5
        t1 = norm_temp[mask1] * 2
        colors[mask1.flatten()] = np.column_stack([
            np.full(t1.shape, 255),
            (t1 * 255).astype(int),
            np.zeros(t1.shape)
        ])

        # Yellow to White (norm_temp >= 0.5)
        mask2 = norm_temp >= 0.5
        t2 = (norm_temp[mask2] - 0.5) * 2
        colors[mask2.flatten()] = np.column_stack([
            np.full(t2.shape, 255),
            np.full(t2.shape, 255),
            (t2 * 255).astype(int)
        ])
        return colors

    def _calculate_gravity(self):
        """
        Calculates gravitational forces using the spatial grid to limit calculations
        to nearby particles. This is a scientifically-grounded abstraction (Rule 8)
        that trades the accuracy of long-range forces for a significant performance
        increase. It approximates a full n-body simulation by considering only
        local interactions, which is a valid approach when emergent behavior is
        driven by dense clusters. The complexity is reduced from O(n^2) to
        roughly O(n*k), where k is the average number of particles in neighboring cells.
        The core calculation is delegated to a Numba JIT-compiled function.
        """
        self.accelerations.fill(0.0)
        g_const = self.config['gravity_constant']
        softening = self.config['softening_factor']

        for cell_idx, cell in enumerate(self.grid):
            cell_x = cell_idx % self.grid_width
            cell_y = cell_idx // self.grid_width

            # 1. Calculate gravity within the cell itself
            for i in range(len(cell)):
                for j in range(i + 1, len(cell)):
                    _calculate_gravity_for_pair_jit(
                        cell[i], cell[j], self.positions, self.masses,
                        self.accelerations, g_const, softening
                    )

            # 2. Calculate gravity with 4 neighboring cells to avoid double-counting.
            neighbor_indices = []
            if cell_x < self.grid_width - 1: # Right
                neighbor_indices.append(cell_idx + 1)
            if cell_y < self.grid_height - 1: # Below
                neighbor_indices.append(cell_idx + self.grid_width)
            if cell_x > 0 and cell_y < self.grid_height - 1: # Bottom-left
                neighbor_indices.append(cell_idx + self.grid_width - 1)
            if cell_x < self.grid_width - 1 and cell_y < self.grid_height - 1: # Bottom-right
                neighbor_indices.append(cell_idx + self.grid_width + 1)

            for neighbor_idx in neighbor_indices:
                neighbor_cell = self.grid[neighbor_idx]
                for p1_index in cell:
                    for p2_index in neighbor_cell:
                        _calculate_gravity_for_pair_jit(
                            p1_index, p2_index, self.positions, self.masses,
                            self.accelerations, g_const, softening
                        )

    def _handle_collisions(self):
        """
        Detects and resolves collisions using the pre-built spatial grid.
        This avoids the O(n^2) check of all possible pairs by only checking
        particles in the same or adjacent grid cells.
        The core calculation is delegated to a Numba JIT-compiled function.
        """
        heat_coeff = self.config['heat_transfer_coefficient']

        for cell_idx, cell in enumerate(self.grid):
            cell_x = cell_idx % self.grid_width
            cell_y = cell_idx // self.grid_width

            # 1. Check for collisions within the cell itself
            for i in range(len(cell)):
                for j in range(i + 1, len(cell)):
                    _resolve_collision_jit(
                        cell[i], cell[j], self.positions, self.velocities,
                        self.masses, self.radii, self.temperatures, heat_coeff, self.restitution
                    )

            # 2. Check for collisions with 4 neighboring cells to avoid double-counting pairs.
            neighbor_indices = []
            if cell_x < self.grid_width - 1:
                neighbor_indices.append(cell_idx + 1)
            if cell_y < self.grid_height - 1:
                neighbor_indices.append(cell_idx + self.grid_width)
            if cell_x > 0 and cell_y < self.grid_height - 1:
                neighbor_indices.append(cell_idx + self.grid_width - 1)
            if cell_x < self.grid_width - 1 and cell_y < self.grid_height - 1:
                neighbor_indices.append(cell_idx + self.grid_width + 1)

            for neighbor_idx in neighbor_indices:
                neighbor_cell = self.grid[neighbor_idx]
                for p1_index in cell:
                    for p2_index in neighbor_cell:
                        _resolve_collision_jit(
                            p1_index, p2_index, self.positions, self.velocities,
                            self.masses, self.radii, self.temperatures, heat_coeff, self.restitution
                        )

    def _check_boundary_collisions(self):
        """
        Vectorized boundary collision check.
        """
        # Check left/right boundaries
        left_mask = self.positions[:, 0] - self.radii.flatten() < 0
        right_mask = self.positions[:, 0] + self.radii.flatten() > self.bounds[0]
        self.positions[left_mask, 0] = self.radii[left_mask].flatten()
        self.positions[right_mask, 0] = self.bounds[0] - self.radii[right_mask].flatten()
        self.velocities[left_mask | right_mask, 0] *= -1

        # Check top/bottom boundaries
        top_mask = self.positions[:, 1] - self.radii.flatten() < 0
        bottom_mask = self.positions[:, 1] + self.radii.flatten() > self.bounds[1]
        self.positions[top_mask, 1] = self.radii[top_mask].flatten()
        self.positions[bottom_mask, 1] = self.bounds[1] - self.radii[bottom_mask].flatten()
        self.velocities[top_mask | bottom_mask, 1] *= -1

    def update(self):
        """
        Runs a full physics update step using Velocity Verlet integration.
        This method conserves energy better than simple Euler integration.
        """
        # 1. Update positions using current velocity and acceleration (t).
        # This is the first half of Velocity Verlet: p(t+dt) = p(t) + v(t)dt + 0.5a(t)dt^2
        # We assume dt=1 tick, so dt and dt^2 are both 1.
        self.positions += self.velocities + 0.5 * self.accelerations

        # 2. Store the acceleration from the previous step (t) before recalculating.
        old_accelerations = np.copy(self.accelerations)

        # 3. Build the spatial grid with the new positions to find new neighbors.
        self._build_spatial_grid()

        # 4. Calculate new forces based on new positions to get acceleration (t+1).
        self._calculate_gravity() # This resets and calculates new accelerations.
        self._handle_collisions() # This can also modify accelerations.

        # 5. Update velocities using the average of old and new accelerations.
        # This is the second half of Velocity Verlet: v(t+dt) = v(t) + 0.5 * (a(t) + a(t+dt)) * dt
        self.velocities += 0.5 * (old_accelerations + self.accelerations)

        # 6. Handle interactions with the simulation boundaries after all movement.
        self._check_boundary_collisions()

    def draw(self, screen: pygame.Surface):
        """
        Draws all particles on the screen.
        """
        colors = self._get_colors()
        for i in range(self.num_particles):
            pygame.draw.circle(
                screen,
                colors[i],
                self.positions[i].astype(int),
                self.radii[i][0]
            )

    def _build_spatial_grid(self):
        """
        Populates a spatial grid to accelerate collision detection.

        Assigns each particle to a grid cell based on its position. This is an
        O(n) operation that allows subsequent neighbor searches to be much faster
        than the naive O(n^2) approach.
        """
        # Clear the grid for the new frame
        for cell in self.grid:
            cell.clear()

        # Place particle indices into grid cells
        for i in range(self.num_particles):
            pos = self.positions[i]
            cell_x = int(pos[0] / self.cell_size)
            cell_y = int(pos[1] / self.cell_size)

            # Clamp values to be within grid bounds
            cell_x = max(0, min(cell_x, self.grid_width - 1))
            cell_y = max(0, min(cell_y, self.grid_height - 1))

            grid_index = cell_y * self.grid_width + cell_x
            self.grid[grid_index].append(i)

    def get_total_kinetic_energy(self):
        """
        Calculates the total kinetic energy of the system.
        KE = sum(0.5 * m * v^2)
        """
        # For a 2D velocity vector v = (vx, vy), v^2 = vx^2 + vy^2.
        # np.sum(axis=1) calculates this for each particle.
        vel_sq = np.sum(self.velocities**2, axis=1, keepdims=True)
        kinetic_energy = 0.5 * self.masses * vel_sq
        return np.sum(kinetic_energy)

    def get_total_thermal_energy(self):
        """
        Calculates the total thermal energy of the system.
        Assumes thermal energy is proportional to temperature * mass.
        This is an abstraction; in reality it depends on specific heat capacity.
        """
        # E_thermal = sum(mass * temperature)
        return np.sum(self.masses * self.temperatures)