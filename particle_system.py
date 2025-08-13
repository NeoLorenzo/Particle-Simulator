# particle_system.py

import numpy as np
import pygame
import logging

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

        logging.info(f"ParticleSystem created for {num_particles} particles.")
        logging.info(f"Spatial grid initialized with cell size {self.cell_size} ({self.grid_width}x{self.grid_height} cells).")

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
        """
        self.accelerations.fill(0.0)

        for cell_idx, cell in enumerate(self.grid):
            cell_x = cell_idx % self.grid_width
            cell_y = cell_idx // self.grid_width

            # 1. Calculate gravity within the cell itself
            for i in range(len(cell)):
                for j in range(i + 1, len(cell)):
                    self._calculate_gravity_for_pair(cell[i], cell[j])

            # 2. Calculate gravity with 4 neighboring cells to avoid double-counting.
            # This pattern ensures each pair of adjacent cells is checked only once.
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
                # Check all pairs between the current cell and the neighbor cell
                for p1_index in cell:
                    for p2_index in neighbor_cell:
                        self._calculate_gravity_for_pair(p1_index, p2_index)

    def _handle_collisions(self):
        """
        Detects and resolves collisions using the pre-built spatial grid.
        This avoids the O(n^2) check of all possible pairs by only checking
        particles in the same or adjacent grid cells.
        """
        # The spatial grid is now built in the main update() loop.

        for cell_idx, cell in enumerate(self.grid):
            cell_x = cell_idx % self.grid_width
            cell_y = cell_idx // self.grid_width

            # 1. Check for collisions within the cell itself
            for i in range(len(cell)):
                for j in range(i + 1, len(cell)):
                    self._resolve_collision(cell[i], cell[j])

            # 2. Check for collisions with 4 neighboring cells to avoid double-counting pairs.
            # This pattern ensures each pair of adjacent cells is checked only once.
            neighbor_indices = []
            # Neighbor to the right
            if cell_x < self.grid_width - 1:
                neighbor_indices.append(cell_idx + 1)
            # Neighbor below
            if cell_y < self.grid_height - 1:
                neighbor_indices.append(cell_idx + self.grid_width)
            # Neighbor to the bottom-left
            if cell_x > 0 and cell_y < self.grid_height - 1:
                neighbor_indices.append(cell_idx + self.grid_width - 1)
            # Neighbor to the bottom-right
            if cell_x < self.grid_width - 1 and cell_y < self.grid_height - 1:
                neighbor_indices.append(cell_idx + self.grid_width + 1)

            for neighbor_idx in neighbor_indices:
                self._check_cell_pairs(cell, self.grid[neighbor_idx])

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
        Runs a full physics update step for the entire system.
        The order is optimized to build the spatial grid once and reuse it.
        """
        # 1. Build the spatial grid to accelerate neighbor-finding.
        self._build_spatial_grid()

        # 2. Calculate forces.
        self._calculate_gravity() # Uses the pre-built grid.
        self._handle_collisions() # Uses the pre-built grid.

        # 3. Update velocities and positions (Euler integration).
        self.velocities += self.accelerations
        self.positions += self.velocities

        # 4. Handle interactions with the simulation boundaries.
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

    def _check_cell_pairs(self, cell1: list, cell2: list):
        """Helper to check all particle pairs between two cells."""
        for p1_index in cell1:
            for p2_index in cell2:
                self._resolve_collision(p1_index, p2_index)

    def _resolve_collision(self, i: int, j: int):
        """
        Checks and resolves a potential collision between two particles, i and j.
        This contains the original collision logic, now called from the new grid system.
        """
        pos_i, pos_j = self.positions[i], self.positions[j]
        rad_i, rad_j = self.radii[i][0], self.radii[j][0]

        distance_vec = pos_j - pos_i
        # Use squared distance for the initial check to avoid an expensive sqrt
        distance_sq = np.sum(distance_vec**2)
        min_distance = rad_i + rad_j

        if distance_sq < min_distance**2 and distance_sq > 0:
            distance = np.sqrt(distance_sq) # Only calculate sqrt when a collision is likely
            # --- Collision Detected ---
            # 1. Resolve overlap
            overlap = min_distance - distance
            correction = 0.5 * overlap * (distance_vec / distance)
            self.positions[i] -= correction
            self.positions[j] += correction

            # 2. Heat Transfer
            t1_before, t2_before = self.temperatures[i], self.temperatures[j]
            temp_diff = t1_before - t2_before
            heat_transfer = self.config['heat_transfer_coefficient'] * temp_diff
            self.temperatures[i] -= heat_transfer / self.masses[i]
            self.temperatures[j] += heat_transfer / self.masses[j]
            # The following log is throttled by commenting out, per Rule 2.4, as it's in a hot loop.
            # logging.debug(f"Collision heat transfer: T1_before={t1_before[0]:.2f} T2_before={t2_before[0]:.2f} -> T1_after={self.temperatures[i][0]:.2f} T2_after={self.temperatures[j][0]:.2f}")

            # 3. Elastic Collision Response
            normal = distance_vec / distance
            v_rel = self.velocities[j] - self.velocities[i]
            m_i, m_j = self.masses[i], self.masses[j]
            impulse_j = (-2 * m_i * m_j * np.dot(v_rel, normal)) / (m_i + m_j)

            self.velocities[i] -= (impulse_j / m_i) * normal
            self.velocities[j] += (impulse_j / m_j) * normal

    def _calculate_gravity_for_pair(self, i: int, j: int):
        """
        Calculates and applies the gravitational force between two particles, i and j.
        The force is applied symmetrically to both particles.
        """
        pos_i, pos_j = self.positions[i], self.positions[j]
        mass_i, mass_j = self.masses[i], self.masses[j]

        diff = pos_j - pos_i
        dist_sq = np.sum(diff**2)

        # Avoid division by zero and instability at close ranges
        dist_sq += self.config['softening_factor']

        # Calculate force magnitude: F = G * (m1*m2) / r^2
        force_magnitude = (self.config['gravity_constant'] * mass_i * mass_j) / dist_sq

        # Calculate force vector
        dist = np.sqrt(dist_sq)
        force_vector = force_magnitude * (diff / dist)

        # Apply acceleration (a = F/m) to both particles
        self.accelerations[i] += force_vector / mass_i
        self.accelerations[j] -= force_vector / mass_j