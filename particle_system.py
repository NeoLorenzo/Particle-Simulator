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

        logging.info(f"ParticleSystem created for {num_particles} particles.")

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
        Calculates gravitational forces between all pairs of particles.
        This is a vectorized implementation of the O(n^2) calculation.
        """
        # Reset accelerations
        self.accelerations.fill(0.0)

        # Calculate pairwise differences in position (broadcasting)
        # p1_pos is (N, 1, 2), p2_pos is (1, N, 2) -> diffs is (N, N, 2)
        diffs = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]

        # Calculate squared distances
        dist_sq = np.sum(diffs**2, axis=2)

        # Add softening factor to avoid division by zero and instability
        dist_sq += self.config['softening_factor']

        # Calculate force magnitude: F = G * (m1*m2) / r^2
        # masses is (N, 1), masses.T is (1, N) -> mass_product is (N, N)
        mass_product = self.masses @ self.masses.T
        force_magnitudes = (self.config['gravity_constant'] * mass_product) / dist_sq

        # Calculate force vectors
        # We need to reshape force_magnitudes to (N, N, 1) for broadcasting
        force_vectors = force_magnitudes[:, :, np.newaxis] * diffs / np.sqrt(dist_sq)[:, :, np.newaxis]

        # Sum forces for each particle and calculate acceleration (a = F/m)
        # np.nansum is used to handle the diagonal (particle attracting itself results in NaN)
        total_force = np.nansum(force_vectors, axis=1)
        self.accelerations = total_force / self.masses

    def _handle_collisions(self):
        """
        Detects and resolves collisions between all pairs of particles.
        This is a partially vectorized implementation. The pair-finding is still
        a loop, but the calculations are NumPy-based.
        """
        # This part is still O(n^2) but difficult to fully vectorize due to the
        # sparse and conditional nature of collisions. Future optimizations could
        # use spatial hashing (e.g., a grid) to reduce pair checks.
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                pos_i, pos_j = self.positions[i], self.positions[j]
                rad_i, rad_j = self.radii[i][0], self.radii[j][0]

                distance_vec = pos_j - pos_i
                distance = np.linalg.norm(distance_vec)
                min_distance = rad_i + rad_j

                if distance < min_distance:
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
                    logging.debug(f"Collision heat transfer: T1_before={t1_before[0]:.2f} T2_before={t2_before[0]:.2f} -> T1_after={self.temperatures[i][0]:.2f} T2_after={self.temperatures[j][0]:.2f}")

                    # 3. Elastic Collision Response
                    normal = distance_vec / distance
                    v_rel = self.velocities[j] - self.velocities[i]
                    m_i, m_j = self.masses[i], self.masses[j]
                    impulse_j = (-2 * m_i * m_j * np.dot(v_rel, normal)) / (m_i + m_j)

                    self.velocities[i] -= (impulse_j / m_i) * normal
                    self.velocities[j] += (impulse_j / m_j) * normal

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
        """
        self._calculate_gravity()
        self._handle_collisions() # Note: This contains the O(n^2) loop

        # Update velocities and positions (Euler integration)
        self.velocities += self.accelerations
        self.positions += self.velocities

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