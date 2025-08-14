# particle_system.py

import numpy as np
import pygame
import logging
import numba
from quadtree import QuadTree, BoundingBox

logger = logging.getLogger("particle_sim")

# --- JIT-Compiled Physics Functions ---
# These functions are compiled to machine code by Numba for maximum performance.
# They are deliberately kept outside the ParticleSystem class and operate only on
# NumPy arrays and simple scalar values, as required by Numba's nopython mode.

@numba.jit(nopython=True)
def _calculate_gravity_barnes_hut_jit(num_particles, positions, masses, accelerations, g_const, softening, theta, node_data, node_children, node_particles_map, particle_list):
    """
    Numba-accelerated main loop for Barnes-Hut gravity calculation.
    Iterates through each particle and initiates the tree traversal.
    """
    for i in range(num_particles):
        net_force = _traverse_tree_jit(i, 0, positions, masses, g_const, softening, theta, node_data, node_children, node_particles_map, particle_list)
        accelerations[i] = net_force / masses[i][0]

@numba.jit(nopython=True)
def _calculate_potential_energy_jit(node_idx, node_data, node_children, g_const, softening):
    """
    Recursively calculates the total potential energy of the system using the
    flattened Barnes-Hut tree, ensuring consistency with the force model.
    This avoids double-counting by only considering interactions within a node
    or between a node and its children.
    """
    total_pe = 0.0
    node_mass, com_x, com_y, width = node_data[node_idx]
    is_leaf = node_children[node_idx, 0] == -1

    if not is_leaf:
        # It's an internal node.
        # The PE of this node is the sum of the PEs of its children, plus the
        # interaction energy between all pairs of its children.
        for i in range(4):
            child_idx_i = node_children[node_idx, i]
            if child_idx_i != -1 and node_data[child_idx_i][0] > 0:
                # 1. Add the internal PE of the child by recursing.
                total_pe += _calculate_potential_energy_jit(child_idx_i, node_data, node_children, g_const, softening)

                # 2. Add the interaction PE between this child and its siblings.
                # To avoid double counting, we only pair (i) with (j) where j > i.
                for j in range(i + 1, 4):
                    child_idx_j = node_children[node_idx, j]
                    if child_idx_j != -1 and node_data[child_idx_j][0] > 0:
                        mass_i = node_data[child_idx_i][0]
                        mass_j = node_data[child_idx_j][0]
                        com_i = np.array([node_data[child_idx_i][1], node_data[child_idx_i][2]])
                        com_j = np.array([node_data[child_idx_j][1], node_data[child_idx_j][2]])

                        diff = com_j - com_i
                        dist_sq = np.sum(diff**2)
                        if dist_sq > 0:
                            dist = np.sqrt(dist_sq + softening)
                            total_pe -= (g_const * mass_i * mass_j) / dist
    # Note: For leaf nodes, the PE is considered to be zero at this level.
    # Their contribution is handled when their parent node calculates the
    # interaction energy between its children.
    return total_pe

@numba.jit(nopython=True)
def _traverse_tree_jit(p_idx, node_idx, positions, masses, g_const, softening, theta, node_data, node_children, node_particles_map, particle_list):
    """
    Numba-accelerated recursive traversal of the flattened QuadTree arrays.
    This function calculates the net force on a single particle (p_idx).
    """
    node_mass, com_x, com_y, width = node_data[node_idx]
    net_force = np.zeros(2)

    # Check if the node is internal or a leaf by looking at its children pointers
    is_leaf = node_children[node_idx, 0] == -1

    if not is_leaf:
        # It's an internal node
        p_pos = positions[p_idx]
        node_com = np.array([com_x, com_y])
        diff = p_pos - node_com
        dist = np.sqrt(np.sum(diff**2))

        if dist > 0 and (width / dist) < theta:
            # Node is far enough away, treat as a single body
            force_vec = (g_const * masses[p_idx][0] * node_mass * diff) / (dist**3 + softening)
            return -force_vec # Return force on the particle
        else:
            # Node is too close, recurse into children
            for child_node_idx in node_children[node_idx]:
                if child_node_idx != -1 and node_data[child_node_idx][0] > 0:
                    net_force += _traverse_tree_jit(p_idx, child_node_idx, positions, masses, g_const, softening, theta, node_data, node_children, node_particles_map, particle_list)
            return net_force
    else:
        # It's a leaf node, calculate direct forces from its particles
        start, count, _ = node_particles_map[node_idx]
        if count > 0:
            for i in range(count):
                other_p_idx = particle_list[start + i]
                if p_idx != other_p_idx:
                    p1_pos = positions[p_idx]
                    p2_pos = positions[other_p_idx]
                    diff = p2_pos - p1_pos
                    dist_sq = np.sum(diff**2)
                    if dist_sq > 0:
                        force_mag = (g_const * masses[p_idx][0] * masses[other_p_idx][0]) / (dist_sq + softening)
                        dist = np.sqrt(dist_sq)
                        net_force += force_mag * (diff / dist)
        return net_force

@numba.jit(nopython=True)
def _calculate_potential_energy_for_pair_jit(i, j, positions, masses, g_const, softening):
    """Calculates the softened gravitational potential energy for a single pair."""
    pos_i, pos_j = positions[i], positions[j]
    mass_i, mass_j = masses[i][0], masses[j][0]

    diff = pos_j - pos_i
    dist_sq = np.sum(diff**2)

    # Potential energy U = -G * (m1*m2) / r
    # We use the softened distance sqrt(dist_sq + softening) to match the force calculation
    potential = (-g_const * mass_i * mass_j) / np.sqrt(dist_sq + softening)
    return potential

@numba.jit(nopython=True)
def _handle_collisions_jit(grid_indices, grid_offsets, grid_width, grid_height, positions, velocities, masses, radii, temperatures, heat_coeff, restitution, g_const, softening):
    """
    Numba-accelerated broad-phase and narrow-phase collision detection and resolution.
    Iterates through the spatial grid and resolves collisions for neighboring particles.
    """
    total_pe_change = 0.0
    total_ke_lost = 0.0
    num_cells = grid_width * grid_height

    for cell_idx in range(num_cells):
        cell_x = cell_idx % grid_width
        cell_y = cell_idx // grid_width

        start_idx = grid_offsets[cell_idx]
        end_idx = grid_offsets[cell_idx + 1]

        # 1. Check for collisions within the cell itself
        for i in range(start_idx, end_idx):
            p1_index = grid_indices[i]
            for j in range(i + 1, end_idx):
                p2_index = grid_indices[j]
                pe_change, ke_lost = _resolve_collision_jit(
                    p1_index, p2_index, positions, velocities,
                    masses, radii, temperatures, heat_coeff, restitution,
                    g_const, softening
                )
                total_pe_change += pe_change
                total_ke_lost += ke_lost

        # 2. Check for collisions with neighboring cells
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                neighbor_x, neighbor_y = cell_x + dx, cell_y + dy

                if 0 <= neighbor_x < grid_width and 0 <= neighbor_y < grid_height:
                    neighbor_idx = neighbor_y * grid_width + neighbor_x
                    neighbor_start_idx = grid_offsets[neighbor_idx]
                    neighbor_end_idx = grid_offsets[neighbor_idx + 1]

                    # Avoid double-counting by only checking pairs where p1_index < p2_index
                    for p1_idx_ptr in range(start_idx, end_idx):
                        p1_index = grid_indices[p1_idx_ptr]
                        for p2_idx_ptr in range(neighbor_start_idx, neighbor_end_idx):
                            p2_index = grid_indices[p2_idx_ptr]
                            if p1_index < p2_index:
                                pe_change, ke_lost = _resolve_collision_jit(
                                    p1_index, p2_index, positions, velocities,
                                    masses, radii, temperatures, heat_coeff, restitution,
                                    g_const, softening
                                )
                                total_pe_change += pe_change
                                total_ke_lost += ke_lost
    return total_pe_change, total_ke_lost

@numba.jit(nopython=True)
def _resolve_collision_jit(i, j, positions, velocities, masses, radii, temperatures, heat_coeff, restitution, g_const, softening):
    """
    Numba-accelerated inelastic collision and energy conversion for a single pair.
    Modifies positions, velocities, and temperatures in place.
    Returns the change in potential energy and the kinetic energy lost.
    """
    pos_i, pos_j = positions[i], positions[j]
    rad_i, rad_j = radii[i][0], radii[j][0]

    distance_vec = pos_j - pos_i
    distance_sq = np.sum(distance_vec**2)
    min_distance = rad_i + rad_j

    pe_change = 0.0
    ke_lost = 0.0

    if distance_sq < min_distance**2 and distance_sq > 0:
        distance = np.sqrt(distance_sq)
        m_i, m_j = masses[i][0], masses[j][0]
        total_mass = m_i + m_j

        # --- Store pre-collision state for energy calculation ---
        v_i_before = velocities[i].copy()
        v_j_before = velocities[j].copy()
        ke_before = 0.5 * m_i * np.sum(v_i_before**2) + 0.5 * m_j * np.sum(v_j_before**2)
        pe_before = (-g_const * m_i * m_j) / np.sqrt(distance_sq + softening)

        # 1. Resolve Overlap
        overlap = min_distance - distance
        normal = distance_vec / distance
        positions[i] -= (m_j / total_mass) * overlap * normal
        positions[j] += (m_i / total_mass) * overlap * normal

        # --- Calculate PE change from overlap resolution ---
        final_dist_sq = np.sum((positions[j] - positions[i])**2)
        pe_after = (-g_const * m_i * m_j) / np.sqrt(final_dist_sq + softening)
        pe_change = pe_after - pe_before

        # 2. Inelastic Collision Response
        v_rel = v_j_before - v_i_before
        v_rel_dot_normal = v_rel[0] * normal[0] + v_rel[1] * normal[1]

        if v_rel_dot_normal < 0:
            impulse_j = (-(1 + restitution) * m_i * m_j * v_rel_dot_normal) / total_mass
            velocities[i] -= (impulse_j / m_i) * normal
            velocities[j] += (impulse_j / m_j) * normal

        # 3. Convert Lost Kinetic Energy to Thermal Energy
        ke_after = 0.5 * m_i * np.sum(velocities[i]**2) + 0.5 * m_j * np.sum(velocities[j]**2)
        ke_lost = ke_before - ke_after

        # The energy gained by pushing particles apart must be paid for.
        # We subtract it from the heat generated by the inelastic collision.
        # This ensures the entire collision interaction is energy-neutral.
        heat_generated = ke_lost - pe_change

        if heat_generated != 0: # Can be positive or negative
            # Distribute the net energy change as heat.
            # A 50/50 split is a reasonable abstraction.
            temperatures[i][0] += (heat_generated * 0.5) / m_i
            temperatures[j][0] += (heat_generated * 0.5) / m_j

    return pe_change, ke_lost


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
        self._cached_flat_tree = None # Cache for flattened tree data

        # --- Energy tracking variables for logging ---
        self.pe_gain_collisions = 0.0
        self.ke_loss_collisions = 0.0

        # --- Initialize properties using NumPy arrays (Structure of Arrays) ---
        self.positions = rng.random((num_particles, 2)) * self.bounds
        self.velocities = np.zeros((num_particles, 2), dtype=float)
        self.accelerations = np.zeros((num_particles, 2), dtype=float)
        self.masses = rng.uniform(config['min_mass'], config['max_mass'], (num_particles, 1))
        self.temperatures = rng.uniform(config['min_temp'], config['max_temp'], (num_particles, 1))
        self.radii = np.sqrt(self.masses).astype(int)

        # --- Spatial Grid Optimization ---
        # Heuristic: Cell size is based on the largest possible particle's diameter,
        # adjusted by a tunable multiplier from the config.
        max_radius = np.sqrt(config['max_mass']).astype(int)
        base_cell_size = max_radius * 2
        multiplier = config.get('grid_cell_size_multiplier', 1.0) # Default to 1 if not in config
        self.cell_size = int(base_cell_size * multiplier)

        if self.cell_size == 0: # Avoid division by zero if masses are tiny
            self.cell_size = 10 # A reasonable default
        self.grid_width = int(np.ceil(self.bounds[0] / self.cell_size))
        self.grid_height = int(np.ceil(self.bounds[1] / self.cell_size))
        num_cells = self.grid_width * self.grid_height
        # grid_offsets[i] stores the starting index in grid_indices for cell i.
        # The count of items in cell i is grid_offsets[i+1] - grid_offsets[i].
        self.grid_offsets = np.zeros(num_cells + 1, dtype=np.int32)
        # grid_indices stores the particle indices, sorted by cell.
        self.grid_indices = np.zeros(self.num_particles, dtype=np.int32)

        self.barnes_hut_theta = config['barnes_hut_theta']

        # --- QuadTree for Barnes-Hut Optimization ---
        qtree_boundary = BoundingBox(x=0, y=0, width=self.bounds[0], height=self.bounds[1])
        self.qtree = QuadTree(qtree_boundary)
        # Initially build the tree with starting positions. It will be rebuilt each frame.
        self.qtree.build(self.positions, self.masses)

        logger.info(f"ParticleSystem created for {num_particles} particles.")
        logger.info(f"Spatial grid initialized with cell size {self.cell_size} ({self.grid_width}x{self.grid_height} cells).")
        logger.info(f"QuadTree initialized with boundary: {qtree_boundary}")

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
        Calculates gravitational forces using the Numba-accelerated Barnes-Hut algorithm.
        This is a scientifically-grounded abstraction (Rule 8) that approximates
        the full N-body simulation by treating distant clusters of particles as
        single macro-particles. This reduces the complexity from O(n^2) to O(n log n),
        allowing for the simulation of realistic long-range gravitational effects.
        The QuadTree is flattened into NumPy arrays for Numba compatibility.
        """
        self.accelerations.fill(0.0)

        # The tree is now built directly into a flattened format. Get the cached arrays.
        self._cached_flat_tree = self.qtree.get_flattened_tree()

        # Call the JIT-compiled function with the flattened data
        _calculate_gravity_barnes_hut_jit(
            self.num_particles,
            self.positions,
            self.masses,
            self.accelerations,
            self.config['gravity_constant'],
            self.config['softening_factor'],
            self.barnes_hut_theta,
            self._cached_flat_tree[0], # node_data
            self._cached_flat_tree[1], # node_children
            self._cached_flat_tree[2], # node_particles_map
            self._cached_flat_tree[3]  # particle_list
        )

    def _handle_collisions(self):
        """
        Detects and resolves collisions using the pre-built spatial grid.
        This avoids the O(n^2) check of all possible pairs by only checking
        particles in the same or adjacent grid cells.
        The core calculation is delegated to a Numba JIT-compiled function.
        """
        pe_change, ke_lost = _handle_collisions_jit(
            self.grid_indices,
            self.grid_offsets,
            self.grid_width,
            self.grid_height,
            self.positions,
            self.velocities,
            self.masses,
            self.radii,
            self.temperatures,
            self.config['heat_transfer_coefficient'],
            self.restitution,
            self.config['gravity_constant'],
            self.config['softening_factor']
        )
        self.pe_gain_collisions = pe_change
        self.ke_loss_collisions = ke_lost

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
        Runs a full physics update step using a corrected sequence to ensure energy conservation.
        First, the continuous forces (gravity) are integrated using Velocity Verlet.
        Second, discrete events (collisions, boundaries) are handled.
        """
        # --- 1. Continuous Force Integration (Velocity Verlet) ---

        # p(t+dt) = p(t) + v(t)dt + 0.5a(t)dt^2
        # We assume dt=1, so we omit it.
        self.positions += self.velocities + 0.5 * self.accelerations
        old_accelerations = np.copy(self.accelerations)

        # --- Rebuild Spatial Partitioning Structures BEFORE Force/Collision Calculations ---
        # The structures MUST be updated with the new positions before we use them.
        self._build_spatial_grid() # Still needed for collisions
        self.qtree.build(self.positions, self.masses) # Rebuild the QuadTree each frame

        # Calculate new forces/accelerations a(t+dt) based on new positions
        self._calculate_gravity()

        # v(t+dt) = v(t) + 0.5 * (a(t) + a(t+dt)) * dt
        self.velocities += 0.5 * (old_accelerations + self.accelerations)

        # --- 2. Discrete Event Handling ---

        # Handle collisions, which modifies velocities and positions based on discrete events.
        # The grid is already up-to-date from the step above.
        self._handle_collisions()

        # Handle boundary interactions as the final step
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
        Populates a spatial grid to accelerate collision detection using a
        Numba-friendly flattened array format.

        This O(n) operation involves three passes:
        1. Count particles per cell.
        2. Calculate the starting offset for each cell in the final flat array.
        3. Populate the flat array with particle indices.
        """
        num_cells = self.grid_width * self.grid_height
        # 1. Determine cell index for each particle
        cell_xs = (self.positions[:, 0] / self.cell_size).astype(np.int32)
        cell_ys = (self.positions[:, 1] / self.cell_size).astype(np.int32)
        # Clamp to bounds
        np.clip(cell_xs, 0, self.grid_width - 1, out=cell_xs)
        np.clip(cell_ys, 0, self.grid_height - 1, out=cell_ys)
        particle_cell_indices = cell_ys * self.grid_width + cell_xs

        # 2. Count particles in each cell
        counts = np.bincount(particle_cell_indices, minlength=num_cells)

        # 3. Calculate offsets
        self.grid_offsets[1:num_cells + 1] = np.cumsum(counts)

        # 4. Populate the grid_indices array
        current_placement_indices = self.grid_offsets[:-1].copy()
        for i in range(self.num_particles):
            cell_idx = particle_cell_indices[i]
            placement_idx = current_placement_indices[cell_idx]
            self.grid_indices[placement_idx] = i
            current_placement_indices[cell_idx] += 1

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

    def get_total_potential_energy(self):
        """
        Calculates the total gravitational potential energy of the system using the
        Numba-accelerated Barnes-Hut tree. This is consistent with the force
        calculation and is highly performant. It uses the cached flattened tree
        data from the current frame to avoid redundant calculations.
        """
        if self.qtree.num_active_nodes == 0 or self._cached_flat_tree is None:
            return 0.0

        # Use the cached flattened tree data from the current frame.
        node_data, node_children, _, _ = self._cached_flat_tree

        return _calculate_potential_energy_jit(
            0, # Start at the root node (index 0)
            node_data,
            node_children,
            self.config['gravity_constant'],
            self.config['softening_factor']
        )

    def apply_global_temperature_correction(self, energy_change: float):
        """
        Applies a global energy correction by distributing it as thermal energy
        across all particles, weighted by their mass.

        Data Contract:
        - Inputs: energy_change (float) - The amount of energy to add (if > 0) or remove (if < 0).
        - Outputs: None. Modifies self.temperatures in place.
        - Side Effects: Changes the thermal energy of the entire system.
        - Invariants: Total mass must be greater than zero.
        """
        if self.masses.sum() > 0:
            # The change in thermal energy is E = m * T, so delta_T = E / m.
            # We apply this change across all particles.
            total_mass = np.sum(self.masses)
            # This is a simplification. A true mass-weighting would be more complex.
            # For now, we find the average temperature change and apply it.
            # E = M_total * T_avg_change -> T_avg_change = E / M_total
            temp_change = energy_change / total_mass
            self.temperatures += temp_change