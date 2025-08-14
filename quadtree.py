# quadtree.py

import numpy as np
from collections import namedtuple
import logging
import numba

logger = logging.getLogger("particle_sim")

# A simple structure for defining the bounding box of a QuadTree node.
BoundingBox = namedtuple('BoundingBox', ['x', 'y', 'width', 'height'])

@numba.jit(nopython=True)
def _get_quadrant(pos, center_x, center_y):
    """Determine which of the four quadrants a particle belongs to."""
    if pos[0] < center_x:
        return 0 if pos[1] < center_y else 2 # NW or SW
    else:
        return 1 if pos[1] < center_y else 3 # NE or SE

@numba.jit(nopython=True)
def _subdivide_jit(node_idx, next_node_idx, node_boundaries, node_children):
    """JIT-friendly subdivision of a node."""
    x, y, w, h = node_boundaries[node_idx]
    half_w, half_h = w / 2, h / 2

    # Assign children indices from the pre-allocated pool
    nw_idx, ne_idx, sw_idx, se_idx = next_node_idx, next_node_idx + 1, next_node_idx + 2, next_node_idx + 3
    node_children[node_idx, 0] = nw_idx
    node_children[node_idx, 1] = ne_idx
    node_children[node_idx, 2] = sw_idx
    node_children[node_idx, 3] = se_idx

    # Define boundaries for the new children
    node_boundaries[nw_idx] = (x, y, half_w, half_h)
    node_boundaries[ne_idx] = (x + half_w, y, half_w, half_h)
    node_boundaries[sw_idx] = (x, y + half_h, half_w, half_h)
    node_boundaries[se_idx] = (x + half_w, y + half_h, half_w, half_h)

    return next_node_idx + 4

@numba.jit(nopython=True)
def _insert_jit(p_idx, node_idx, positions, next_node_idx, node_boundaries, node_children, node_is_leaf, node_particle_idx):
    """JIT-friendly recursive insertion."""
    # Loop to avoid deep recursion stacks which Numba can struggle with
    while True:
        # If it's a leaf node
        if node_is_leaf[node_idx]:
            # If the leaf is empty, place the particle here
            if node_particle_idx[node_idx] == -1:
                node_particle_idx[node_idx] = p_idx
                return next_node_idx

            # Leaf is not empty, so we must subdivide
            existing_p_idx = node_particle_idx[node_idx]
            node_particle_idx[node_idx] = -1 # It's an internal node now
            node_is_leaf[node_idx] = False

            next_node_idx = _subdivide_jit(node_idx, next_node_idx, node_boundaries, node_children)

            # Re-insert the existing particle into the correct new child
            x, y, w, h = node_boundaries[node_idx]
            center_x, center_y = x + w / 2, y + h / 2
            
            old_pos = positions[existing_p_idx]
            quadrant = _get_quadrant(old_pos, center_x, center_y)
            child_to_insert_old = node_children[node_idx, quadrant]
            next_node_idx = _insert_jit(existing_p_idx, child_to_insert_old, positions, next_node_idx, node_boundaries, node_children, node_is_leaf, node_particle_idx)

            # Continue the loop to insert the NEW particle into the correct child
            new_pos = positions[p_idx]
            quadrant = _get_quadrant(new_pos, center_x, center_y)
            node_idx = node_children[node_idx, quadrant] # Tail-recursion optimization
            continue

        # If it's an internal node, find the correct child and recurse
        else:
            x, y, w, h = node_boundaries[node_idx]
            center_x, center_y = x + w / 2, y + h / 2
            pos = positions[p_idx]
            quadrant = _get_quadrant(pos, center_x, center_y)
            node_idx = node_children[node_idx, quadrant] # Tail-recursion optimization
            continue

@numba.jit(nopython=True)
def _calculate_mass_distribution_jit(node_idx, positions, masses, node_data, node_children, node_is_leaf, node_particle_idx):
    """JIT-friendly post-order traversal to calculate center of mass."""
    if node_is_leaf[node_idx]:
        p_idx = node_particle_idx[node_idx]
        if p_idx != -1:
            mass = masses[p_idx, 0]
            pos = positions[p_idx]
            node_data[node_idx, 0] = mass
            node_data[node_idx, 1] = pos[0]
            node_data[node_idx, 2] = pos[1]
        return

    # It's an internal node, recurse on children first
    total_mass = 0.0
    com_x_num = 0.0
    com_y_num = 0.0
    for child_idx in node_children[node_idx]:
        if child_idx != -1:
            _calculate_mass_distribution_jit(child_idx, positions, masses, node_data, node_children, node_is_leaf, node_particle_idx)
            child_mass = node_data[child_idx, 0]
            if child_mass > 0:
                total_mass += child_mass
                com_x_num += node_data[child_idx, 1] * child_mass
                com_y_num += node_data[child_idx, 2] * child_mass

    node_data[node_idx, 0] = total_mass
    if total_mass > 0:
        node_data[node_idx, 1] = com_x_num / total_mass
        node_data[node_idx, 2] = com_y_num / total_mass

@numba.jit(nopython=True)
def _build_tree_and_map_jit(positions, masses, node_boundaries, node_data, node_children, node_is_leaf, node_particle_idx, node_particles_map, particle_list):
    """Main JIT function to build the tree and create the final flattened maps."""
    num_particles = len(positions)
    
    # --- 1. Build the tree structure ---
    next_node_idx = 1 # Start allocating from index 1 (0 is root)
    for i in range(num_particles):
        next_node_idx = _insert_jit(i, 0, positions, next_node_idx, node_boundaries, node_children, node_is_leaf, node_particle_idx)

    # --- 2. Calculate mass distribution ---
    _calculate_mass_distribution_jit(0, positions, masses, node_data, node_children, node_is_leaf, node_particle_idx)

    # --- 3. Create the final flattened particle map for gravity calculation ---
    particle_pos_counter = 0
    for i in range(next_node_idx):
        node_data[i, 3] = node_boundaries[i, 2] # Set node width
        if node_is_leaf[i]:
            p_idx = node_particle_idx[i]
            if p_idx != -1:
                start = particle_pos_counter
                particle_list[start] = p_idx
                particle_pos_counter += 1
                node_particles_map[i, 0] = start
                node_particles_map[i, 1] = 1
            # else: empty leaf, map is already [-1, 0, -1]
    return next_node_idx # Return number of active nodes

class QuadTree:
    def __init__(self, boundary: BoundingBox, capacity: int = 1):
        self.boundary = boundary
        self.capacity = capacity # Note: JIT implementation assumes capacity=1
        
        # Pre-allocate arrays for the tree. Heuristic: max_nodes is ~2*num_particles
        # This avoids costly re-allocation. If it ever fails, increase the multiplier.
        self.max_nodes = 0
        self.node_boundaries = np.empty(0, dtype=np.float64)
        self.node_data = np.empty(0, dtype=np.float64)
        self.node_children = np.empty(0, dtype=np.int32)
        self.node_is_leaf = np.empty(0, dtype=np.bool_)
        self.node_particle_idx = np.empty(0, dtype=np.int32)
        self.node_particles_map = np.empty(0, dtype=np.int32)
        self.particle_list = np.empty(0, dtype=np.int32)
        self.num_active_nodes = 0

    def _ensure_capacity(self, num_particles):
        """Ensure arrays are large enough for the current number of particles."""
        # In a capacity=1 tree where every subdivision creates 4 new nodes, a safe
        # upper bound is 1 (root) + 4 * (N-1) (for N-1 splits). We use 4*N for simplicity.
        required_nodes = num_particles * 4 + 1
        if required_nodes > self.max_nodes:
            self.max_nodes = required_nodes
            self.node_boundaries = np.zeros((self.max_nodes, 4), dtype=np.float64)
            self.node_data = np.zeros((self.max_nodes, 4), dtype=np.float64)
            self.node_children = np.full((self.max_nodes, 4), -1, dtype=np.int32)
            self.node_is_leaf = np.ones(self.max_nodes, dtype=np.bool_)
            self.node_particle_idx = np.full(self.max_nodes, -1, dtype=np.int32)
            self.node_particles_map = np.full((self.max_nodes, 3), -1, dtype=np.int32)
            self.node_particles_map[:, 1] = 0
            self.particle_list = np.zeros(num_particles, dtype=np.int32)

    def build(self, positions: np.ndarray, masses: np.ndarray):
        num_particles = len(positions)
        self._ensure_capacity(num_particles)

        # --- Reset Tree State ---
        self.node_boundaries[0] = self.boundary
        self.node_children.fill(-1)
        self.node_is_leaf.fill(True)
        self.node_particle_idx.fill(-1)
        self.node_data.fill(0.0)
        self.node_particles_map.fill(-1)
        self.node_particles_map[:, 1] = 0

        # --- Build Tree using JIT function ---
        if num_particles > 0:
            self.num_active_nodes = _build_tree_and_map_jit(
                positions, masses, self.node_boundaries, self.node_data,
                self.node_children, self.node_is_leaf, self.node_particle_idx,
                self.node_particles_map, self.particle_list
            )

    def get_flattened_tree(self):
        """Returns the flattened tree arrays, trimmed to the number of active nodes."""
        if self.num_active_nodes == 0:
            return (
                np.empty((0, 4), dtype=np.float64),
                np.empty((0, 4), dtype=np.int32),
                np.empty((0, 3), dtype=np.int32),
                np.empty(0, dtype=np.int32)
            )
        
        return (
            self.node_data[:self.num_active_nodes],
            self.node_children[:self.num_active_nodes],
            self.node_particles_map[:self.num_active_nodes],
            self.particle_list
        )