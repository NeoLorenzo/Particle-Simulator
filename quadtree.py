# quadtree.py

import numpy as np
from collections import namedtuple
import logging

logger = logging.getLogger("particle_sim")

# A simple structure for defining the bounding box of a QuadTree node.
BoundingBox = namedtuple('BoundingBox', ['x', 'y', 'width', 'height'])

class QuadNode:
    """
    Represents a node in the QuadTree. Each node has a boundary and can
    either be a leaf (containing particles) or an internal node (containing
    four children nodes).

    Data Contract:
    - Inputs: boundary (BoundingBox)
    - Outputs: None
    - Side Effects: Manages its own children and particle list.
    - Invariants:
        - A node can have children or particles, but not both.
        - The total number of particles in a node's children must equal the
          number of particles it would hold if it were a leaf.
    """
    def __init__(self, boundary: BoundingBox):
        self.boundary = boundary
        self.particles = []  # List of particle indices in this node
        self.children = None  # Will be a list of 4 QuadNodes if subdivided

        # --- Center of Mass Attributes ---
        self.center_of_mass = np.zeros(2)
        self.total_mass = 0.0
        self.is_leaf = True

    def subdivide(self):
        """
        Creates four children nodes that partition the current node's space.
        This is called when a leaf node exceeds its particle capacity.
        """
        self.is_leaf = False
        x, y, w, h = self.boundary
        half_w, half_h = w / 2, h / 2

        # Create the boundaries for the four new quadrants
        nw_boundary = BoundingBox(x, y, half_w, half_h)
        ne_boundary = BoundingBox(x + half_w, y, half_w, half_h)
        sw_boundary = BoundingBox(x, y + half_h, half_w, half_h)
        se_boundary = BoundingBox(x + half_w, y + half_h, half_w, half_h)

        self.children = [
            QuadNode(nw_boundary),
            QuadNode(ne_boundary),
            QuadNode(sw_boundary),
            QuadNode(se_boundary)
        ]

class QuadTree:
    """
    A QuadTree data structure for spatially partitioning particles. This is the
    foundation for the Barnes-Hut algorithm.

    Data Contract:
    - Inputs:
        - boundary (BoundingBox): The root boundary of the entire space.
        - capacity (int): The max number of particles in a node before it subdivides.
    - Outputs: None.
    - Side Effects: Builds a tree structure based on particle positions.
    - Invariants: The tree must be rebuilt each frame to reflect new particle positions.
    """
    def __init__(self, boundary: BoundingBox, capacity: int = 1):
        self.boundary = boundary
        self.capacity = capacity
        self.root = QuadNode(boundary)

    def insert(self, particle_index: int, position: np.ndarray, mass: float, node: QuadNode):
        """
        Recursively inserts a particle into the correct node in the tree.
        If a node reaches capacity, it subdivides and its particles are
        re-inserted into the new children.
        """
        # If the particle is not in this node's boundary, ignore it.
        if not self._boundary_contains(node.boundary, position):
            return False

        # If the node is a leaf and has space, add the particle.
        if node.is_leaf and len(node.particles) < self.capacity:
            node.particles.append(particle_index)
            return True

        # If the node is a leaf but is now over capacity, we must subdivide it.
        if node.is_leaf:
            node.subdivide()
            # Re-insert all particles from the now-internal node into its new children.
            for p_idx in node.particles:
                # This requires access to the main particle system's data,
                # which is a dependency we will pass in during the build step.
                p_pos = self.positions[p_idx]
                p_mass = self.masses[p_idx][0]
                self.insert(p_idx, p_pos, p_mass, node)
            node.particles = [] # Clear particles from the internal node

        # If the node is already an internal node, pass the new particle down
        # to the correct child.
        for child in node.children:
            if self.insert(particle_index, position, mass, child):
                return True

        return False # Should not happen if logic is correct

    def build(self, positions: np.ndarray, masses: np.ndarray):
        """
        Builds the entire QuadTree from scratch based on the current particle data.
        Also calculates the center of mass for each node.
        """
        self.root = QuadNode(self.boundary)
        # Store references to the particle data for the duration of the build.
        # This avoids passing them through every recursive call.
        self.positions = positions
        self.masses = masses

        for i in range(len(positions)):
            self.insert(i, positions[i], masses[i][0], self.root)

        # After the tree is built, recursively calculate centers of mass.
        self._calculate_mass_distribution(self.root)

    def _calculate_mass_distribution(self, node: QuadNode):
        """
        Recursively calculates the total mass and center of mass for each node.
        For a leaf node, it's the average of its particles.
        For an internal node, it's the weighted average of its children's centers of mass.
        """
        if node.is_leaf:
            if len(node.particles) > 0:
                particle_indices = node.particles
                node.total_mass = np.sum(self.masses[particle_indices])
                if node.total_mass > 0:
                    # Weighted average of positions: sum(m_i * p_i) / sum(m_i)
                    node.center_of_mass = np.sum(self.masses[particle_indices] * self.positions[particle_indices], axis=0) / node.total_mass
            else:
                node.total_mass = 0.0
        else:
            # Recursively call on children first
            for child in node.children:
                self._calculate_mass_distribution(child)

            # Calculate this node's properties from its children
            node.total_mass = sum(child.total_mass for child in node.children)
            if node.total_mass > 0:
                # Weighted average of children's centers of mass
                com_numerator = sum(child.center_of_mass * child.total_mass for child in node.children)
                node.center_of_mass = com_numerator / node.total_mass

    def _boundary_contains(self, boundary: BoundingBox, position: np.ndarray) -> bool:
        """Checks if a 2D point is within a bounding box."""
        x, y, w, h = boundary
        return (x <= position[0] < x + w) and (y <= position[1] < y + h)

    def flatten_tree(self):
        """
        Flattens the tree of QuadNode objects into a set of NumPy arrays that
        can be passed to a Numba JIT-compiled function. This version uses
        dynamic Python lists to avoid pre-allocation errors.

        Returns a tuple of NumPy arrays:
        - node_data: Properties of each node (mass, com_x, com_y, width).
        - node_children: Maps a node index to its four children's indices.
        - node_particles_map: Maps a leaf node index to its particle data.
        """
        # Use dynamic lists to build the data
        node_data_list = []
        node_children_list = []
        node_particles_map_list = []
        particle_list = []

        particle_pos_counter = 0
        node_idx_counter = 0

        # Queue for breadth-first traversal: (node_object, assigned_node_index)
        queue = [(self.root, 0)]
        # Add placeholder entries for the root node
        node_data_list.append(None)
        node_children_list.append(None)
        node_particles_map_list.append(None)
        node_idx_counter += 1

        head = 0
        while head < len(queue):
            current_node, current_idx = queue[head]
            head += 1

            # Populate data for the current node
            node_data_list[current_idx] = [
                current_node.total_mass,
                current_node.center_of_mass[0],
                current_node.center_of_mass[1],
                current_node.boundary.width
            ]

            if current_node.is_leaf:
                num_particles = len(current_node.particles)
                if num_particles > 0:
                    start = particle_pos_counter
                    particle_list.extend(current_node.particles)
                    particle_pos_counter += num_particles
                    node_particles_map_list[current_idx] = [start, num_particles, -1]
                else:
                    node_particles_map_list[current_idx] = [-1, 0, -1]
                # Mark as leaf by having no children
                node_children_list[current_idx] = [-1, -1, -1, -1]
            else:
                # It's an internal node, store its children
                child_indices = []
                for child_node in current_node.children:
                    child_idx = node_idx_counter
                    queue.append((child_node, child_idx))
                    child_indices.append(child_idx)
                    # Add placeholders for the new child node
                    node_data_list.append(None)
                    node_children_list.append(None)
                    node_particles_map_list.append(None)
                    node_idx_counter += 1
                node_children_list[current_idx] = child_indices
                # Mark as internal node
                node_particles_map_list[current_idx] = [-1, 0, -1]

        # Convert the lists to NumPy arrays at the end
        return (
            np.array(node_data_list, dtype=np.float64),
            np.array(node_children_list, dtype=np.int32),
            np.array(node_particles_map_list, dtype=np.int32),
            np.array(particle_list, dtype=np.int32)
        )