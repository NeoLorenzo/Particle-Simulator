# particle.py

import pygame
import logging
import numpy as np
from constants import WHITE

class Particle:
    """
    Represents a single particle in the simulation.

    Data Contract:
    - Inputs to __init__:
        - mass (float): Mass of the particle. Units: kilograms (conceptual).
        - temperature (float): Temperature of the particle. Units: Kelvin (conceptual).
        - position (np.ndarray): 2D vector for position [x, y]. Units: meters (conceptual).
        - velocity (np.ndarray): 2D vector for velocity [vx, vy]. Units: m/s (conceptual).
    - Side Effects: Logs its own creation at the DEBUG level.
    - Invariants: Position and velocity are always 2D NumPy arrays.
    """
    def __init__(self, mass: float, temperature: float, position: np.ndarray, velocity: np.ndarray):
        self.mass = mass
        self.temperature = temperature
        self.position = position
        self.velocity = velocity

        # For now, radius is proportional to mass. This is a simplification.
        self.radius = int(np.sqrt(mass))

        logging.debug(f"Particle created: mass={self.mass}, temp={self.temperature}, pos={self.position}")

    def update(self):
        """
        Updates the particle's state for one time step.
        Currently, only updates position based on velocity.
        """
        self.position = self.position + self.velocity

    def draw(self, screen: pygame.Surface):
        """
        Draws the particle on the screen.
        """
        pygame.draw.circle(screen, WHITE, self.position.astype(int), self.radius)