# constants.py

"""
Application Constants

This module defines static configuration values for the application's framework.
These are not expected to change between simulation runs.

Data Contract:
- All values are immutable constants.
- Units are specified in comments where applicable.
"""

# Screen dimensions
WIDTH = 1280  # Pixels
HEIGHT = 720  # Pixels

# Framerate
FPS = 60  # Frames per second

# Colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Window Title
TITLE = "Particle Simulator"