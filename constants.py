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
WIDTH = 2000  # Pixels
HEIGHT = 1000  # Pixels

# Framerate
FPS = 60  # Frames per second

# Colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Window Title
TITLE = "Particle Simulator"

# Color Mapping for Visualization
# Defines the temperature range for rendering particles.
COLOR_MIN_TEMP = 0.0      # Temperature that maps to the "coldest" color.
COLOR_MAX_TEMP = 5.0    # Temperature that maps to the "hottest" color.

# Defines the color spectrum as a series of keyframes.
# Each keyframe is a tuple: (normalized_position, (R, G, B) color).
COLOR_GRADIENT_KEYFRAMES = [
    (0.0,   (0, 0, 0)),          # Black
    (0.1,   (75, 0, 130)),       # Purple
    (0.25,  (0, 0, 255)),        # Blue
    (0.4,   (0, 255, 0)),        # Green
    (0.55,  (255, 255, 0)),      # Yellow
    (0.7,   (255, 0, 0)),        # Red
    (1.0,   (255, 255, 255))     # White
]

# Visual Effects
TRAIL_EFFECT_COLOR = (0, 0, 0, 30) # RGBA. Alpha controls trail length (lower = longer).

# Bloom effect settings
BLOOM_RADIUS = 20 # The radius of the glow effect in pixels. Larger is more diffuse.
BLOOM_INTENSITY = 30 # The brightness of the glow (0-255).