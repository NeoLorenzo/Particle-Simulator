# main.py

import pygame
import constants
import json
import logging
import logger_setup
import numpy as np
from particle import Particle

def main():
    """
    Main function to initialize and run the particle simulation.
    """
    # --- Setup ---
    logger_setup.setup_logging()

    with open('config.json', 'r') as f:
        config = json.load(f)

    logging.info("Application starting...")
    logging.info(f"Loaded configuration: {config}")

    # Initialize the master random number generator (RNG)
    rng = np.random.default_rng(config['master_seed'])
    logging.info(f"Master RNG initialized with seed: {config['master_seed']}")

    # --- Initialization ---
    pygame.init()
    screen = pygame.display.set_mode((constants.WIDTH, constants.HEIGHT))
    pygame.display.set_caption(constants.TITLE)
    clock = pygame.time.Clock()

    # Create a single particle for testing
    particles = [
        Particle(
            mass=100.0,
            temperature=300.0,
            position=rng.random(2) * np.array([constants.WIDTH, constants.HEIGHT]),
            velocity=rng.random(2) * 2 - 1 # Random velocity between -1 and 1
        )
    ]

    # --- Main loop ---
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Logic Update
        for p in particles:
            p.update()

        # Drawing
        screen.fill(constants.BLACK)
        for p in particles:
            p.draw(screen)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(constants.FPS)

    logging.info("Application shutting down.")
    pygame.quit()

if __name__ == "__main__":
    main()