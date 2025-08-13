import pygame
import constants
import json
import logging
import logger_setup
import numpy as np
from particle_system import ParticleSystem

# Get the application's dedicated logger
logger = logging.getLogger("particle_sim")

def main():
    """
    Main function to initialize and run the particle simulation.
    """
    # --- Setup ---
    logger_setup.setup_logging()

    with open('config.json', 'r') as f:
        config = json.load(f)
    sim_config = config['simulation']

    logger.info("Application starting...")
    logger.info(f"Loaded configuration: {config}")

    # Initialize the master random number generator (RNG)
    rng = np.random.default_rng(config['master_seed'])
    logger.info(f"Master RNG initialized with seed: {config['master_seed']}")

    # --- Initialization ---
    pygame.init()
    screen = pygame.display.set_mode((constants.WIDTH, constants.HEIGHT))
    pygame.display.set_caption(constants.TITLE)
    clock = pygame.time.Clock()

    # Create the particle system, which manages all particle data and physics
    particle_system = ParticleSystem(
        num_particles=sim_config['particle_count'],
        config=sim_config,
        rng=rng,
        bounds=(constants.WIDTH, constants.HEIGHT)
    )

    # --- Main loop ---
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Physics & Logic Update ---
        # All complex logic is now encapsulated in the ParticleSystem
        particle_system.update()

        # --- Drawing ---
        screen.fill(constants.BLACK)
        particle_system.draw(screen)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(constants.FPS)

    logger.info("Application shutting down.")
    pygame.quit()

if __name__ == "__main__":
    main()