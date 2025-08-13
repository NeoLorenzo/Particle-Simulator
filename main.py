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
    tick = 0
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Physics & Logic Update ---
        # All complex logic is now encapsulated in the ParticleSystem
        particle_system.update()

                # --- Logging (Rule 2.4, throttled) ---
        if tick % 100 == 0:
            ke = particle_system.get_total_kinetic_energy()
            te = particle_system.get_total_thermal_energy()
            pe = particle_system.get_total_potential_energy()
            total_energy = ke + te + pe
            # Using DEBUG level for dense trace info (Rule 2.5)
            logger.debug(
                f"Tick={tick}, "
                f"Kinetic={ke:.2f}, "
                f"Thermal={te:.2f}, "
                f"Potential={pe:.2f}, "
                f"TotalSystemEnergy={total_energy:.2f}"
            )

        # --- Drawing ---
        screen.fill(constants.BLACK)
        particle_system.draw(screen)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(constants.FPS)

        tick += 1

    logger.info("Application shutting down.")
    pygame.quit()

if __name__ == "__main__":
    main()