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
    last_logged_energy = 0.0
    first_log = True
    # Accumulators for energy changes over the logging interval
    accumulated_pe_gain = 0.0
    accumulated_ke_loss = 0.0
    accumulated_integration_error = 0.0
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Physics & Logic Update ---
        # Measure energy before the tick's physics are processed
        energy_before_tick = (
            particle_system.get_total_kinetic_energy() +
            particle_system.get_total_potential_energy() +
            particle_system.get_total_thermal_energy()
        )

        particle_system.update()

        # Measure energy after the tick
        energy_after_tick = (
            particle_system.get_total_kinetic_energy() +
            particle_system.get_total_potential_energy() +
            particle_system.get_total_thermal_energy()
        )

        # --- Calculate and accumulate energy changes for this tick ---
        total_delta_this_tick = energy_after_tick - energy_before_tick
        pe_gain_this_tick = particle_system.pe_gain_collisions
        ke_loss_this_tick = particle_system.ke_loss_collisions

        # The integration error is the total change minus the known change from collisions
        integration_error_this_tick = total_delta_this_tick - pe_gain_this_tick
        accumulated_integration_error += integration_error_this_tick

        accumulated_pe_gain += pe_gain_this_tick
        accumulated_ke_loss += ke_loss_this_tick


        # --- Logging (Rule 2.4, throttled) ---
        if tick % 100 == 0:
            # --- BEGIN ENERGY CORRECTION (USER HYPOTHESIS) ---
            # This is a non-physical step to test forcing energy conservation.
            thermal_correction = -accumulated_integration_error
            particle_system.apply_global_temperature_correction(thermal_correction)
            # --- END ENERGY CORRECTION ---

            # Recalculate final energy state after the correction
            ke = particle_system.get_total_kinetic_energy()
            te = particle_system.get_total_thermal_energy()
            pe = particle_system.get_total_potential_energy()
            total_energy = ke + te + pe

            # Calculate the change in energy since the last log event
            if first_log:
                delta_e = 0.0
                # On the first run, last_logged_energy should be the initial state
                last_logged_energy = energy_before_tick
                first_log = False

            delta_e = total_energy - last_logged_energy
            last_logged_energy = total_energy

            # Using DEBUG level for dense trace info (Rule 2.5)
            logger.debug(
                f"Tick={tick}, "
                f"Kinetic={ke:.2f}, "
                f"Thermal={te:.2f}, "
                f"Potential={pe:.2f}, "
                f"TotalSystemEnergy={total_energy:.2f}, "
                f"Delta_E={delta_e:+.2f}, "
                f"AccumulatedIntegrationError={accumulated_integration_error:+.2f}, "
                f"ThermalCorrection={thermal_correction:+.2f}"
            )

            # Reset accumulators for the next interval
            accumulated_pe_gain = 0.0
            accumulated_ke_loss = 0.0
            accumulated_integration_error = 0.0

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