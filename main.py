# main.py

import pygame
import constants
import json
import logging
import logger_setup
import numpy as np
from particle_system import ParticleSystem

# Get the application's dedicated logger
logger = logging.getLogger("particle_sim")

import cProfile, pstats

def run_simulation_loop_for_profiling(particle_system, screen, clock, trail_surface):
    """
    The main simulation loop, extracted to be run under the profiler.
    This function is temporary for diagnostics, per Rule 11.
    """
    # --- Loop Setup ---
    running = True
    tick = 0
    last_logged_energy = 0.0
    first_log = True
    accumulated_pe_gain = 0.0
    accumulated_ke_loss = 0.0
    accumulated_heat = 0.0

    # --- Profiling Configuration (Rule 11) ---
    PROFILING_TICKS = 100000

    while running and tick < PROFILING_TICKS:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Physics & Logic Update ---
        energy_before_tick = (
            particle_system.get_total_kinetic_energy() +
            particle_system.get_total_potential_energy() +
            particle_system.get_total_thermal_energy()
        )
        particle_system.update()
        energy_after_tick = (
            particle_system.get_total_kinetic_energy() +
            particle_system.get_total_potential_energy() +
            particle_system.get_total_thermal_energy()
        )

        # --- Accumulate energy changes from discrete events for logging ---
        accumulated_pe_gain += particle_system.pe_gain_collisions
        accumulated_ke_loss += particle_system.ke_loss_collisions
        accumulated_heat += particle_system.heat_generated_collisions

        # --- Logging (Rule 2.4, throttled) ---
        if tick % 100 == 0:
            ke = particle_system.get_total_kinetic_energy()
            te = particle_system.get_total_thermal_energy()
            pe = particle_system.get_total_potential_energy()
            total_energy = ke + te + pe

            if first_log:
                delta_e = 0.0
                # Set the initial energy baseline correctly on the first log
                last_logged_energy = total_energy
                first_log = False
            else:
                delta_e = total_energy - last_logged_energy

            last_logged_energy = total_energy

            logger.debug(
                f"Tick={tick}, "
                f"Kinetic={ke:.2f}, "
                f"Thermal={te:.2f}, "
                f"Potential={pe:.2f}, "
                f"TotalSystemEnergy={total_energy:.2f}, "
                f"Delta_E={delta_e:+.2f}, "
                f"AccumulatedKE_Loss={accumulated_ke_loss:+.2f}, "
                f"AccumulatedPE_Gain={accumulated_pe_gain:+.2f}, "
                f"AccumulatedHeat={accumulated_heat:+.2f}"
            )

            accumulated_pe_gain = 0.0
            accumulated_ke_loss = 0.0
            accumulated_heat = 0.0

        # --- Drawing ---
        trail_surface.fill(constants.TRAIL_EFFECT_COLOR)
        screen.blit(trail_surface, (0, 0))

        glow_surface = pygame.Surface((constants.WIDTH, constants.HEIGHT), pygame.SRCALPHA)
        particle_system.draw(glow_surface, is_glow_pass=True)

        scale = constants.BLOOM_RADIUS
        scaled_size = (constants.WIDTH // scale, constants.HEIGHT // scale)
        scaled_surface = pygame.transform.smoothscale(glow_surface, scaled_size)
        blurred_surface = pygame.transform.smoothscale(scaled_surface, (constants.WIDTH, constants.HEIGHT))

        intensity = constants.BLOOM_INTENSITY
        blurred_surface.fill((intensity, intensity, intensity), special_flags=pygame.BLEND_RGB_MULT)
        screen.blit(blurred_surface, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        particle_system.draw(screen, is_glow_pass=False)
        pygame.display.flip()
        clock.tick(constants.FPS)
        tick += 1

def main():
    """
    Main function to initialize and run the particle simulation.
    This version is modified to run the profiler for a fixed number of ticks.
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

    particle_system = ParticleSystem(
        num_particles=sim_config['particle_count'],
        config=sim_config,
        rng=rng,
        bounds=(constants.WIDTH, constants.HEIGHT)
    )

    trail_surface = pygame.Surface((constants.WIDTH, constants.HEIGHT), pygame.SRCALPHA)

    # --- Prime the physics engine (Rule 5) ---
    # This initial calculation is crucial. It populates the QuadTree's flattened
    # cache, which is required for an accurate potential energy calculation on the
    # very first tick. Without this, the first energy delta is miscalculated as
    # the entire potential energy of the system, causing a massive, incorrect
    # thermal correction that makes the system instantly unstable.
    particle_system._calculate_gravity()


    # --- Profiling Run (Rule 11) ---
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the loop function
    run_simulation_loop_for_profiling(particle_system, screen, clock, trail_surface)

    profiler.disable()
    logger.info("Profiling complete. Printing stats...")
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20) # Print the top 20 time-consuming functions

    logger.info("Application shutting down.")
    pygame.quit()

if __name__ == "__main__":
    main()