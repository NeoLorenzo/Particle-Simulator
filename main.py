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
    sim_config = config['simulation']

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

    # Create a list of particles
    particles = []
    for _ in range(sim_config['particle_count']):
        mass = rng.uniform(sim_config['min_mass'], sim_config['max_mass'])
        particles.append(
            Particle(
                mass=mass,
                temperature=300.0, # Static for now
                position=rng.random(2) * np.array([constants.WIDTH, constants.HEIGHT]),
                velocity=np.zeros(2, dtype=float) # Start at rest
            )
        )

    # --- Main loop ---
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Physics Update ---
        # Reset acceleration for all particles
        for p in particles:
            p.acceleration = np.zeros(2, dtype=float)

        # Calculate gravitational forces (O(n^2) complexity)
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                p1 = particles[i]
                p2 = particles[j]

                # Vector from p1 to p2
                direction_vec = p2.position - p1.position
                distance_sq = np.sum(direction_vec**2)

                # Abstraction (Rule 8): Add a softening factor to prevent extreme forces
                # at very close distances. This avoids division by zero and instability.
                distance_sq += sim_config['softening_factor']

                # Force calculation: F = G * (m1*m2) / r^2
                force_magnitude = (sim_config['gravity_constant'] * p1.mass * p2.mass) / distance_sq
                force_vector = force_magnitude * direction_vec / np.sqrt(distance_sq)

                # Apply force to particles (a = F/m)
                p1.acceleration += force_vector / p1.mass
                p2.acceleration -= force_vector / p2.mass

        # --- Collision Detection and Response (O(n^2) complexity) ---
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                p1 = particles[i]
                p2 = particles[j]

                distance_vec = p2.position - p1.position
                distance = np.linalg.norm(distance_vec)
                min_distance = p1.radius + p2.radius

                if distance < min_distance:
                    # Collision detected
                    # 1. Resolve overlap to prevent sticking
                    overlap = min_distance - distance
                    p1.position -= 0.5 * overlap * (distance_vec / distance)
                    p2.position += 0.5 * overlap * (distance_vec / distance)

                    # 2. Calculate elastic collision response (2D)
                    # Source: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
                    normal = distance_vec / distance
                    v_rel = p2.velocity - p1.velocity
                    impulse_j = (-2 * p1.mass * p2.mass * np.dot(v_rel, normal)) / (p1.mass + p2.mass)

                    # Apply impulse to velocities
                    p1.velocity -= (impulse_j / p1.mass) * normal
                    p2.velocity += (impulse_j / p2.mass) * normal


        # --- Logic Update ---
        for p in particles:
            p.update()
            p.check_boundary_collision(constants.WIDTH, constants.HEIGHT)

        # --- Drawing ---
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