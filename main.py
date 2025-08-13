# main.py

import pygame
import constants

def main():
    """
    Main function to initialize and run the particle simulation.
    """
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((constants.WIDTH, constants.HEIGHT))
    pygame.display.set_caption(constants.TITLE)
    clock = pygame.time.Clock()

    # Main loop
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Drawing
        screen.fill(constants.BLACK)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(constants.FPS)

    pygame.quit()

if __name__ == "__main__":
    main()