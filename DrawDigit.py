import sys
import pygame
import os


class DrawDigitClass:

    bgd_col = None
    digit_col = None
    dis_size = None

    surface_shape = None

    surface_width = None
    surface_height = None

    surface_radius = None
    surface_thickness = None # of if shape == "circle"

    def __init__(self, display_size=(500, 500), background_color=(0, 0, 0), digit_color=(255, 255, 255), shape="box",
                 shape_width=40, shape_height=40, shape_radius=20, shape_thickness=20):
        self.surface_shape = shape
        self.bgd_col = background_color
        self.digit_col = digit_color
        self.dis_size = display_size
        self.surface_width = shape_width
        self.surface_height = shape_height
        self.surface_thickness = shape_thickness
        self.surface_radius = shape_radius
        print("\nDraw a digit")
        print("Space : Redo\nEnter : Proceed\nEsc : Quit")

    def start(self):
        pygame.init()

        window = pygame.display.set_mode(self.dis_size)
        pygame.display.set_caption("Draw Canvas")
        window.fill(self.bgd_col)

        run = True
        while run:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise Exception("Not an Exception. Pyautogiu window was closed")

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                run = False
            if keys[pygame.K_SPACE]:
                window.fill(self.bgd_col)
            if keys[pygame.K_RETURN]:
                print("\nSaved screen as image in : " + os.getcwd() + " as screenshot.jpeg")
                pygame.image.save(window, "screenshot.jpeg")
                run = False

            mouse_x, mouse_y = pygame.mouse.get_pos()

            if pygame.mouse.get_pressed() == (1, 0, 0):  # as in (Right , middle, left) in mouse
                if self.surface_shape == "box":
                    pygame.draw.rect(window, self.digit_col, (mouse_x, mouse_y, self.surface_width, self.surface_height))
                elif self.surface_shape == "circle":
                    pygame.draw.circle(window, self.digit_col, (mouse_x, mouse_y), self.surface_radius, self.surface_thickness)
                else:
                    print("Invalid Shape")
                    sys.exit(1)

            pygame.display.update()
        pygame.quit()
