import os

import numpy as np
from .state import State
from .constants import Field, Direction
import pygame

if os.environ["DISPLAY"]:
    pygame.init()
    pygame.font.init()

LEADERBOARD_WIDTH = 300
LEADERBOARD_ITEM_HEIGHT = 100
GAME_PADDING = 30
SCREEN_BACKGROUND = (170, 204, 153)
WALL = (56, 56, 56)
SNAKE = (80, 140, 215)
FRUIT = (215, 115, 85)



class GameRenderer:
    """
    Diese Klasse sorgt dafür, dass das Spiel mittels pygame angezeigt werden kann.
    """
    def __init__(self, game_width, game_height, nb_snakes):
        """
        Erstellt ein Fenster für das Spiel
        :param game_width: Spielfeldbreite
        :param game_height: Spielfeldhöhe
        :param nb_snakes: Anzahl der Schlangen im Leaderboard
        """

        self.pixel_per_field = 30

        game_pixel_width = game_width * self.pixel_per_field
        game_pixel_height = game_height * self.pixel_per_field

        leaderboard_pixel_height = nb_snakes*LEADERBOARD_ITEM_HEIGHT

        total_width = GAME_PADDING + game_pixel_width + GAME_PADDING + LEADERBOARD_WIDTH + GAME_PADDING
        total_height = GAME_PADDING + max(game_pixel_height, leaderboard_pixel_height) + GAME_PADDING

        self.screen = pygame.display.set_mode((total_width, total_height))

        pygame.display.set_caption('Battlesnake')

        self.surface_game = pygame.Surface((game_pixel_width, game_pixel_height))
        self.surface_leaderboard = pygame.Surface((LEADERBOARD_WIDTH, leaderboard_pixel_height))

        self.health_bar_rects = None
        self.game_pixel_width = game_pixel_width
        self.game_pixel_height = game_pixel_height
        self.leaderboard_pixel_height = leaderboard_pixel_height

    def display(self, state: State):
        """
        Zeigt das Spiel an
        :param game: Game Objekt, das angezeigt wird
        :return:
        """

        # Der Hintergrund wird mit schwarz gefüllt
        self.screen.fill(SCREEN_BACKGROUND)

        # Die Spieloberfläche wird gezeichnet
        self.render(state, self.surface_game)
        # self.render_leaderboard(state, self.surface_leaderboard)

        game_position = (GAME_PADDING, GAME_PADDING)
        self.screen.blit(self.surface_game, game_position)

        # Das Leaderboard wird gezeichnet
        # leaderboard_y_start = GAME_PADDING + max((self.game_pixel_height - self.leaderboard_pixel_height) / 2, 0)
        # leaderboard_position = (GAME_PADDING + self.game_pixel_width + GAME_PADDING, leaderboard_y_start)
        # self.screen.blit(self.surface_leaderboard, leaderboard_position)

        # Wenn das Spiel gewonnen wurde, Info anzeigen
        '''
        if game.is_finished():
            winner = game.get_winner()

            if game.finish_when_winner:
                if winner is not None:
                    message = winner.get_name() + ' hat gewonnen'
                else:
                    message = 'Unentschieden'
            else:
                message = 'Spiel beendet'

            font = pygame.font.Font(None, 40)
            text = font.render(message, True, (255, 255, 255))

            text_rect = text.get_rect()
            text_x = self.screen.get_width() / 2 - text_rect.width / 2
            text_y = self.screen.get_height() / 2 - text_rect.height / 2
            self.screen.blit(text, [text_x, text_y])
        '''
        # GUI aktualisieren
        pygame.display.flip()

    def game_to_pixel_coordinates(self, game_x, game_y):
        """
        Umrechnung der Spielfeldkoordinaten in Pixelkoordinaten
        """
        return game_x*self.pixel_per_field, game_y*self.pixel_per_field

    def box_coordinates(self, game_x, game_y):
        """
        Diese Funktion berechnet die Eckpunkte in Pixelkoordinaten eines Feldes
        """
        x_min, y_min = self.game_to_pixel_coordinates(game_x, game_y)
        x_max, y_max = self.game_to_pixel_coordinates(game_x + 1, game_y + 1)
        return x_min, y_min, x_max - 1, y_max - 1

    @staticmethod
    def rotate_points_around_center(pts: np.ndarray, cnt: np.ndarray, degrees: float):
        """
        Rotiert die Punkte pts um das Zentrum cnt um die angegebene Gradzahl
        :param pts: Punkte, die rotiert werden sollen
        :param cnt: Zentrum der Rotation
        :param degrees: Grad um die rotiert werden soll
        :return: rotierte Punkte
        """
        ang = degrees / 180*np.pi
        return np.dot(pts - cnt, np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])) + cnt

    @staticmethod
    def flip_points_around_center(pts: np.ndarray, cnt: np.ndarray, vertical=False, horizontal=False):

        flip_mult = np.eye(2, 2)

        if vertical:
            flip_mult[0, 0] = -1

        if horizontal:
            flip_mult[1, 1] = -1

        return np.dot(pts - cnt, flip_mult) + cnt

    @staticmethod
    def rotate_points(direction: Direction, pts: np.ndarray, center: np.ndarray):
        """
        Rotiert Punkte um eine Richtung
        Grundausrichtung ist  Direction.RIGHT
        """

        if direction == Direction.up:
            return GameRenderer.rotate_points_around_center(pts, center, -90)

        elif direction == Direction.right:
            return pts

        elif direction == Direction.down:
            return GameRenderer.rotate_points_around_center(pts, center, 90)

        elif direction == Direction.left:
            return GameRenderer.flip_points_around_center(pts, center, vertical=True)

        else:
            print('ERROR unknown head direction')

    def render(self, state, surface: pygame.Surface):
        surface.fill(SCREEN_BACKGROUND)

        # Wände zeichnen
        for x in range(state.width):
            for y in range(state.height):
                if x == 0 or y == 0 or x == state.width - 1 or y == state.height - 1:
                    x_min, y_min, x_max, y_max = self.box_coordinates(x, y)

                    pygame.draw.rect(surface, WALL, pygame.Rect(x_min, y_min, self.pixel_per_field, self.pixel_per_field))

        # Snakes zeichnen
        for snake_index, snake in enumerate(state.snakes):
            if snake.is_dead():
                continue

            snake_length = len(snake.body)
            for body_idx, [_x, _y] in enumerate(snake.body):

                x, y = self.game_to_pixel_coordinates(_x, _y)
                pre_x, pre_y = self.game_to_pixel_coordinates(snake.body[body_idx - 1][0], snake.body[body_idx - 1][1])

                r = int(self.pixel_per_field / 5)
                if body_idx == 1:
                    if pre_x == x:
                        if pre_y > y:
                            tr1 = [[pre_x, pre_y], [pre_x + self.pixel_per_field - 1, pre_y], [pre_x, pre_y + self.pixel_per_field - 1]]
                            tr2 = [[pre_x + self.pixel_per_field/2 - 1, pre_y], [pre_x + self.pixel_per_field - 1, pre_y], [pre_x + self.pixel_per_field - 1, pre_y + self.pixel_per_field - 1]]
                            circle = [pre_x + r, pre_y + r]
                        else:
                            tr1 = [[x, y], [x + self.pixel_per_field - 1, y], [pre_x, pre_y]]
                            tr2 = [[x + self.pixel_per_field/2 - 1, y], [x + self.pixel_per_field - 1, y], [pre_x + self.pixel_per_field - 1, pre_y]]
                            circle = [x + r, y - r]
                    if pre_y == y:
                        if pre_x > x:
                            tr1 = [[pre_x, pre_y], [pre_x + self.pixel_per_field - 1, pre_y], [pre_x, pre_y + self.pixel_per_field - 1]]
                            tr2 = [[pre_x, pre_y + self.pixel_per_field/2 - 1], [pre_x, pre_y + self.pixel_per_field - 1], [pre_x + self.pixel_per_field - 1, pre_y + self.pixel_per_field - 1]]
                            circle = [pre_x + r, pre_y + r]
                        else:
                            tr1 = [[x, y], [x, y + self.pixel_per_field - 1], [pre_x, pre_y]]
                            tr2 = [[x, y + self.pixel_per_field/2 - 1], [x, y + self.pixel_per_field - 1], [pre_x, pre_y + self.pixel_per_field - 1]]
                            circle = [x - r, y + r]
                    pygame.draw.polygon(surface, SNAKE, tr1)
                    pygame.draw.polygon(surface, SNAKE, tr2)
                    pygame.draw.rect(surface, SNAKE, [x, y, self.pixel_per_field, self.pixel_per_field])
                    pygame.draw.circle(surface, (0, 0, 0), circle, int(r/3))
                if body_idx > 1:
                    if body_idx == len(snake.body) - 1:
                        triangle1 = 0
                        if pre_x == x:
                            if pre_y > y:
                                triangle1 = [[pre_x, pre_y], [pre_x + self.pixel_per_field - 1, pre_y], [x + self.pixel_per_field/2 - 1, y]]
                            else:
                                triangle1 = [[x, y], [x + self.pixel_per_field - 1, y], [x + self.pixel_per_field/2 - 1, y + self.pixel_per_field - 1]]
                        if pre_y == y:
                            if pre_x > x:
                                triangle1 = [[pre_x, pre_y], [pre_x, pre_y + self.pixel_per_field - 1], [x, y + self.pixel_per_field/2 - 1]]
                            else:
                                triangle1 = [[x, y], [x, y + self.pixel_per_field - 1], [x + self.pixel_per_field - 1, y + self.pixel_per_field/2 - 1]]
                        pygame.draw.polygon(surface, SNAKE, triangle1)

                    else:
                        pygame.draw.rect(surface, SNAKE, [x, y, self.pixel_per_field, self.pixel_per_field])


        # Früchte zeichnen

        for [x, y] in state.fruits:
            x_min, y_min, x_max, y_max = self.box_coordinates(x, y)
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)
            radius = int(0.8*self.pixel_per_field / 2)
            center = (center_x, center_y)

            pygame.draw.circle(surface, FRUIT, center, radius)

    def render_leaderboard(self, state: State, surface: pygame.Surface):
        """
        Leaderboard zeichnen
        :param game: aktuelles Spiel
        :param surface: pygame surface (Fenster)
        :return:
        """

        surface_width = surface.get_width()

        for i in range(len(state.snakes)):
            x_start = 0
            y_start = i*LEADERBOARD_ITEM_HEIGHT

            snake = state.snakes[i]
            snake_name = snake.get_name()
            snake_health = max(snake.get_health(), 0)
            snake_color = snake.get_color()

            # myfont = pygame.font.SysFont('Comic Sans MS', 30)
            myfont = pygame.font.Font('fonts/karla/Karla-Regular.ttf', 25)
            textsurface = myfont.render(snake_name, True, (255, 255, 255))

            surface.blit(textsurface, (x_start, y_start))

            bar_y_start = y_start + 50

            background_bar_color = GameRenderer.mix_colors(snake_color, Field.background, 0.5)
            background_bar_rect = pygame.Rect(x_start, bar_y_start, surface_width, 30)
            pygame.draw.rect(surface, background_bar_color, background_bar_rect)

            if snake_health > 0:
                bar_rect = pygame.Rect(x_start, bar_y_start, snake_health/100*surface_width, 30)
                pygame.draw.rect(surface, snake_color, bar_rect)

    @staticmethod
    def mix_colors(color_a, color_b, ratio):
        """
        Mischt zwei Farben
        :param color_a:
        :param color_b:
        :param ratio: Verhältnis der Mischung. 0 entsprecht nur Farbe B und 1 nur Farbe A
        :return:
        """
        return ratio * np.array(color_a) + (1 - ratio) * np.array(color_b)