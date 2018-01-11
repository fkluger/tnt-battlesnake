import numpy as np
from .utils import get_next_coord, is_coord_on_board, DIRECTIONS


class Snake:

    def __init__(self, head, length, board_width, board_height, occupied_coords, idx):
        self.body = self.generate_body(
            head, length, board_width, board_height, occupied_coords)
        self.direction = DIRECTIONS[np.random.randint(0, len(DIRECTIONS))]
        self.health = 100
        self.id = idx

    def generate_body(self, head, length, board_width, board_height, occupied_coords):
        '''
        Generates a random body for the snake on the board.
        '''

        body = [head]
        for n in range(0, length - 1):
            direction = None
            tries = 0
            while direction is None:
                direction = DIRECTIONS[np.random.randint(
                    0, len(DIRECTIONS))]
                next_coord = get_next_coord(body[n], direction)
                direction = direction if is_coord_on_board(
                    next_coord, board_width, board_height) and next_coord not in body and next_coord not in occupied_coords else None
                tries += 1
                if tries > 10:
                    length = n
                    next_coord = None
                    break
            if next_coord is not None:
                body.append(next_coord)
            else:
                break
        return body
