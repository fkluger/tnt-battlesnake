import numpy as np

from .snake import Snake
from .enums import Field


class State:
    '''
    Represents the state of the simulator.
    '''

    def __init__(self, width, height, num_snakes, num_fruits):
        self.width = width
        self.height = height
        self.fruits = []
        self.snakes = []

        self.generate_fruits(num_fruits, [])
        self.generate_snakes(num_snakes, list(self.fruits))

    def generate_fruits(self, num_fruits, occupied_coords):
        '''
        Generate a number of fruits on random coordinates that are not occupied.
        '''

        for _ in range(0, num_fruits):
            fruit = None
            # Find free field for the fruit
            while fruit is None:
                fruit = [np.random.randint(1, self.width - 2),
                         np.random.randint(1, self.height - 2)]
                for coord in occupied_coords:
                    if np.array_equal(fruit, coord):
                        fruit = None
            self.fruits.append(fruit)

    def eat_fruit(self, fruit):
        '''
        Remove the given fruit and generate a new one.
        '''

        self.fruits.remove(fruit)
        occupied_coords = list(self.snakes)
        occupied_coords.extend(self.fruits)
        self.generate_fruits(1, occupied_coords)

    def generate_snakes(self, num_snakes, occupied_coords):
        '''
        Generate a number of snakes at random coordinates
        '''

        for i in range(0, num_snakes):
            head = None
            # Find free field for snake head
            while head is None:
                head = [np.random.randint(1, self.width - 2),
                        np.random.randint(1, self.height - 2)]
                if head in occupied_coords:
                    head = None
            snake = Snake(head, 3, self.width, self.height, occupied_coords, i)
            occupied_coords.extend(snake.body)
            self.snakes.append(snake)

    def observe(self):
        '''
        Create a tensor with shape (width+1, height) describing the current state.
        '''

        observation = np.zeros([self.width + 1, self.height], dtype=int)
        for x in range(0, self.width):
            for y in range(0, self.height):
                if x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1:
                    observation[x, y] = Field.body
        for snake_idx, snake in enumerate(self.snakes):
            for idx, [x, y] in enumerate(snake.body):
                # Snake at position 0 is the agent
                if snake_idx == 0:
                    if idx == 0:
                        if snake.direction == 'up':
                            observation[x][y] = Field.own_head_up
                        elif snake.direction == 'right':
                            observation[x][y] = Field.own_head_right
                        elif snake.direction == 'down':
                            observation[x][y] = Field.own_head_down
                        else:
                            observation[x][y] = Field.own_head_left
                    else:
                        observation[x][y] = Field.own_tail if idx == len(snake.body) - 1 else Field.own_body
                else:
                    observation[x][y] = Field.head if idx == 0 else Field.tail if idx == len(
                        snake.body) - 1 else Field.body
                if snake_idx == 0:
                    observation[self.width][0] = snake.health
        for [x, y] in self.fruits:
            observation[x][y] = Field.fruit
        return observation
