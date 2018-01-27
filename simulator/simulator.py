import json
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .utils import getDirection, is_coord_on_board, get_next_snake_coords
from .state import State
from .enums import Reward
from . import Simulator


class BattlesnakeSimulator(Simulator):

    longest_episode = 0
    longest_run_states = []
    episodes = 0
    steps = 0
    state_history = []

    def __init__(self, width, height, num_snakes, num_fruits, num_frames):
        self.width = width
        self.height = height
        self.num_snakes = num_snakes
        self.num_fruits = num_fruits
        self.state = None
        self.num_frames = num_frames
        self.frames = None

    def reset(self):
        '''
        Reset the state and return the initial observation.
        '''

        self.steps = 0
        self.episodes += 1
        self.state_history = []
        self.state = State(self.width, self.height, self.num_snakes, self.num_fruits)

        return self.get_last_frames(self.state.observe())

    def to_battlesnake_json(self, idx):
        bs_state = {
            'you': str(idx),
            'width': self.width - 2,
            'height': self.height - 2,
            'turn': self.steps,
            'game_id': 0,
            'food': self.state.fruits,
            'dead_snakes': [],
            'mode': 'simple'
        }
        snakes = []
        for s in self.state.snakes:
            snake = {'id': 'de508402-17c8-4ac7-ab0b-f96cb53fbee8', 'name': str(idx), 'health_points': s.health}
            coords = [[coord[0] - 1, coord[1] - 1] for coord in s.body]
            snake['coords'] = coords
            snakes.append(snake)
        bs_state['snakes'] = snakes
        return bs_state

    def save_longest_episode(self, output_directory):
        '''
        Persist longest run since this method was called last time as a *.mp4.
        '''

        if self.longest_episode > 1:
            self.longest_episode = 0
            ims = []
            fig = plt.figure()
            for s in self.longest_run_states:
                if s is not None:
                    img = s[0:self.width, :, self.num_frames - 1]
                    im = plt.imshow(img, animated=True)
                    ims.append([im])
            ani = animation.ArtistAnimation(
                fig, ims, interval=100, repeat=False, blit=True)
            print('Saving longest episode till {} with length {}.'.format(self.episodes, len(self.longest_run_states)))
            ani.save('{}/episode-{}-steps-{}.mp4'.format(output_directory, self.episodes, len(self.longest_run_states)))
            plt.close()

    def get_last_frames(self, observation):
        '''
        Return a tensor that contains the last num_frames states. The first state will
        be repeated num_frames times.
        '''

        frame = observation
        if self.frames is None:
            self.frames = deque([frame] * self.num_frames)
        else:
            self.frames.append(frame)
            self.frames.popleft()
        return np.moveaxis(np.array(self.frames), 0, -1)

    def step(self, actions):
        '''
        Perform one simulation step with the given actions. Each action corresponds to the
        snake at the same index.
        '''

        self.steps += 1

        next_state = None
        terminal = False
        reward = None

        # Compute next snake positions and health
        for idx, action in enumerate(actions):
            snake = self.state.snakes[idx]
            direction = getDirection(action, snake.direction)
            snake_next_body = get_next_snake_coords(
                snake.body, direction, self.state.fruits)
            self.state.snakes[idx].direction = direction
            self.state.snakes[idx].body = snake_next_body
            self.state.snakes[idx].health -= 1

        # Compute rewards and whether the episode ended (terminal)
        for idx, snake in enumerate(self.state.snakes):
            collided = self.check_collision(snake)
            ate_fruit = self.check_fruit(snake)
            starved = False if ate_fruit else self.check_starved(snake)

            if idx == 0:
                if collided:
                    terminal = True
                    reward = Reward.collision
                elif ate_fruit:
                    snake.health = 100
                    self.state.eat_fruit(snake.body[0])
                    terminal = False
                    reward = Reward.fruit
                elif starved:
                    terminal = True
                    reward = Reward.starve
                else:
                    terminal = False
                    reward = Reward.nothing

        # Compute next state
        next_state = None if terminal else self.get_last_frames(self.state.observe())

        # Update statistics
        if self.steps >= self.num_frames:
            self.state_history.append(next_state)
        if terminal and self.steps > self.longest_episode:
            self.longest_episode = self.steps
            self.longest_run_states = list(self.state_history)

        return next_state, reward, terminal

    def check_collision(self, snake):
        '''
        Check whether the snake collided with itself or another snake or the wall.
        '''

        collision = False
        head = snake.body[0]
        for idx, s in enumerate(self.state.snakes):
            for body_idx, coord in enumerate(s.body):
                if idx == snake.id and body_idx == 0:
                    continue
                else:
                    if np.array_equal(head, coord):
                        collision = True

        if not is_coord_on_board(head, self.width, self.height):
            collision = True
        return collision

    def check_starved(self, snake):
        '''
        Check whether the snake starved.
        '''

        return snake.health == 0

    def check_fruit(self, snake):
        '''
        Check whether the snake ate a fruit.
        '''

        head = snake.body[0]
        ate_fruit = False
        for fruit in self.state.fruits:
            if np.array_equal(head, fruit):
                ate_fruit = True
        return ate_fruit
