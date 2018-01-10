from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .utils import getDirection, get_next_coord, is_coord_on_board, get_next_snake_coords
from .state import State


class Reward:
    collision = -1
    starve = 0
    nothing = 0
    fruit = 1


def get_state_shape(width, height, num_frames):
    return (width + 1, height, num_frames)


class BattlesnakeSimulator:

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

    def states(self):
        return get_state_shape(self.width, self.height, self.num_frames)

    def reset(self):
        self.steps = 0
        self.episodes += 1
        self.state_history = []
        self.state = State(self.width, self.height,
                           self.num_snakes, self.num_fruits)

        return self.get_last_frames(self.state.observe())

    def play_longest_episode(self):
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
        ani.save('{}.mp4'.format(self.episodes))
        plt.close()

    def get_last_frames(self, observation):
        frame = observation
        if self.frames is None:
            self.frames = deque([frame] * self.num_frames)
        else:
            self.frames.append(frame)
            self.frames.popleft()
        return np.moveaxis(np.array(self.frames), 0, -1)

    def step(self, actions):
        self.steps += 1

        next_state = None
        terminal = False
        reward = None

        for idx, action in enumerate(actions):
            snake = self.state.snakes[idx]
            direction = getDirection(action, snake.direction)
            snake_next_body = get_next_snake_coords(
                snake.body, direction, self.state.fruits)
            self.state.snakes[idx].direction = direction
            self.state.snakes[idx].body = snake_next_body
            self.state.snakes[idx].health -= 1

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

        if not terminal:
            next_state = self.get_last_frames(self.state.observe())
        if self.steps >= self.num_frames:
            self.state_history.append(next_state)
        if terminal and self.steps > self.longest_episode:
            self.longest_episode = self.steps
            self.longest_run_states = list(self.state_history)
        return next_state, reward, terminal

    def check_collision(self, snake):
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
        return snake.health == 0

    def check_fruit(self, snake):
        head = snake.body[0]
        ate_fruit = False
        for fruit in self.state.fruits:
            if np.array_equal(head, fruit):
                ate_fruit = True
        return ate_fruit
