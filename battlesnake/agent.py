import json
import logging
import os
import time
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import loader

from .constants import Direction
from .snake import Snake

from .data_to_state import data_to_state

LOGGER = logging.getLogger("Agent")


class Agent(Snake):
    def __init__(self, width: int, height: int, stacked_frames: int, path: str):
        self.width = width
        self.height = height
        self.stacked_frames = stacked_frames
        self.path = path

        self.observation_ph, self.q_values = self._load_graph()

    def _compute_actions(self, observation):
        q_values = self.sess.run(self.q_values, {self.observation_ph: [observation]})[0]
        actions = np.argsort(q_values)[::-1]
        return actions

    def _load_graph(self):
        with tf.Graph().as_default() as graph:
            self.sess = tf.Session(graph=graph)
            loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], self.path)
            observation_ph = graph.get_tensor_by_name("snake_0/Placeholder:0")
            q_values = graph.get_tensor_by_name("snake_0/q_func/Sum:0")
            return observation_ph, q_values

    def on_reset(self):
        self.head_direction = Direction.up
        self.frames = deque(
            np.zeros([self.stacked_frames, self.width, self.height], dtype=np.uint8),
            self.stacked_frames,
        )

    def get_direction(self, data):
        state = data_to_state(self.width, self.height, data, self.head_direction)
        self.frames.appendleft(state)
        observation = np.moveaxis(self.frames, 0, -1)
        actions = self._compute_actions(observation)
        self.head_direction = self._find_best_action(actions, data)
        if self.head_direction == Direction.up:
            return "up"
        elif self.head_direction == Direction.left:
            return "left"
        elif self.head_direction == Direction.down:
            return "down"
        else:
            return "right"

    def _find_best_action(self, actions, data):
        head = data["you"]["body"][0]
        head = [head["x"] + 1, head["y"] + 1]
        directions = [self._get_direction(i) for i in actions]
        next_coords = [self._get_next_head(direction, head) for direction in directions]
        low_health = data["you"]["health"] <= 75
        if low_health:
            for direction, next_coord in zip(directions, next_coords):
                coord_is_food = any(
                    [
                        np.array_equal(next_coord, [coord["x"], coord["y"]])
                        for coord in data["board"]["food"]
                    ]
                )
                if coord_is_food:
                    return direction
        for direction, next_coord in zip(directions, next_coords):
            if not self._check_no_collision(next_coord, data):
                return direction
            else:
                print("Avoided collision! Trying other directions...")
                continue
        print("Giving up!")
        return directions[0]

    def _check_no_collision(self, head, data):
        collision = False
        for s in data["board"]["snakes"]:
            for body_idx, coord in enumerate(s["body"]):
                coord = [coord["x"] + 1, coord["y"] + 1]
                if s["id"] == data["you"]["id"] and body_idx == 0:
                    continue
                else:
                    if np.array_equal(head, coord):
                        collision = True
        snake_head_x, snake_head_y = head[0], head[1]
        hit_wall = (
            snake_head_x <= 0
            or snake_head_y <= 0
            or snake_head_x >= self.width - 1
            or snake_head_y >= self.height - 1
        )
        if hit_wall:
            collision = True
        return collision
