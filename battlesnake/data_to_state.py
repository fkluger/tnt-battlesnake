import numpy as np
from gym_battlesnake.envs.constants import Field, Direction


def data_to_state(width, height, data, snake_direction):

    current_state = np.zeros((width, height), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                current_state[x, y] = Field.body.value
    for idx, snake in enumerate(data["board"]["snakes"]):
        if snake["id"] == data["you"]["id"]:
            current_state[0, height - 1] = snake["health"]
        else:
            current_state[idx, 0] = snake["health"]
        if snake["health"] == 0:
            continue
        snake_length = len(snake["body"])
        for body_idx, coords in enumerate(snake["body"]):
            x, y = coords["x"] + 1, coords["y"] + 1
            if snake["id"] == data["you"]["id"]:
                if body_idx == 0:
                    if snake_direction == Direction.up:
                        current_state[x, y] = Field.own_head_up.value
                    elif snake_direction == Direction.right:
                        current_state[x, y] = Field.own_head_right.value
                    elif snake_direction == Direction.down:
                        current_state[x, y] = Field.own_head_down.value
                    else:
                        current_state[x, y] = Field.own_head_left.value
                else:
                    current_state[x, y] = (
                        Field.own_tail.value
                        if body_idx == snake_length - 1
                        else Field.own_body.value
                    )
            else:
                current_state[x, y] = (
                    Field.head.value
                    if body_idx == 0
                    else Field.tail.value
                    if body_idx == snake_length - 1
                    else Field.body.value
                )
    for coords in data["board"]["food"]:
        x, y = coords["x"], coords["y"]
        current_state[x, y] = Field.fruit.value
    if snake_direction == Direction.left:
        current_state = np.rot90(current_state, k=3)
    elif snake_direction == Direction.down:
        current_state = np.rot90(current_state, k=2)
    elif snake_direction == Direction.right:
        current_state = np.rot90(current_state, k=1)
    return current_state
