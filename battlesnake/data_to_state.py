import numpy as np
from environment.serializers.state_serializer import Field


def data_to_state(data, snake_direction):

    width = data['width']
    height = data['height']

    state = np.zeros([width + 2, height + 2], dtype=np.int8)

    for x in range(width + 2):
        for y in range(height + 2):
            if x == 0 or y == 0 or x == width + 1 or y == height + 1:
                state[x, y] = Field.body

    for snake in data['snakes']['data']:
        for idx, body in enumerate(snake['body']['data']):
            x, y = body['x'], body['y']
            if snake['id'] == data['you']['id']:
                if idx == 0:
                    if snake_direction == 'up':
                        state[x + 1, y + 1] = Field.own_head_up
                    elif snake_direction == 'right':
                        state[x + 1, y + 1] = Field.own_head_right
                    elif snake_direction == 'down':
                        state[x + 1, y + 1] = Field.own_head_down
                    else:
                        state[x + 1, y + 1] = Field.own_head_left
                else:
                    state[x + 1, y + 1] = Field.own_tail if idx == len(snake['body']['data']) - 1 else Field.own_body
            else:
                state[x + 1, y +
                      1] = Field.head if idx == 0 else Field.tail if idx == len(snake['body']['data']) - 1 else Field.body
    for food in data['food']['data']:
        state[food['x'] + 1][food['y'] + 1] = Field.fruit
    return state
