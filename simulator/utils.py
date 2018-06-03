DIRECTIONS = ['up', 'right', 'down', 'left']


def get_state_shape(width, height, num_frames):
    '''
    Get the shape of the state tensor. The health of the snake is encoded
    next to the board with the given width.
    '''
    return width + 1, height, num_frames


def get_next_coord(coord, direction):
    '''
    Get the next coordinate when moving from coord in the given direction.
    '''

    if direction == 'up':
        return [coord[0], coord[1] - 1]
    elif direction == 'right':
        return [coord[0] + 1, coord[1]]
    elif direction == 'down':
        return [coord[0], coord[1] + 1]
    else:
        return [coord[0] - 1, coord[1]]


def is_coord_on_board(coord, width, height):
    '''
    Check whether the given coord is on the board.
    (The width and height include the border.)
    '''

    x, y = coord[0], coord[1]
    if x < 1 or y < 1 or x >= width - 1 or y >= height - 1:
        return False
    else:
        return True


def getDirection(action, snake_direction):
    '''
    Get the direction of movement for a given action and snake direction.
    '''

    if snake_direction == 'up':
        if action == 0:
            return 'left'
        elif action == 1:
            return 'up'
        else:
            return 'right'
    elif snake_direction == 'right':
        if action == 0:
            return 'up'
        elif action == 1:
            return 'right'
        else:
            return 'down'
    elif snake_direction == 'down':
        if action == 0:
            return 'right'
        elif action == 1:
            return 'down'
        else:
            return 'left'
    else:
        if action == 0:
            return 'down'
        elif action == 1:
            return 'left'
        else:
            return 'up'


def get_next_snake_coords(snake, direction, fruits):
    '''
    Get the snake body if it moves in a given direction. Grow the snake
    if there is a fruit on the next coordinate.
    '''

    next_snake = [get_next_coord(snake[0], direction)]
    if next_snake[0] in fruits:
        next_snake.extend(snake)
    else:
        next_snake.extend(snake[0:-1])
    return next_snake
