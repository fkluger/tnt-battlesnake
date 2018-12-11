from environment.battlesnake_environment import BattlesnakeEnvironment
from environment.snake import Snake


def snake_to_data(snake: Snake):
    data = {
        "body": {
            "data": [
                {"object": "point", "x": body_part[0], "y": body_part[1]}
                for body_part in snake.body
            ],
            "object": "list",
        },
        "health": snake.health,
        "id": None,
        "length": len(snake.body),
        "name": None,
        "object": "snake",
        "taunt": "",
    }
    return data


def state_to_data(env: BattlesnakeEnvironment, index: int):
    data = {
        "food": {"data": [], "object": "list"},
        "height": env.height,
        "id": None,
        "object": "world",
        "snakes": {
            "data": [snake_to_data(s) for s in env.state.snakes],
            "object": "list",
        },
        "turn": env.stats.episode_steps_current,
        "width": env.width,
        "you": snake_to_data(env.state.snakes[index]),
    }

    return data
