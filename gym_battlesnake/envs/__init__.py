"""
Battlesnake Gym enviroments.
"""


from gym_battlesnake.envs.battlesnake_env import BattlesnakeEnv


class BattlesnakeEnv7x7(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=7, height=7, num_fruits=1)


class BattlesnakeEnv11x11(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=11, height=11, num_fruits=1)


class BattlesnakeEnv19x19(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=19, height=19, num_fruits=1)
