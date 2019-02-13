"""
Battlesnake Gym enviroments.
"""


from gym_battlesnake.envs.battlesnake_env import BattlesnakeEnv


class BattlesnakeEnv7x7(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=8, height=8, num_fruits=1)


class BattlesnakeEnv11x11(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=12, height=12, num_fruits=1)


class BattlesnakeEnv19x19(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=20, height=20, num_fruits=1)
