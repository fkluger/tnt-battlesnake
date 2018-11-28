"""
Battlesnake Gym enviroments.
"""


from gym_battlesnake.envs.battlesnake_env import BattlesnakeEnv
from gym_battlesnake.envs.state import State


class BattlesnakeEnv9x9Easy(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=9, height=9, num_fruits=3)


class BattlesnakeEnv9x9Medium(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=9, height=9, num_fruits=2)


class BattlesnakeEnv9x9Hard(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=9, height=9, num_fruits=1)


class BattlesnakeEnv9x9EasySparse(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=9, height=9, num_fruits=3, sparse_rewards=True)


class BattlesnakeEnv9x9MediumSparse(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=9, height=9, num_fruits=2, sparse_rewards=True)


class BattlesnakeEnv9x9HardSparse(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=9, height=9, num_fruits=1, sparse_rewards=True)


class BattlesnakeEnv12x12Easy(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=12, height=12, num_fruits=3)


class BattlesnakeEnv12x12Medium(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=12, height=12, num_fruits=2)


class BattlesnakeEnv12x12Hard(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=12, height=12, num_fruits=1)


class BattlesnakeEnv12x12EasySparse(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=12, height=12, num_fruits=3, sparse_rewards=True)


class BattlesnakeEnv12x12MediumSparse(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=12, height=12, num_fruits=2, sparse_rewards=True)


class BattlesnakeEnv12x12HardSparse(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=12, height=12, num_fruits=1, sparse_rewards=True)


class BattlesnakeEnv15x15Easy(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=15, height=15, num_fruits=3)


class BattlesnakeEnv15x15Medium(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=15, height=15, num_fruits=2)


class BattlesnakeEnv15x15Hard(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=15, height=15, num_fruits=1)


class BattlesnakeEnv15x15EasySparse(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=15, height=15, num_fruits=3, sparse_rewards=True)


class BattlesnakeEnv15x15MediumSparse(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=15, height=15, num_fruits=2, sparse_rewards=True)


class BattlesnakeEnv15x15HardSparse(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=15, height=15, num_fruits=1, sparse_rewards=True)


class BattlesnakeEnv18x18Easy(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=18, height=18, num_fruits=3)


class BattlesnakeEnv18x18Medium(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=18, height=18, num_fruits=2)


class BattlesnakeEnv18x18Hard(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=18, height=18, num_fruits=1)


class BattlesnakeEnv18x18EasySparse(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=18, height=18, num_fruits=3, sparse_rewards=True)


class BattlesnakeEnv18x18MediumSparse(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=18, height=18, num_fruits=2, sparse_rewards=True)


class BattlesnakeEnv18x18HardSparse(BattlesnakeEnv):
    def __init__(self):
        super().__init__(width=18, height=18, num_fruits=1, sparse_rewards=True)
