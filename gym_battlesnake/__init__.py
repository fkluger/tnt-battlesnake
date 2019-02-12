from gym.envs.registration import register

register(id="battlesnake-v0", entry_point="gym_battlesnake.envs:BattlesnakeEnv7x7")

register(id="battlesnake-v1", entry_point="gym_battlesnake.envs:BattlesnakeEnv11x11")

register(id="battlesnake-v2", entry_point="gym_battlesnake.envs:BattlesnakeEnv19x19")
