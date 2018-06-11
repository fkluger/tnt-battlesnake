class Experience:
    def __init__(self, observation, error):
        self.observation = observation
        self.error = error


class Observation:
    def __init__(self, state, action, reward, next_state, discount_factor):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.discount_factor = discount_factor
