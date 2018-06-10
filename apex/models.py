class Experience:
    def __init__(self, observation, error):
        self.observation = observation
        self.error = error


class Observation:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state