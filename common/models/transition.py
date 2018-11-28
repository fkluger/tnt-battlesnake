from typing import Optional

import numpy as np


class Transition:
    def __init__(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray],
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
