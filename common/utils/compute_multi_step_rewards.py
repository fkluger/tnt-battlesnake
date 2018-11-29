from typing import List

import numpy as np

from common.models.transition import Transition


def compute_multi_step_rewards(
    transition_episode: List[Transition], num_steps: int, discount_factor: float
):
    episode_length = len(transition_episode)
    for idx, transition in enumerate(transition_episode):
        multi_step_reward = transition.reward
        nth_transition = transition
        for i in range(1, num_steps):
            if idx + i < episode_length:
                multi_step_reward += (
                    np.power(discount_factor, i) * transition_episode[idx + i].reward
                )
                nth_transition = transition_episode[idx + i]
            else:
                break
        transition.reward = multi_step_reward
        transition.next_state = nth_transition.next_state
    return transition_episode
