from .plain_dqn import PlainDQNBrain


class DoubleDQNBrain(PlainDQNBrain):

    def __init__(self, input_shape, num_actions, learning_rate=0.00025):
        super().__init__(input_shape, num_actions, learning_rate)

        self.model = super().create_model()
        self.target_model = super().create_model()

    def predict(self, state, target=False):
        if target:
            return self.target_model.predict(state)
        else:
            return self.model.predict(state)

    def update_target(self):
        print('Updating target network.')
        self.target_model.set_weights(self.model.get_weights())