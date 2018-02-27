from keras.optimizers import RMSprop

from .plain_dqn import PlainDQNBrain
from .huber_loss import huber_loss


class DoubleDQNBrain(PlainDQNBrain):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = self.create_model()
        self.target_model = self.create_model()

    def predict(self, state, target=False):
        if target:
            return self.target_model.predict(state)
        else:
            return self.model.predict(state)

    def update_target(self):
        print('Updating target network.')
        self.target_model.set_weights(self.model.get_weights())
