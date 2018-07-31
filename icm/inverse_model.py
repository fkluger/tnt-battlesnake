from keras import Model
from keras.layers import Dense, Reshape


class InverseModel(Model):

    def __init__(self, num_actions):
        super().__init__(name='inverse_model')
        self.num_actions = num_actions
        self.hidden = Dense(units=256, activation='relu', name='inverse_model_hidden')
        self.predicted_action = Dense(units=self.num_actions, activation='softmax')

    def call(self, x):
        x = self.hidden(x)
        return self.predicted_action(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_actions)
