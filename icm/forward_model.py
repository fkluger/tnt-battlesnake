from keras import Model
from keras.layers import Dense, Reshape


class ForwardModel(Model):

    def __init__(self, num_features):
        super().__init__(name='forward_model')
        self.num_features = num_features
        self.hidden = Dense(units=256, activation='relu', name='forward_model_hidden')
        self.predicted_next_state = Dense(self.num_features, name='forward_model_output')

    def call(self, x):
        x = self.hidden(x)
        return self.predicted_next_state(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_features)