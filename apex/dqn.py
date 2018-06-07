from keras import Model, Input
from keras.layers import Conv2D, Flatten, Lambda, Add
from keras.optimizers import Adam
from tensorflow import reduce_mean, tile
import numpy as np
from noisy_dense_layer import NoisyDense

class DQN:

    losses = []

    def __init__(self, input_shape, num_actions, hidden_size, learning_rate, report_interval):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.report_interval = report_interval
        self.model = self._create_model()
        self.callbacks = []

    def _create_model(self):
        inputs = Input(shape=self.input_shape)
        cnn_features = Conv2D(32, 3, activation='relu', strides=(1, 1))(inputs)
        cnn_features = Conv2D(64, 2, activation='relu', strides=(2, 2))(cnn_features)
        cnn_features = Conv2D(64, 2, activation='relu', strides=(1, 1))(cnn_features)
        cnn_features = Flatten()(cnn_features)
        
        advantage = NoisyDense(self.hidden_size, activation='relu')(cnn_features)
        advantage = NoisyDense(self.num_actions, activation='relu')(advantage)
        advantage = Lambda(lambda advt: advt - reduce_mean(advt, axis=-1, keepdims=True))(advantage)

        value = NoisyDense(self.hidden_size, activation='relu')(cnn_features)
        value = NoisyDense(1, activation='relu')(cnn_features)
        value = Lambda(lambda value: tile(value, [1, self.num_actions]))(value)

        outputs = Add()([value, advantage])
        model = Model(inputs, outputs)

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def predict(self, state):
        if len(state.shape) == 3:
            # Single state
            state = np.expand_dims(state, 0)
        return self.model.predict(state)
    
    def compute_q_values(self, observations):
        no_state = np.zeros(self.input_shape)
        next_states = np.array([(no_state
                                 if o.next_state is None else o.next_state)
                                for o in observations])
        states = np.array([o.state for o in observations])
        q_values = np.array(self.model.predict(states))
        q_values_next = np.array(self.model.predict(next_states))
        return q_values, q_values_next
    
    def train(self, x, y, batch_size, weights):
        history = self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            verbose=0,
            sample_weight=weights,
            callbacks=self.callbacks)
        self.losses.append(history.history['loss'][0])
    
    def get_metrics(self):
        return [{
            'name': 'mean_loss',
            'value': np.mean(self.losses[-self.report_interval:]),
            'type': 'value'
        }]
