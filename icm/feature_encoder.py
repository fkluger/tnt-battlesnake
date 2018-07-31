from keras import Model
from keras.utils import conv_utils
from keras.layers import Conv2D, Flatten


class FeatureEncoder(Model):

    def __init__(self, filters=32, kernel_size=3, strides=2, padding='same'):
        super().__init__(name='feature_encoder')
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv1 = Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                            padding=self.padding, activation='relu', name='features_1')
        self.conv2 = Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                            padding=self.padding, activation='relu', name='features_2')
        self.conv3 = Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                            padding=self.padding, activation='relu', name='features_3')
        self.conv4 = Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                            padding=self.padding, activation='relu', name='features_4')
        self.features = Flatten(name='features')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.features(x)

    def compute_output_shape(self, input_shape):
        shape = self.conv1.compute_output_shape(input_shape)
        shape = self.conv2.compute_output_shape(shape)
        shape = self.conv3.compute_output_shape(shape)
        shape = self.conv4.compute_output_shape(shape)
        return self.features.compute_output_shape(shape)
