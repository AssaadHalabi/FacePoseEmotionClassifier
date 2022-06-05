from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer


class AffectNetModel(Model):
    def __init__(self, base_model, preprocess_input, num_classes, base_model_output_dropout=0.0, middle_layer=None):
        super(AffectNetModel, self).__init__()
        self.num_classes = num_classes
        self.preprocess_input = preprocess_input
        self.base_model = base_model
        self.global_pool = layers.GlobalAveragePooling2D()
        self.base_model_output_dropout = base_model_output_dropout
        self.middle_layer = middle_layer
        self.output_layer = OutputLayer(num_classes)

    def call(self, inputs, training=False):
        if self.preprocess_input:
            x = self.preprocess_input(inputs)
        else:
            x = inputs
        x = self.base_model(x, training)
        x = self.global_pool(x)
        if self.base_model_output_dropout:
            x = layers.Dropout(self.base_model_output_dropout)(x)
        if self.middle_layer:
            x = self.middle_layer(x)
        x = self.output_layer(x)
        return x


class OutputLayer(Layer):
    def __init__(self, num_classes):
        super(OutputLayer, self).__init__()
        self.output_expression = layers.Dense(
            num_classes, name="output_expression")
        # self.output_valence = layers.Dense(1, name="output_valence")
        # self.output_arousal = layers.Dense(1, name="output_arousal")

    def call(self, inputs):
        # Single output
        return {"output_expression": self.output_expression(inputs)}
        # return {"output_expression": self.output_expression(inputs),
        #         "output_valence": self.output_valence(inputs),
        #         "output_arousal": self.output_arousal(inputs)}
