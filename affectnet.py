import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics

from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

import tensorflow_addons as tfa

from model import AffectNetModel
from weighted_loss_functions import weighted_sparse_categorical_crossentropy


class AffectNet:
    def __init__(self,
                 checkpoint=None,
                 base_model=None,
                 base_model_output_dropout=0.0,
                 middle_layer=None,
                 preprocess_input=None,
                 num_classes=8,
                 img_height=224,
                 img_width=224,
                 num_channels=3,
                 class_weights=None,
                 learning_rate=5e-4,
                 optimizer=None,
                 callbacks=[]):
        self.middle_layer = middle_layer
        self.preprocess_input = preprocess_input
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = num_channels
        self.class_weights = class_weights
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.callbacks = callbacks
        if checkpoint:
            self.model = load_model(checkpoint, compile=False)
            self.base_model = self.model.base_model
        else:
            self.base_model = base_model
            self.model = AffectNetModel(
                base_model, preprocess_input, num_classes, base_model_output_dropout, middle_layer)
            self.model.build(
                (None, self.img_height, self.img_width, self.num_channels))

        if not self.optimizer:
            self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        self.compile(self.optimizer, self.class_weights)

    def compile(self, optimizer, class_weights):
        # loss_expression = SparseCategoricalCrossentropy(from_logits=True)
        loss_expression = weighted_sparse_categorical_crossentropy(
            class_weights)
        loss_valence = MeanSquaredError()
        loss_arousal = MeanSquaredError()

        self.model.compile(loss=loss_expression,
                           # self.model.compile(loss={"output_expression": loss_expression,
                           #                          "output_valence": loss_valence,
                           #                          "output_arousal": loss_arousal},
                           optimizer=optimizer,
                           #    metrics={"output_expression": ["accuracy"], "output_valence": ["mae"], "output_arousal": ["mae"]})
                           metrics={"output_expression": ["accuracy"]})

    def train(self, train_dataset, val_dataset=None, epochs=1, verbose=1):
        return self.model.fit(train_dataset,
                              validation_data=val_dataset,
                              epochs=epochs,
                              callbacks=self.callbacks,
                              verbose=verbose)

    def unfreeze(self, num_layers=0):
        if num_layers == 0:
            self.base_model.trainable = False
            for layer in self.base_model.layers:
                layer.trainable = False
        else:
            self.base_model.trainable = True
            for layer in self.base_model.layers[:-num_layers]:
                layer.trainable = False
            for layer in self.base_model.layers[-num_layers:]:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

    # Is this broken?
    # def save(self, save_dir, only_base=False):
    #     if only_base:
    #         tf.saved_model.save(self.base_model, save_dir)
    #     else:
    #         tf.saved_model.save(self.model, save_dir)
