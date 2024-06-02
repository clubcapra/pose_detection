import numpy as np
import keras


class MLP_ensemble(keras.Model):
    def __init__(self, output_size=4, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.backbone = keras.saving.load_model("./trainings/training63_best4class/model/model63.keras", compile=False)
        self.skyward = keras.saving.load_model("./trainings/training80_skyward/model/model80.keras", compile=False)
        self.bucket = keras.saving.load_model("./trainings/training87_bucket/model/model87.keras", compile=False)
        self.tpose = keras.saving.load_model("./trainings/training86_tpose/model/model86.keras", compile=False)
        self.base_models = [self.backbone, self.skyward, self.bucket, self.tpose]
        self.dense1 = keras.layers.Dense(14, activation="relu")
        self.bn1 = keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(32, activation="relu")
        self.bn2 = keras.layers.BatchNormalization()
        self.dense3 = keras.layers.Dense(32, activation="relu")
        self.bn3 = keras.layers.BatchNormalization()

        self.dense9 = keras.layers.Dense(output_size, activation="softmax")
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.concat = keras.layers.Concatenate()

    def call(self, inputs):
        # Base models
        base_outputs = [model(inputs) for model in self.base_models]
        concatenated_baseoutputs = self.concat(base_outputs)
        x = self.concat(concatenated_baseoutputs, inputs)
        # Bloc 2
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        return self.dense9(x)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dropout_rate': self.dropout.rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
