import numpy as np
import keras


class MLP(keras.Model):
    def __init__(self, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(4, activation="relu")
        self.bn1 = keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(8, activation="relu")
        self.bn2 = keras.layers.BatchNormalization()
        self.dense3 = keras.layers.Dense(32, activation="relu")
        self.bn3 = keras.layers.BatchNormalization()
        self.dense4 = keras.layers.Dense(16, activation="relu")
        self.bn4 = keras.layers.BatchNormalization()
        self.dense5 = keras.layers.Dense(8, activation="relu")
        self.bn5 = keras.layers.BatchNormalization()
        # self.dense6 = keras.layers.Dense(16, activation="relu")
        # self.bn6 = keras.layers.BatchNormalization()
        # self.dense7 = keras.layers.Dense(16, activation="relu")
        # self.bn7 = keras.layers.BatchNormalization()
        # self.dense8 = keras.layers.Dense(16, activation="relu")
        # self.bn8 = keras.layers.BatchNormalization()
        self.dense9 = keras.layers.Dense(4, activation="softmax")
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        # Bloc 1
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        # Bloc 2
        y = self.dense3(x)
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.dense4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        # Bloc 3
        x = self.dense5(x)
        x = self.bn5(x)
        x = self.dropout(x)
        # x = self.dense6(x)
        # x = self.bn6(x)
        # x = self.dropout(x)
        # # z = keras.layers.Add()([x, y])
        # # bloc 4
        # x = self.dense7(x)
        # x = self.bn7(x)
        # x = self.dropout(x)
        # x = self.dense8(x)
        # x = self.bn8(x)
        # x = self.dropout(x)
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
