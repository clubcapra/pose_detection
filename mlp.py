import numpy as np
import keras


class MLP(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.bn1 = keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(64, activation="relu")
        self.bn2 = keras.layers.BatchNormalization()
        self.dense3 = keras.layers.Dense(256, activation="relu")
        self.bn3 = keras.layers.BatchNormalization()
        self.dense4 = keras.layers.Dense(512, activation="relu")
        self.bn4 = keras.layers.BatchNormalization()
        self.dense5 = keras.layers.Dense(256, activation="relu")
        self.bn5 = keras.layers.BatchNormalization()
        self.dense6 = keras.layers.Dense(64, activation="relu")
        self.bn6 = keras.layers.BatchNormalization()
        self.dense7 = keras.layers.Dense(4, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = self.dense4(x)
        x = self.bn4(x)
        x = self.dense5(x)
        x = self.bn5(x)
        x = self.dense6(x)
        x = self.bn6(x)
        return self.dense7(x)
