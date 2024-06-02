import numpy as np
import keras


# class MLP(keras.Model):
#     def __init__(self, output_size=4, dropout_rate=0.3, **kwargs):
#         super().__init__(**kwargs)
#         self.dense1 = keras.layers.Dense(4, activation="relu")
#         self.bn1 = keras.layers.BatchNormalization()
#         self.dense2 = keras.layers.Dense(8, activation="relu")
#         self.bn2 = keras.layers.BatchNormalization()
#         self.dense3 = keras.layers.Dense(8, activation="relu")
#         self.bn3 = keras.layers.BatchNormalization()
#         self.dense4 = keras.layers.Dense(8, activation="relu")
#         self.bn4 = keras.layers.BatchNormalization()
#         # self.dense5 = keras.layers.Dense(8, activation="relu")
#         # self.bn5 = keras.layers.BatchNormalization()
#         # self.dense6 = keras.layers.Dense(16, activation="relu")
#         # self.bn6 = keras.layers.BatchNormalization()
#         # self.dense7 = keras.layers.Dense(16, activation="relu")
#         # self.bn7 = keras.layers.BatchNormalization()
#         # self.dense8 = keras.layers.Dense(16, activation="relu")
#         # self.bn8 = keras.layers.BatchNormalization()
#         self.dense9 = keras.layers.Dense(output_size, activation="softmax")
#         self.dropout = keras.layers.Dropout(dropout_rate)

#     def call(self, inputs):
#         # Bloc 1
#         x = self.dense1(inputs)
#         x = self.bn1(x)
#         x = self.dropout(x)
#         x = self.dense2(x)
#         x = self.bn2(x)
#         x = self.dropout(x)
#         # Bloc 2
#         y = self.dense3(x)
#         x = self.bn3(x)
#         x = self.dropout(x)
#         x = self.dense4(x)
#         x = self.bn4(x)
#         x = self.dropout(x)
#         # Bloc 3
#         # x = self.dense5(x)
#         # x = self.bn5(x)
#         # x = self.dropout(x)
#         # x = self.dense6(x)
#         # x = self.bn6(x)
#         # x = self.dropout(x)
#         # # z = keras.layers.Add()([x, y])
#         # # bloc 4
#         # x = self.dense7(x)
#         # x = self.bn7(x)
#         # x = self.dropout(x)
#         # x = self.dense8(x)
#         # x = self.bn8(x)
#         # x = self.dropout(x)
#         return self.dense9(x)

#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'dropout_rate': self.dropout.rate,
#         })
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)


def create_model_expert(output_size=4):
    model = keras.Sequential(
        [
            keras.Input(shape=(4,)),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(units=output_size, activation="softmax"),
        ]
    )

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    )

    return model

def create_model_general(output_size=4):
    model = keras.Sequential(
        [
            keras.Input(shape=(4,)),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(units=output_size, activation="softmax"),
        ]
    )

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    )

    return model

def load_and_rename_model(filepath, new_name):
    model = keras.saving.load_model(filepath, compile=False)
    model.trainable = False  # Ensure the base model is not trainable
    
    # Create a new functional model with a unique name
    input_layer = keras.Input(shape=model.input_shape[1:])
    output_layer = model(input_layer)
    new_model = keras.Model(inputs=input_layer, outputs=output_layer, name=new_name)
    
    return new_model

def create_model_ensemble(output_size=4):
    general_model = load_and_rename_model("./trainings/training106_4class/model/model106.keras", "general_model")
    skyward_model = load_and_rename_model("./trainings/training93_skyward/model/model93.keras", "skyward_model")
    bucket_model = load_and_rename_model("./trainings/training95_bucket/model/model95.keras", "bucket_model")
    tpose_model = load_and_rename_model("./trainings/training97_tpose/model/model97.keras", "tpose_model")

    base_models = [general_model, skyward_model, bucket_model, tpose_model]

    # Input layer
    input_layer = keras.Input(shape=(4,))

    # Get outputs from each base model
    base_outputs = [model(input_layer) for model in base_models]

    # Concatenate the outputs from base models and the original input
    concatenated = keras.layers.Concatenate()([*base_outputs, input_layer])

    # Define the decision head
    x = keras.layers.Dense(14, activation="relu")(concatenated)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    output_layer = keras.layers.Dense(units=output_size, activation="softmax")(x)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
        optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    )

    return model
