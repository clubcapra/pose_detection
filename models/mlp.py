import numpy as np
import keras

def create_model_expert():
    model = keras.Sequential(
        [
            keras.Input(shape=(7,3)),
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(units=2, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )

    return model

def create_model_general(input_shape=(7,3), output_size=4):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.BatchNormalization(),
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
    print(input_layer)
    output_layer = model(input_layer)
    new_model = keras.Model(inputs=input_layer, outputs=output_layer, name=new_name)
    
    return new_model

def create_model_ensemble(input_shape=(7,3), output_size=4):
    general_model = load_and_rename_model("./trainings/training80_4class/model/model80.keras", "general_model")
    skyward_model = load_and_rename_model("./trainings/training77_skyward/model/model77.keras", "skyward_model")
    bucket_model = load_and_rename_model("./trainings/training91_bucket/model/model91.keras", "bucket_model")
    tpose_model = load_and_rename_model("./trainings/training78_tpose/model/model78.keras", "tpose_model")

    base_models = [general_model, skyward_model, bucket_model, tpose_model]

    # Input layer
    input_layer = keras.Input(shape=input_shape)
    flatten_input = keras.layers.Flatten()(input_layer)

    # Get outputs from each base model
    base_outputs = [model(input_layer) for model in base_models]

    # Concatenate the outputs from base models and the original input
    concatenated = keras.layers.Concatenate()([*base_outputs, flatten_input])

    # Define the decision head
    x = keras.layers.Dense(64, activation="relu")(concatenated)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
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
