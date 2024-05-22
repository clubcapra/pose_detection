from json_extractor import process_dataset
from logic.pose_detection import getAnglesFromBodyData
from tools.data import convertToOneHot, convertLabelsToInt, convertSoftmaxToIndex
import numpy as np
from mlp import MLP
import keras
from sklearn.model_selection import train_test_split
from tools.traces import generateTraces

EPOCHS = 200

directory = "dataset/"
X, y = process_dataset(directory)

# Will be implemented in main?
angles = []
for i in range(X.shape[0]):
    angles.append(getAnglesFromBodyData(X[i]))
X = np.array(angles)

# Split dataset into training and test (validation split is done by model)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Get one hot vector for training
y_train = convertToOneHot(y_train, 4)
y_test_oh = convertToOneHot(y_test, 4)

# Initialize model
model = MLP()
# Compile model
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()],
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)
# Training
history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=True, validation_split=0.2)

results = model.evaluate(X_test, y_test_oh)
print("test loss, test acc:", results)
print("number of epochs: ", EPOCHS)

# Make prediction (we do this instead of evaluate to have access to result vector)
y_pred = model.predict(X_test)

y_pred = convertSoftmaxToIndex(y_pred)
y_test = convertLabelsToInt(y_test)

generateTraces(history, model, y_test, y_pred)



