from tools.json_extractor import process_dataset
from logic.pose_detection import getAnglesFromBodyData
from tools.data import convertToOneHot, convertLabelsToInt, convertSoftmaxToIndex, balanceDataset
import numpy as np
from models.mlp import create_model_ensemble
import keras
from sklearn.model_selection import train_test_split
from tools.traces import generateTraces

EPOCHS = 80

directory = "dataset/"
pose_dict = {"none": 0, "tpose": 1, "bucket": 2, "skyward": 3}
classes = ["none", "tpose", "bucket", "skyward"]
X, y = process_dataset(directory)

unique, counts = np.unique(y, return_counts=True)
print("Incoming data: ",dict(zip(unique, counts)))

# X_balanced, y_balanced = X, y
X_balanced, y_balanced = balanceDataset(X, y)
print("Shape of X after balancing: ", X_balanced.shape)
print("Shape of y after balancing: ", y_balanced.shape)

unique, counts = np.unique(y_balanced, return_counts=True)
print("Data being fed to model:", dict(zip(unique, counts)))

# Will be implemented in main?
# angles = []
# for i in range(X_balanced.shape[0]):
#     angles.append(getAnglesFromBodyData(X_balanced[i]))
# X_balanced = np.array(angles)

# Split dataset into training and test (validation split is done by model)
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=1
)

# Get one hot vector for training
y_train = convertToOneHot(y_train, 4, pose_dict)
y_test_oh = convertToOneHot(y_test, 4, pose_dict)

# Initialize model
model = create_model_ensemble(input_shape=(7,3))
# Training
history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=True, validation_split=0.2)

results = model.evaluate(X_test, y_test_oh)
print("test loss, test acc:", results)
print("number of epochs: ", EPOCHS)

# Make prediction (we do this instead of evaluate to have access to result vector)
y_pred = model.predict(X_test)

y_pred = convertSoftmaxToIndex(y_pred)
y_test = convertLabelsToInt(y_test, pose_dict)

generateTraces(history, classes, model, y_test, y_pred)



