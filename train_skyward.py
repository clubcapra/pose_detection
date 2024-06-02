from tools.json_extractor import process_dataset_group
from logic.pose_detection import getAnglesFromBodyData
from tools.data import (
    convertToOneHot,
    convertLabelsToInt,
    convertSoftmaxToIndex,
    balanceDataset,
)
import numpy as np
from models.mlp import create_model_export
import keras
from sklearn.model_selection import train_test_split
from tools.traces import generateTraces

EPOCHS = 1500
pose_dict = {"none": 0, "skyward": 1}
ensemble_classes = ["none", "skyward"]
directories = [
    "dataset/skyward1.json",
    "dataset/skyward2.json",
    "dataset/skyward4.json",
    "dataset/skyward5.json",
    "dataset/none1.json",
    "dataset/none2.json",
    "dataset/none3.json",
    "dataset/none4.json",
    "dataset/none5.json",
]

X, y = process_dataset_group(directories)

unique, counts = np.unique(y, return_counts=True)
print("Incoming data: ", dict(zip(unique, counts)))

# X_balanced, y_balanced = X, y
X_balanced, y_balanced = balanceDataset(X, y)
print("Shape of X after balancing: ", X_balanced.shape)
print("Shape of y after balancing: ", y_balanced.shape)

unique, counts = np.unique(y_balanced, return_counts=True)
print("Data being fed to model:", dict(zip(unique, counts)))

# Will be implemented in main?
angles = []
for i in range(X_balanced.shape[0]):
    angles.append(getAnglesFromBodyData(X_balanced[i]))
X_balanced = np.array(angles)

# Split dataset into training and test (validation split is done by model)
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=1
)

# Get one hot vector for training
y_train = convertToOneHot(y_train, 2, pose_dict)
y_test_oh = convertToOneHot(y_test, 2, pose_dict)

# Initialize model
model = create_model_export(output_size=2)

# Training
history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=True, validation_split=0.2)

results = model.evaluate(X_test, y_test_oh)
print("test loss, test acc:", results)
print("number of epochs: ", EPOCHS)

# Make prediction (we do this instead of evaluate to have access to result vector)
y_pred = model.predict(X_test)

y_pred_softmaxed = convertSoftmaxToIndex(y_pred)
y_test = convertLabelsToInt(y_test, pose_dict)

generateTraces(history, ensemble_classes, model, y_test, y_pred_softmaxed)
