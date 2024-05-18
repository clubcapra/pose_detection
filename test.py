from json_extractor import process_dataset
from pose_detection import getVectorFromPoints, getAnglesFromBodyData
import numpy as np
from mlp import MLP
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

pose_dict = {"none": 0, "tpose": 1, "bucket": 2, "skyward": 3}

directory = "dataset/"
X, y = process_dataset(directory)

angles = []
for i in range(X.shape[0]):
    angles.append(getAnglesFromBodyData(X[i]))
X = np.array(angles)

oh_y = np.zeros((X.shape[0], 4))

for i in range(y.shape[0]):
    index = pose_dict[y[i]]
    oh_y[i][index] = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, oh_y, test_size=0.2, random_state=1
)

model = MLP()
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalCrossentropy()],
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)
model.fit(X_train, y_train, epochs=25, verbose=True, validation_split=0.2)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
print(confusion_matrix(y_test, y_pred))
