from json_extractor import process_dataset
from logic.pose_detection import getAnglesFromBodyData
from utils.data import convertToOneHot, convertLabelsToInt, convertSoftmaxToIndex
import numpy as np
from mlp import MLP
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

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

# Initialize model
model = MLP()
# Compile model
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalCrossentropy()],
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)
# Training
model.fit(X_train, y_train, epochs=50, verbose=True, validation_split=0.2)
# Save resulting weights
model.save_weights("weights/weights.weights.h5", overwrite=True)

# Make prediction (we do this instead of evaluate to have access to result vector)
y_pred = model.predict(X_test)

y_pred = convertSoftmaxToIndex(y_pred)
y_test = convertLabelsToInt(y_test)

cfm = confusion_matrix(y_test, y_pred)
classes = ["none", "tpose", "bucket", "skyward"]

df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True, fmt=".1f")
cfm_plot.figure.savefig("metrics/cfm.png")

