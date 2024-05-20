import numpy as np

pose_dict = {"none": 0, "tpose": 1, "bucket": 2, "skyward": 3}

def convertToOneHot(y, nb_outputs):
  oh_y = np.zeros((y.shape[0], nb_outputs))

  for i in range(y.shape[0]):
    index = pose_dict[y[i]]
    oh_y[i][index] = 1

  return oh_y

def convertLabelsToInt(y):
  y_int = np.zeros_like(y, dtype=int)

  for i in range(y.shape[0]):
    y_int[i] = pose_dict[y[i]]

  return y_int

def convertSoftmaxToIndex(y):
  return np.argmax(y, axis=1)