from models.mlp import MLP
import keras
from tools.json_extractor import process_dataset
from logic.pose_detection import getAnglesFromBodyData
from tools.data import convertToOneHot, balanceDataset
import numpy as np
import keras
import hashlib

def model_weights_checksum(model):
  weights = model.get_weights()
  weights_concat = np.concatenate([w.flatten() for w in weights])
  return hashlib.md5(weights_concat).hexdigest()

mlp = keras.saving.load_model("./trainings/training28/model/model28.keras", custom_objects={'MLP': MLP})

checksum_after = model_weights_checksum(mlp)

with open("./trainings/training28/model/weights_checksum28.txt", 'r') as f:
  checksum_before = f.read().strip()

assert checksum_before == checksum_after, "The weights have not been loaded correctly!"
print("The weights have been loaded correctly!")



