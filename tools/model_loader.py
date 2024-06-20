from constants import MODEL_PATH, CHECKSUM_PATH
import keras
import hashlib
import numpy as np

model_path = MODEL_PATH
checksum_path = CHECKSUM_PATH

def model_weights_checksum(model):
    weights = model.get_weights()
    weights_concat = np.concatenate([w.flatten() for w in weights])
    return hashlib.md5(weights_concat).hexdigest()

def get_model():
  
    print("Loading model...")
    model = keras.saving.load_model(model_path)
    checksum_after = model_weights_checksum(model)
    with open(checksum_path, 'r') as f:
      checksum_before = f.read().strip()

    assert checksum_before == checksum_after, "Non matching checksum - weights loaded incorrectly"
    print("The model has been loaded correctly!")
    
    return model