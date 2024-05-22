import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

def generateTraces(history, model=None, y_test=None, y_pred=None):
  trainingNb = getNumberOfTrainings()
  generateNewTrainingDir(trainingNb)
  generateMetrics(history, trainingNb, y_test, y_pred)
  if(model is not None):
    model.save_weights(f"trainings/training{trainingNb}/weights/weights{trainingNb}.weights.h5", overwrite=True)

def generateMetrics(history, trainingNb, y_test=None, y_pred=None):
  generateLossGraph(history, trainingNb)
  if all([y_test is not None, y_pred is not None]):
    generateCfm(trainingNb, y_test, y_pred)

def generateNewTrainingDir(trainingNb):
  # Define the base directory
  base_dir = f'trainings/training{trainingNb}'

  # Define the full directory structure
  directories = [
      os.path.join(base_dir, 'metrics', 'loss'),
      os.path.join(base_dir, 'metrics', 'cfm'),
      os.path.join(base_dir, 'weights')
  ]

  # Create each directory in the list
  for directory in directories:
      os.makedirs(directory, exist_ok=True)

def getNumberOfTrainings():
  return len(os.listdir('trainings/'))

def generateLossGraph(history, trainingNb):
  training_loss = history.history['loss']
  validation_loss = history.history['val_loss']
  plt.figure(figsize=(10, 6))
  plt.plot(training_loss, label='Training Loss')
  plt.plot(validation_loss, label='Validation Loss')
  plt.title('Training and Validation Loss Over Epochs')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)
  plt.savefig(f"trainings/training{trainingNb}/metrics/loss/loss{trainingNb}.png")

def generateCfm(trainingNb, y_test, y_pred):
  cfm = confusion_matrix(y_test, y_pred)
  classes = ["none", "tpose", "bucket", "skyward"]

  df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
  plt.figure(figsize = (10,7))
  cfm_plot = sn.heatmap(df_cfm, annot=True, fmt=".1f")
  cfm_plot.figure.savefig(f"trainings/training{trainingNb}/metrics/cfm/cfm{trainingNb}.png")