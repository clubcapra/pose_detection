import matplotlib.pyplot as plt
import os

def generateLossGraph(history):
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
  loss_figures = os.listdir('metrics/losses/')
  plt.savefig(f"metrics/losses/loss{len(loss_figures)}.png")