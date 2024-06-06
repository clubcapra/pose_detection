import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.json_extractor import process_dataset_group
from logic.pose_detection import getAnglesFromBodyData

# Example angle data
directories = [
    "dataset/skyward1.json",
    "dataset/skyward1.json",
    "dataset/skyward4.json",
    "dataset/skyward5.json"
]

# Example skeleton data
X, y = process_dataset_group(directories)

X, y = np.array(X), np.array(y)

angles = []
for i in range(X.shape[0]):
    angles.append(getAnglesFromBodyData(X[i]))

# Flatten the list of lists to a single list of angles
all_angles = [angle for sublist in angles for angle in sublist]

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(all_angles, bins=np.arange(0, 181, 5), edgecolor='black', alpha=0.7)
plt.title('Distribution of Angles')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.xticks(np.arange(0, 181, 10))
plt.xlim(0, 180)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the histogram
plt.show()
