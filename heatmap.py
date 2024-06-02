import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tools.json_extractor import process_dataset_group

directories = [
    "dataset/none1.json",
    "dataset/none2.json",
    "dataset/none3.json",
    "dataset/none4.json"
]

# Example skeleton data
X, y = process_dataset_group(directories)

# Normalize each skeleton to be centered around the origin
normalized_skeletons = []
for skeleton in X:
    skeleton = np.array(skeleton)
    centroid = np.mean(skeleton, axis=0)
    normalized_skeleton = skeleton - centroid
    normalized_skeletons.append(normalized_skeleton)

# Flatten the data and project to 2D (e.g., ignoring the z dimension for simplicity)
projected_points = []
for skeleton in normalized_skeletons:
    for point in skeleton:
        x, y, z = point
        projected_points.append([x, y])  # Projection by ignoring the z coordinate

projected_points = np.array(projected_points)

# Create a heatmap using a 2D histogram
heatmap, xedges, yedges = np.histogram2d(projected_points[:,0], projected_points[:,1], bins=50)

# Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap.T, cmap='viridis', cbar=True)
plt.title('Heatmap of Normalized 3D Skeleton Keypoints Projected to 2D')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.gca().invert_yaxis()

# Adding the number of skeletons used in the heatmap
num_skeletons = len(normalized_skeletons)
plt.annotate(f'Number of skeletons: {num_skeletons}', xy=(0.7, 1.05), xycoords='axes fraction', fontsize=12, color='white', backgroundcolor='black')

plt.show()
