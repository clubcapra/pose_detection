import os
import json
import numpy as np
import glob


def extract_keypoints(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    keypoints_list = []
    for instance_id, instance_data in data.items():
        for body in instance_data["body_list"]:
            keypoints = np.array(body["keypoint"])
            keypoints_list.append(keypoints)

    return keypoints_list


def process_json_files(directory):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    data = []
    labels = []

    for json_file in json_files:
        file_name = os.path.basename(json_file)
        label = file_name[:-6]
        keypoints_list = extract_keypoints(json_file)

        for keypoints in keypoints_list:
            # Create a new array with an additional dimension for the label
            # keypoints = np.nan_to_num(keypoints)
            if not np.isnan(keypoints).any():
              data.append(keypoints)
              labels.append(label)

    return np.array(data), np.array(labels)


def process_dataset(directory):
    data, labels = process_json_files(directory)
    return data, labels


