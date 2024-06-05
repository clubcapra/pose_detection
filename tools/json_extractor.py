import os
import json
import numpy as np
import glob
from constants import KEYPOINT_OF_INTEREST
from tools.data import normalize_skeleton


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
            if not any(np.isnan(keypoints[id]).any() for id in KEYPOINT_OF_INTEREST):
              keypoints = normalize_skeleton([keypoints[x] for x in KEYPOINT_OF_INTEREST])
              data.append(keypoints)
              labels.append(label)

    return np.array(data), np.array(labels)

def process_json_file(directory):
    data = []
    labels = []

    file_name = os.path.basename(directory)
    label = file_name[:-6]
    keypoints_list = extract_keypoints(directory)

    for keypoints in keypoints_list:
        if not any(np.isnan(keypoints[id]).any() for id in KEYPOINT_OF_INTEREST):
            keypoints = normalize_skeleton([keypoints[x] for x in KEYPOINT_OF_INTEREST])
            data.append(keypoints)
            labels.append(label)

    return np.array(data), np.array(labels)


def process_dataset(directory):
    data, labels = process_json_files(directory)
    return data, labels

def process_dataset_single(directory):
    data, labels = process_json_file(directory)
    return data, labels

def process_dataset_except(rest_directory, directories):
    json_files = glob.glob(os.path.join(rest_directory, "*.json"))
    data = []
    labels = []

    for json_file in json_files:
        if json_file not in directories:
          label = "none"
          keypoints_list = extract_keypoints(json_file)

          for keypoints in keypoints_list:
              # Create a new array with an additional dimension for the label
              # keypoints = np.nan_to_num(keypoints)
              if not any(np.isnan(keypoints[id]).any() for id in KEYPOINT_OF_INTEREST):
                keypoints = normalize_skeleton([keypoints[x] for x in KEYPOINT_OF_INTEREST])
                data.append(keypoints)
                labels.append(label)

    return np.array(data), np.array(labels)


def process_dataset_group(directories, rest_directory=None):
    data = []
    labels = []
    for directory in directories: 
        data_temp, labels_temp = process_dataset_single(directory)
        data.extend(data_temp)
        labels.extend(labels_temp)
    
    if rest_directory is not None:
        data_temp, labels_temp = process_dataset_except(rest_directory, directories)
        data.extend(data_temp)
        labels.extend(labels_temp)

    return np.array(data), np.array(labels)


