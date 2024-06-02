import os
import json
import numpy as np
import glob
from constants import KEYPOINT_COUPLES


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
    unique_keypoint_ids = set()
    for couple in KEYPOINT_COUPLES:
      unique_keypoint_ids.update(couple)

    for json_file in json_files:
        file_name = os.path.basename(json_file)
        label = file_name[:-6]
        keypoints_list = extract_keypoints(json_file)

        for keypoints in keypoints_list:
            # Create a new array with an additional dimension for the label
            # keypoints = np.nan_to_num(keypoints)
            if not any(np.isnan(keypoints[id]).any() for id in unique_keypoint_ids):
              data.append(keypoints)
              labels.append(label)

    return np.array(data), np.array(labels)

def process_json_file(directory):

    data = []
    labels = []
    unique_keypoint_ids = set()
    for couple in KEYPOINT_COUPLES:
      unique_keypoint_ids.update(couple)

    file_name = os.path.basename(directory)
    label = file_name[:-6]
    keypoints_list = extract_keypoints(directory)

    for keypoints in keypoints_list:
        if not any(np.isnan(keypoints[id]).any() for id in unique_keypoint_ids):
            data.append(keypoints)
            labels.append(label)

    return np.array(data), np.array(labels)


def process_dataset(directory):
    data, labels = process_json_files(directory)
    return data, labels

def process_dataset_single(directory):
    data, labels = process_json_file(directory)
    return data, labels

def process_dataset_except(directories):
    process_json_files


def process_dataset_group(directories, rest_as_none=False):
    data = []
    labels = []
    for directory in directories: 
        data_temp, labels_temp = process_dataset_single(directory)
        data.extend(data_temp)
        labels.extend(labels_temp)
    
    if rest_as_none:
        data_temp, labels_temps = 

    return np.array(data), np.array(labels)


