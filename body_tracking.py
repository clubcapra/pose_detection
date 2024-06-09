########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
from typing import Collection, Dict
import cv2
import sys
import pyzed.sl as sl
from sympy import E
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse
import csv
from tools import data
import keras
import hashlib
from constants import PREDICTION_OUTPUT_DICT, STATE_DICT, FACE_KEYPOINTS, POSE_DICT
from tools.person import Person
from state import SystemState
import time


CONFIDENCE_THRESHOLD = 0.8
model_path = './trainings/training80_4class/model/model80.keras'
checksum_path = './trainings/training80_4class/model/weights_checksum80.txt'

def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")

def model_weights_checksum(model):
    weights = model.get_weights()
    weights_concat = np.concatenate([w.flatten() for w in weights])
    return hashlib.md5(weights_concat).hexdigest()

def display_tracking_id(image, person_id):
    # Get the image dimensions
    height, width, _ = image.shape

    # Prepare the text to be displayed
    tracking_text = f'TRACKING PERSON {person_id}'

    # Define font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)  # Red color in BGR
    font_thickness = 2

    # Calculate text size to position it at the top center
    text_size = cv2.getTextSize(tracking_text, font, font_scale, font_thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = text_size[1] + 50  # Adding some padding from the top

    # Draw the text on the image
    image = cv2.putText(image, tracking_text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    return image

def display_persons_with_poses(persons: Dict, image):
    y = 40
    max_time = 4  # Maximum time for the progress bar to reach 100%

    # Calculate the height of the dynamic box
    box_height = 30 * len(persons) + 20
    box_y2 = y + box_height

    # Define the box coordinates
    box_x1 = 30
    box_y1 = 20
    box_x2 = 450

    # Create a blurred and darkened version of the box area
    box_area = image[box_y1:box_y2, box_x1:box_x2]
    blurred_box_area = cv2.GaussianBlur(box_area, (15, 15), 0)
    darkened_box_area = cv2.addWeighted(blurred_box_area, 0.4, np.zeros_like(blurred_box_area), 0.6, 0)

    # Replace the area in the original image with the darkened, blurred area
    image[box_y1:box_y2, box_x1:box_x2] = darkened_box_area

    if len(persons) > 0:
        for k, v in persons.items():
            if v.start_time is not None:
                current_time = time.time()
                elapsed_time = current_time - v.start_time
                progress = min(elapsed_time / max_time, 1)  # Ensure the progress doesn't exceed 1 (100%)

                # Convert progress to the width of the progress bar (max width is 100 pixels)
                progress_width = int(progress * 100)

                # Calculate the remaining time
                remaining_time = max_time - elapsed_time
                remaining_time = max(remaining_time, 0)  # Ensure remaining time doesn't go below 0

                # Draw the progress bar
                progress_bar_start = (250, y - 10)  # Adjust the position as needed
                progress_bar_end = (250 + progress_width, y + 10)
                image = cv2.rectangle(image, progress_bar_start, progress_bar_end, (0, 255, 0), -1)  # Filled rectangle for progress

                # Draw the outline of the progress bar
                image = cv2.rectangle(image, (250, y - 10), (350, y + 10), (255, 255, 255), 1)  # Outline of the progress bar

                # Draw the remaining time text
                image = cv2.putText(image, f"{remaining_time:.1f}s left", (360, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw the person's ID and pose text
            image = cv2.putText(image, f"{v.id} - Pose: {PREDICTION_OUTPUT_DICT.get(v.pose)}", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            y += 30
    
    else:
        image = cv2.putText(image, f"No bodies detected", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image

def display_system_state(image, system_state):
    # Get the image dimensions
    height, width, _ = image.shape

    # Prepare the text to be displayed
    state_text = f'State: {STATE_DICT[system_state.state]}'
    focus_text = f'Focus Body ID: {system_state.focus_body_id}'

    # Define font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # Red color in BGR
    font_thickness = 1

    # Calculate text size to position it at the bottom right corner
    state_text_size = cv2.getTextSize(state_text, font, font_scale, font_thickness)[0]
    focus_text_size = cv2.getTextSize(focus_text, font, font_scale, font_thickness)[0]

    # Set the positions for the text
    state_text_x = width - state_text_size[0] - 45
    state_text_y = 50
    focus_text_x = width - focus_text_size[0] - 45
    focus_text_y = 75

    # Define the box coordinates
    box_x1 = min(state_text_x, focus_text_x) - 20
    box_y1 = min(state_text_y - state_text_size[1], focus_text_y - focus_text_size[1]) - 20
    box_x2 = width - 20
    box_y2 = 100

    # Create a blurred and darkened version of the box area
    box_area = image[box_y1:box_y2, box_x1:box_x2]
    blurred_box_area = cv2.GaussianBlur(box_area, (15, 15), 0)
    darkened_box_area = cv2.addWeighted(blurred_box_area, 0.4, np.zeros_like(blurred_box_area), 0.6, 0)

    # Replace the area in the original image with the darkened, blurred area
    image[box_y1:box_y2, box_x1:box_x2] = darkened_box_area

    # Draw the text on the image
    image = cv2.putText(image, state_text, (state_text_x, state_text_y), font, font_scale, font_color, font_thickness)
    image = cv2.putText(image, focus_text, (focus_text_x, focus_text_y), font, font_scale, font_color, font_thickness)

    return image

def clean_persons(bodies, persons: Dict):
    ids = [body.id for body in bodies.body_list]
    return {k: v for k, v in persons.items() if k in ids}

def clear_persons_except(id: int, persons: Dict):
    return {k: v for k, v in persons.items() if k==id}

def infere(body, mlp):
    keypoints = data.getKeypointsOfInterestFromBodyData(body.keypoint)
    predictions = mlp.call(keypoints)
    max_idx = np.argmax(predictions)
    
    return max_idx, predictions[0][max_idx]


def main():
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    mlp = keras.saving.load_model(model_path)

    checksum_after = model_weights_checksum(mlp)

    with open(checksum_path, 'r') as f:
      checksum_before = f.read().strip()

    assert checksum_before == checksum_after, "The weights have not been loaded correctly!"
    print("The weights have been loaded correctly!")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    parse_args(init_params)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                # Track people across images flow
    body_param.enable_body_fitting = False            # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST 
    body_param.body_format = sl.BODY_FORMAT.BODY_18  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                 , display_resolution.height / camera_info.camera_configuration.resolution.height]

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, body_param.enable_tracking,body_param.body_format)
    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10 

    persons = {}
    system_state = SystemState()

    with open('keypoint_data.csv', 'a', newline='') as csvfile:
        fieldnames = ['Right shoulder', 'Right elbow', 'Left shoulder', 'Left elbow', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
        while viewer.is_available():
            # Grab an image
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Retrieve left image
                zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                # Retrieve bodies
                zed.retrieve_bodies(bodies, body_runtime_param)
                # Update GL view
                viewer.update_view(image, bodies) 
                # Update OCV view
                image_left_ocv = image.get_data()
                # Write the text on the image

                persons = clean_persons(bodies, persons)

                if system_state.state != POSE_DICT["T-POSE"]:
                    if system_state.focus_body_id in persons:

                        display_tracking_id(image_left_ocv, system_state.focus_body_id)

                        prediction, confidence = infere(persons[system_state.focus_body_id].body, mlp)

                        if confidence > CONFIDENCE_THRESHOLD:

                                focused_id = person.add_pose(prediction)

                                if focused_id > -1:
                                    system_state.set_state(person.pose)
                                    system_state.set_focus_body_id(focused_id)
                                    persons = clear_persons_except(focused_id, persons)
                        
                    else:
                        system_state.set_state(POSE_DICT["T-POSE"])
                    
                else:
                    for body in bodies.body_list:

                        if body.id not in persons:

                            persons[body.id] = Person(body.id, body)

                        person = persons[body.id]

                        if not any(np.isnan(body.keypoint[id]).any() for id in FACE_KEYPOINTS):

                            prediction, confidence = infere(body, mlp)

                            if confidence > CONFIDENCE_THRESHOLD:

                                focused_id = person.add_pose(prediction)

                                if focused_id > -1:
                                    system_state.set_state(person.pose)
                                    if person.pose != 1:
                                        system_state.set_focus_body_id(focused_id)
                                        persons = clear_persons_except(focused_id, persons)
                            
                        else:
                            persons[body.id].add_pose(POSE_DICT["NO POSE"])

                display_persons_with_poses(persons, image_left_ocv)
                display_system_state(image_left_ocv, system_state)

                cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
                # cv2.putText(image_left_ocv, PREDICTION_OUTPUT_DICT, POSE_DICT[pose], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("ZED | 2D View", image_left_ocv)
                key = cv2.waitKey(key_wait)
                if key == 113: # for 'q' key
                    print("Exiting...")
                    break
                if key == 109: # for 'm' key
                    if (key_wait>0):
                        print("Pause")
                        key_wait = 0 
                    else : 
                        print("Restart")
                        key_wait = 10

    
    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 