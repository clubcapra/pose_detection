import cv2
from typing import Dict
import time
import numpy as np
from constants import PREDICTION_OUTPUT_DICT, STATE_DICT, POSE_DICT

def cvt(pt, scale):
    out = [pt[0]*scale[0], pt[1]*scale[1]]
    return out

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

    regular_color = (0, 255, 0)
    focused_color = (0, 0, 255)

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
                image = cv2.rectangle(image, progress_bar_start, progress_bar_end, focused_color if v.focus else regular_color , -1)  # Filled rectangle for progress

                # Draw the outline of the progress bar
                image = cv2.rectangle(image, (250, y - 10), (350, y + 10), (255, 255, 255), 1)  # Outline of the progress bar

                # Draw the remaining time text
                image = cv2.putText(image, f"{remaining_time:.1f}s left", (360, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, focused_color if v.focus else regular_color, 1)

            # Draw the person's ID and pose text
            image = cv2.putText(image, f"{v.id} - Pose: {PREDICTION_OUTPUT_DICT.get(v.pose)}", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, focused_color if v.focus else regular_color, 1)

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

def draw_2d_bounding_box(image, bounding_box, scale=[1,1]):
    
    """
    Scale is given by formula: 
    
    [ display_res.width / camera_config_res.width , display_res.height / camera_config_res.height ]
    """    

    pt1 = cvt(bounding_box[0], scale)
    pt2 = cvt(bounding_box[2], scale)

    pt1 = np.array(pt1, dtype=np.int32)
    pt2 = np.array(pt2, dtype=np.int32)

    image = cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)  # Red color for the box edges

    return image