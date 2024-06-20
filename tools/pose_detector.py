import numpy as np
from tools import data
from constants import FACE_KEYPOINTS, POSE_DICT
from tools.person import Person
from state import SystemState
import pyzed.sl as sl

class PoseDetector():
    def __init__(self, model) -> None:
        self.model = model
        self.persons = {}
        self.system_state = SystemState()
        self.confidence_threshold = 0.8
        
    def clean_persons(self, bodies):
        ids = [body.id for body in bodies.body_list]

    def clean_persons(self, bodies):
        ids = [body.id for body in bodies.body_list]
        return {k: v for k, v in self.persons.items() if k in ids}

    def clear_persons_except(self, id: int):
        return {k: v for k, v in self.persons.items() if k==id}

    def infere(self, body):
        keypoints = data.getKeypointsOfInterestFromBodyData(body.keypoint)
        predictions = self.model.call(keypoints)
        max_idx = np.argmax(predictions)
    
        return max_idx, predictions[0][max_idx]

    def detect(self, bodies: sl.Bodies):
        
        self.persons = self.clean_persons(bodies)
        
        if self.system_state.state != POSE_DICT["T-POSE"]:
            if self.system_state.focus_body_id in self.persons:

                person = self.persons[self.system_state.focus_body_id]
                body = sl.BodyData()
                bodies.get_body_data_from_id(body, self.system_state.focus_body_id)
                self.system_state.set_focus_body_bbox(body.bounding_box_2d)

                if not any(np.isnan(body.keypoint[id]).any() for id in FACE_KEYPOINTS):

                    prediction, confidence = self.infere(body)

                    if confidence > self.confidence_threshold:

                            focused_id = person.add_pose(prediction)

                            if focused_id > -1:
                                self.system_state.set_state(person.pose)
                                if person.pose not in [0,1]:
                                    self.system_state.set_focus_body_id(focused_id)                                        
                                else:
                                    self.system_state.set_focus_body_id(None)

                                person.add_pose(0)
                
            else:
                self.system_state.set_state(POSE_DICT["T-POSE"])
                self.system_state.set_focus_body_id(None)
            
        else:
            for body in bodies.body_list:

                if body.id not in self.persons:

                    self.persons[body.id] = Person(body.id)

                person = self.persons[body.id]

                if not any(np.isnan(body.keypoint[id]).any() for id in FACE_KEYPOINTS):

                    prediction, confidence = self.infere(body)

                    if confidence > self.confidence_threshold:

                        focused_id = person.add_pose(prediction)

                        if focused_id > -1:
                            self.system_state.set_state(person.pose)
                            if person.pose != 1:
                                self.system_state.set_focus_body_id(focused_id)
                            person.add_pose(0)
                    
                else:
                    self.persons[body.id].add_pose(POSE_DICT["NO POSE"])
        
        return self.persons