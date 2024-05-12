from typing import Collection, Dict
from math3d import getAngleBetweenVectors, getVectorFromPoints
from constants import VECTOR_COUPLES, KEYPOINT_COUPLES, TPOSE_ANGLES, SKYWARD_ANGLES, BUCKET_ANGLES
from poses.tpose import TPose
from poses.skyward import Skyward
from poses.bucket import Bucket
from poses.pose import Pose

class PoseType(enumerate):
    NONE = -1
    TPOSE = 0
    BUCKET = 1
    SKYWARD = 2

class PoseDectection:
    def __init__(self) -> None:
        pass

    
    def detectPose(self, bodyData:Collection[float]) -> PoseType:
        # Body data is an array of all 2d keypoints of a detected  body

        # Order of angles in list have this significance (by index):
        # 0 = angle between right shoulder and upper arm
        # 1 = angle between right upper arm and lower arm
        # 2 = angle between left shoulder and upperarm
        # 3 = angle between left upperarm and lowerarm

        # Start by getting vectors of interest
        vectors = self.getVectors(bodyData)

        angles = self.getAnglesFromVectors(vectors)

        # Find matching pose for found angles
        tposeDetector = TPose()
        skywardDetector = Skyward()
        bucketDetector = Bucket()
        detectors:Dict[PoseType, Pose] = {
            PoseType.TPOSE : tposeDetector, 
            PoseType.SKYWARD : skywardDetector, 
            PoseType.BUCKET : bucketDetector
        }

        for pose, detector in detectors.items():
            if detector.detect(angles):
                return pose
            
        return PoseType.NONE
        
    def getVectors(self, bodyData) -> list:

        vectors = list()

        # We're only interested in the keypoints with index
        # 1 (chest), 
        # 2 (right shoulder), 
        # 3 (right elbow),
        # 4 (right hand), 
        # 5 (left shoulder), 
        # 6 (left elbow), 
        # 7 (left hand).
        # This assumes the 18 point detection from StereoLabs

        # We therefore want to create 7 vectors to describe the system:
        # a1 = vector from center of chest to right shoulder
        # a2 = vector from right shoulder to right elbow
        # a3 = vector from right elbow to right wrist
        # a4 = vector from center chest to left shoulder
        # a5 = vector from left shoulder to left elbow
        # a6 = vector from left shoulder to left elbow
                    
        # For each couple, calculate vector and add to list
        for couple in KEYPOINT_COUPLES:
            vectors.append(getVectorFromPoints(bodyData.keypoint_2d[couple[0]], bodyData.keypoint_2d[couple[1]]))

        return vectors

    def getAnglesFromVectors(self, vectors): 

        angles = []

        # Calculate angles between vectors
        for couple in VECTOR_COUPLES:
            vector1 = vectors[couple[0]]
            vector2 = vectors[couple[1]]
            angles.append(getAngleBetweenVectors(vector1, vector2))

        return angles
    
    def getAnglesFromBodyData(self, bodyData:Collection[float]) -> PoseType:

        # Start by getting vectors of interest
        vectors = self.getVectors(bodyData)

        angles = self.getAnglesFromVectors(vectors)

        return angles
    

        
    

