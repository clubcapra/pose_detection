from constants import TPOSE_ANGLES, ERROR_MARGIN
from poses.pose import Pose

class TPose(Pose):
    def __init__(self) -> None:
        super().__init__()
        self.pose = TPOSE_ANGLES

    def detect(self, angles: list) -> bool:

        # Assumes a list of 4 angles
        if(len(angles) != 4):
            raise Exception("List of angles must be of size 4, got size ", len(angles))
        
        return all([target - ERROR_MARGIN <= angle <= target + ERROR_MARGIN for angle, target in zip(angles, self.pose)])


        