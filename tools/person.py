import numpy as np
from scipy.ndimage import shift
from constants import POSE_DICT
from state import SystemState

class Person():
    def __init__(self, id):
        self.small_stack = np.full(10, None)
        self.big_stack = None
        self.interest = False
        self.focus = False
        self.id = id
        self.system_state = SystemState()

    def add_pose(self, pose_id):
        if self.interest:
            self.big_stack = shift(self.big_stack, 1, cval=pose_id)
            self.check_big_stack()
        else:
            self.small_stack = shift(self.small_stack, 1, cval=pose_id)
            self.check_small_stack()

    def check_big_stack(self):
        if np.all(self.big_stack == self.big_stack[0] and self.big_stack[0] == POSE_DICT.fromkeys("NO POSE")):
            self.focus = True
            self.system_state.set_state(POSE_DICT.get(self.big_stack[0]))
            self.big_stack = None

    
    def check_small_stack(self):
        if np.all(self.small_stack == self.small_stack[0] and self.small_stack[0] == POSE_DICT.fromkeys("NO POSE")):
            self.big_stack = np.full(100, None)
            self.big_stack = shift(self.big_stack, 10, cval=self.small_stack[0])
            self.interest = True

    def get_pose(self):
        if self.interest:
            return self.big_stack[0]
        
        return self.small_stack[0]
    
    def get_id(self):
        return self.id
    




