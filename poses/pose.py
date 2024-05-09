from abc import ABC, abstractmethod
from typing import List

class Pose(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def detect(self, angles:List[float]) -> bool: ...
