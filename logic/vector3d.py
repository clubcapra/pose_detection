from math import pow, sqrt

class Vector3D:
    def __init__(self, dx, dy, dz) -> None:
        self.dx = dx
        self.dy = dy
        self.dz = dz
    
    def getNorm(self) -> float:

        return sqrt(pow(self.dx, 2) + pow(self.dy, 2) + pow(self.dz, 2))