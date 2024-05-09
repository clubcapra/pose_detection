from math import pow, sqrt

class Vector2D:
    def __init__(self, dx, dy) -> None:
        self.dx = dx
        self.dy = dy
    
    def getNorm(self) -> float:

        return sqrt(pow(self.dx, 2) + pow(self.dy, 2))
    


