import numpy as np
from vector2d import Vector2D
from math import acos, degrees


def getAngleBetweenVectors(vector1: Vector2D, vector2: Vector2D) -> float:
    x = (dot(vector1, vector2)) / (vector1.getNorm() * vector2.getNorm())
    x = min(1, max(-1, x))
    return degrees(acos(x))

def getVectorFromPoints(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    return Vector2D(dx, dy)

def dot(vector1: Vector2D, vector2: Vector2D) -> float:

    return ((vector1.dx * vector2.dx) + (vector1.dy * vector2.dy))
    
