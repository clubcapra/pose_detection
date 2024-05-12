from vector3d import Vector3D
from math import acos, degrees


def getAngleBetweenVectors(vector1: Vector3D, vector2: Vector3D) -> float:
    x = (dot(vector1, vector2)) / (vector1.getNorm() * vector2.getNorm())
    x = min(1, max(-1, x))
    return degrees(acos(x))

def getVectorFromPoints(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dz = point2[2] - point1[2]
    
    return Vector3D(dx, dy, dz)

def dot(vector1: Vector3D, vector2: Vector3D) -> float:

    return ((vector1.dx * vector2.dx) + (vector1.dy * vector2.dy) + (vector1.dz * vector2.dz))
    
