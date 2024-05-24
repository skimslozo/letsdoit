import numpy as np

class SpatialPrimitive(Enum):
    ABOVE = 1
    BELOW = 2
    FRONT = 3
    BEHIND = 4
    RIGHT = 5
    LEFT = 6
    CONTAINS = 7
    NEXT_TO = 8
    BETWEEN = 9

def get_primitive(primitive_str: str) -> SpatialPrimitive:
    out = None
    match primitive_str:
        case 'above':
            out = SpatialPrimitive.ABOVE
        case 'below':
            out = SpatialPrimitive.BELOW
        case 'in front of':
            out = SpatialPrimitive.FRONT
        case 'behind':
            out = SpatialPrimitive.BEHIND
        case 'to the right':
            out = SpatialPrimitive.RIGHT
        case 'to the left':
            out = SpatialPrimitive.LEFT
        case 'contains':
            out = SpatialPrimitive.CONTAINS
        case 'next to':
            out = SpatialPrimitive.NEXT_TO
        case 'between':
            out = SpatialPrimitive.BETWEEN
    return out

def inverseRigid(H):
    H_R = H[0:3, 0:3]
    H_T = H[0:3, 3]

    invH = np.eye(4)
    invH[0:3, 0:3] = H_R.T
    invH[0:3, 3] = -H_R.T @ H_T

    return invH

