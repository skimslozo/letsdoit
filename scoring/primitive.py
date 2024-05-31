import numpy as np
from enum import Enum

from letsdoit.utils.object_instance import ObjectInstance


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

class SpatialPrimitivePair:

    def __init__(self, 
                target: ObjectInstance, 
                reference: ObjectInstance, 
                primitive: SpatialPrimitive, 
                T_world_to_viewpoint: np.ndarray = np.eye(4), 
                norm_factor: float = 1,
                ):
        
        self.target = target
        self.reference = reference
        self.primitive = primitive
        self.T_world_to_viewpoint = T_world_to_viewpoint
        self.norm_factor = norm_factor
        assert ((T_world_to_viewpoint.shape[0] == 4) and (T_world_to_viewpoint.shape[1] == 4) and (len(T_world_to_viewpoint.shape) == 2))

    def _to_viewpoint(self, x):
        return self.T_world_to_viewpoint @ x

    def get_score(self):
        res = None
        match self.primitive:
            case SpatialPrimitive.ABOVE:
                pass
            case SpatialPrimitive.BELOW:
                pass
            case SpatialPrimitive.FRONT:
                pass
            case SpatialPrimitive.BEHIND:
                pass
            case SpatialPrimitive.RIGHT:
                pass
            case SpatialPrimitive.LEFT:
                pass
            case SpatialPrimitive.CONTAINS:
                pass
            case SpatialPrimitive.NEXT_TO:
                res = self.evaluate_next_to()
            case SpatialPrimitive.BETWEEN:
                pass
            case _:
                raise ValueError('Invalid spatial primitive provided')
        return res

    def evaluate_above(self):
        # Target z above reference
        dist = int(self.target.center_3d[2] > self.reference.center_3d[2])
        return dist
    
    def evaluate_below(self):
        # Target z below reference
        dist = int(self.target.center_3d[2] < self.reference.center_3d[2])
        return dist

    def evaluate_front(self):
        # Heuristic: the closer the target is to the reference, the more "in front" it is -> the less negative, the higher the score
        y_dist = self.target.center_3d[1] - self.reference.center_3d[1]
        return y_dist / self.norm_factor

    def evaluate_back(self):
        return -self.evaluate_front()
    
    def evaluate_right(self):
        # Assume now x-axis will match to general "right" in all cases, TODO: check if reasonable.
        return 

    def evaluate_next_to(self):
        dist = np.linalg.norm(self.target.center_3d - self.reference.center_3d)
        return dist / self.norm_factor

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
