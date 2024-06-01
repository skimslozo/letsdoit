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
                reference_2: SpatialPrimitive = None, # Only used if SpatialPrimitive is "between"
                ):
        
        self.target = target
        self.reference = reference
        self.reference_2 = reference_2
        self.primitive = primitive
        self.T_world_to_viewpoint = T_world_to_viewpoint
        self.norm_factor = norm_factor
        assert ((T_world_to_viewpoint.shape[0] == 4) and (T_world_to_viewpoint.shape[1] == 4) and (len(T_world_to_viewpoint.shape) == 2))

    def _to_viewpoint(self, x):
        return self.T_world_to_viewpoint @ x

    def get_score(self) -> int | float:
        res = None
        match self.primitive:
            case SpatialPrimitive.ABOVE:
                res = self._evaluate_above()
            case SpatialPrimitive.BELOW:
                res = self._evaluate_below()
            case SpatialPrimitive.FRONT:
                res = self._evaluate_front()
            case SpatialPrimitive.BEHIND:
                res = self._evaluate_behind()
            case SpatialPrimitive.RIGHT:
                res = self._evaluate_right()
            case SpatialPrimitive.LEFT:
                res = self._evaluate_left()
            case SpatialPrimitive.CONTAINS:
                res = self._evaluate_contains()
            case SpatialPrimitive.NEXT_TO:
                res = self._evaluate_next_to()
            case SpatialPrimitive.BETWEEN:
                res = self._evaluate_between()
            case _:
                raise ValueError('Invalid spatial primitive provided')
        return res

    def _evaluate_above(self) -> int:
        # Target z above reference (higher than reference)
        z_dist = int(self.target.center_3d[2] > self.reference.center_3d[2])
        return z_dist
    
    def _evaluate_below(self) -> int:
        # Target z below reference (lower than reference)
        z_dist = int(self.target.center_3d[2] < self.reference.center_3d[2])
        return z_dist

    def _evaluate_front(self) -> int:
        # Target y smaller than reference y (closer than reference)
        y_dist = int(self.target.center_3d[1] < self.reference.center_3d[1])
        return y_dist

    def _evaluate_behind(self) -> int:
        # Target y larger than reference y (further than reference)
        y_dist = int(self.target.center_3d[1] > self.reference.center_3d[1])
        return y_dist

    def _evaluate_right(self) -> int:
        # Target x larger than reference x
        x_dist = int(self.target.center_3d[0] > self.reference.center_3d[0])
        return x_dist

    def _evaluate_left(self) -> int:
        # Target x smaller than reference x
        x_dist = int(self.target.center_3d[0] < self.reference.center_3d[0])
        return x_dist

    def _evaluate_contains(self, eps: float=0.01, thershold=0.95) -> int:
        # epsilon - error on bounding box dimensions
        max_points = np.max(self.reference.mask_3d, axis=1).reshape(-1, 1) + eps
        min_points = np.min(self.reference.mask_3d, axis=1).reshape(-1, 1) - eps
        # Criterion from contains: at least 95% of points of target is within refernece 3d bbox
        larger = np.all(self.target.mask_3d > min_points, axis=0)
        smaller = np.all(self.target.mask_3d < max_points, axis=0)
        points_inside = larger*smaller
        contains = int((np.sum(points_inside) / len(points_inside)) > thershold)
        return contains

    def _evaluate_between(self, eps: float=0.1, thershold=0.7) -> int:
        # In between: the same as contains, but expand min_points and max_points with points from both references and a different epsilon and threshold
        ref_points = np.hstack([self.reference.mask_3d, self.reference_2.mask_3d])
        max_points = np.max(ref_points, axis=1).reshape(-1, 1) + eps
        min_points = np.min(ref_points, axis=1).reshape(-1, 1) - eps
        # Criterion from contains: at least 70% of points of target is within refernece 3d bbox
        larger = np.all(self.target.mask_3d > min_points, axis=0)
        smaller = np.all(self.target.mask_3d < max_points, axis=0)
        points_inside = larger*smaller
        contains = int((np.sum(points_inside) / len(points_inside)) > thershold)
        return contains

    def _evaluate_next_to(self) -> float:
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