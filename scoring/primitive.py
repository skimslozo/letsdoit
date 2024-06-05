import copy

import numpy as np
from enum import Enum
from typing import Dict, List, Tuple

from pipeline.object_3d import Object3D, plot_objects_3d

class ObjectType(Enum):
    TARGET=0
    REFERENCE=1
    REFERENCE_2=2

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
                target: Object3D, 
                reference: Object3D, 
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
        self.score = self.get_total_score()

    @classmethod
    def from_list(cls, objects: List[Tuple[ObjectType, Object3D]], prim: Dict) -> 'SpatialPrimitivePair':
        t, r, r2 = None, None, None
        prim = get_primitive(prim['primitive'])
        for otype, ob3d in objects:
            if otype == ObjectType.TARGET:
                t = ob3d
            elif otype == ObjectType.REFERENCE:
                r = ob3d
            elif otype == ObjectType.REFERENCE_2:
                r2 = ob3d
        return cls(target=t, reference=r, primitive=prim, reference_2=r2)

    def _to_viewpoint(self, x):
        return self.T_world_to_viewpoint @ x

    def get_total_score(self):
        if self.primitive == SpatialPrimitive.BETWEEN:
            out = self.target.confidence * self.reference.confidence * self.get_spatial_score() * self.reference_2.confidence
        else:
            out = self.target.confidence * self.reference.confidence * self.get_spatial_score()
        return out
    
    def get_spatial_score(self) -> int | float:
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
        max_points = np.max(self.reference.points, axis=1).reshape(-1, 1) + eps
        min_points = np.min(self.reference.points, axis=1).reshape(-1, 1) - eps
        # Criterion from contains: at least 95% of points of target is within refernece 3d bbox
        larger = np.all(self.target.points > min_points, axis=0)
        smaller = np.all(self.target.points < max_points, axis=0)
        points_inside = larger*smaller
        contains = int((np.sum(points_inside) / len(points_inside)) > thershold)
        return contains

    def _evaluate_between(self, eps: float=0.1, thershold=0.7) -> int:
        # In between: the same as contains, but expand min_points and max_points with points from both references and a different epsilon and threshold
        ref_points = np.hstack([self.reference.points, self.reference_2.points])
        max_points = np.max(ref_points, axis=1).reshape(-1, 1) + eps
        min_points = np.min(ref_points, axis=1).reshape(-1, 1) - eps
        # Criterion from contains: at least 70% of points of target is within refernece 3d bbox
        larger = np.all(self.target.points > min_points, axis=0)
        smaller = np.all(self.target.points < max_points, axis=0)
        points_inside = larger*smaller
        contains = int((np.sum(points_inside) / len(points_inside)) > thershold)
        return contains

    def _evaluate_next_to(self) -> float:
        dist = np.linalg.norm(self.target.center_3d - self.reference.center_3d)
        return dist / self.norm_factor
    
    def plot_3d(self):
        t = copy.copy(t)
        t.label = f'TARGET | {t.label}'

        r = copy.copy(r)
        r.label = f'REFERENCE | {r.label}'

        objs = [t, r]
        
        if self.reference_2 is not None:
            r2 = copy.copy(self.reference_2)
            r2.label = f'REFERENCE 2 | {r2.label}'
            objs.append(r2)
        plot_objects_3d(objs)

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

def get_remaining_labels(known: ObjectType, p: Dict):
    '''Given known type, output the labels and ObjectTypes of the remaining objects in primitive'''
    ptype = get_primitive(p['primitive'])
    if ptype == SpatialPrimitive.BETWEEN:
        if known == ObjectType.TARGET:
            return [(ObjectType.REFERENCE, p['reference_object']), (ObjectType.REFERENCE_2, p['reference_object_2'])]

        elif known == ObjectType.REFERENCE:
            return [(ObjectType.TARGET, p['target_object']), (ObjectType.REFERENCE_2, p['reference_object_2'])]

        elif known == ObjectType.REFERENCE_2:
            return [(ObjectType.TARGET, ['target_object']), (ObjectType.REFERENCE, p['reference_object'])]

    else:
        if known == ObjectType.TARGET:
            return (ObjectType.REFERENCE, p['reference_object'])

        elif known == ObjectType.REFERENCE:
            return (ObjectType.TARGET, p['target_object'])

    
def check_object_type(label: str, primitive: Dict):
    type = None
    if label in primitive['target_object']:
        type = ObjectType.TARGET
    elif label in primitive['reference_object']:
        type = ObjectType.REFERENCE
    elif (get_primitive(primitive['primitive']) == SpatialPrimitive.BETWEEN) and (label in primitive['reference_object_2']):
        type = ObjectType.REFERENCE_2
    return type
