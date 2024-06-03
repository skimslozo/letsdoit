import itertools
import numpy as np

from copy import deepcopy
from human_id import generate_id

from typing import List, Dict, Tuple, Optional
from letsdoit.scoring.primitive import SpatialPrimitivePair, SpatialPrimitive, ObjectType, get_primitive, get_remaining_labels, check_object_type
from letsdoit.utils.object_instance import ObjectInstance
from letsdoit.utils.object_3d import Object3D
from letsdoit.utils.misc import get_instances

class GraphNode:
    def __init__(self, primitives: List[Dict], all_objects: List[Object3D], 
                 root: bool = False, object: Object3D = None, 
                 parent: Optional['GraphNode'] = None, 
                 child_num: int = None, used_indices: Optional[set] = None):
        self.parent = parent
        self.object = object
        self.primitives = primitives
        self.children = []
        self.edges = []
        self.root = root
        self.id = generate_id()
        self.all_objs = all_objects
        self.used_indices = used_indices if used_indices is not None else set()
        if self.object is not None:
            self.used_indices.add(self.all_objs.index(self.object))
        if parent:
            self.score = parent.score
            self.level = parent.level + 1
            self.label = self.object.label
        else:
            self.score = 0
            self.level = 0
            self.label = 'root'
        self._best_score = -np.inf  # Use a private attribute for best_score
        self.child_num = child_num

    @property
    def best_score(self):
        return self._best_score

    @best_score.setter
    def best_score(self, value):
        if value > self._best_score:
            # print(f'Updating best score for {self.label}_{self.id} from {self._best_score} to {value}')
            self._best_score = value
            if self.parent:
                self.parent.best_score = value  # Propagate the update to the parent

    def expand(self, children: List[Object3D] = None):
        if self.parent:
            parent_label = self.parent.label
        else:
            parent_label = 'None'
        # print(f'In expand, level {self.level}, parent: {parent_label}, label: {self.label}, child_number: {self.child_num}')
        
        if self.root:
            for i, child in enumerate(children):
                child_index = self.all_objs.index(child)
                if child_index in self.used_indices:
                    continue  # Skip already used objects
                # print(f'Expanding root child {i}/{len(children)}')
                self.edges.append('starting_edge')
                cn = GraphNode(primitives=deepcopy(self.primitives),
                               all_objects=self.all_objs,
                               object=child,
                               parent=self,
                               child_num=i,
                               used_indices=self.used_indices.copy())
                self.children.append(cn)
                cn.expand()
                cn.score_check()
            self.score_check()  # Ensure to call after all children are expanded
        else:
            pi = self._check_if_in_primitives(self.object.label)
            if pi is not None:
                prim = self.primitives.pop(pi)
                otype = check_object_type(self.object.label, prim)

                if get_primitive(prim['primitive']) == SpatialPrimitive.BETWEEN:
                    # print('Handling BETWEEN primitive')
                    c1_instances, c1_otype, c2_instances, c2_otype = self._create_children_instances(otype, prim)
                    child_combs = itertools.product(c1_instances, c2_instances)
                    for i, (c1, c2) in enumerate(child_combs):
                        c1_index = self.all_objs.index(c1)
                        c2_index = self.all_objs.index(c2)
                        if c1_index in self.used_indices or c2_index in self.used_indices:
                            continue  # Skip already used objects
                        objs = [(otype, self.object), (c1_otype, c1), (c2_otype, c2)]
                        spp = SpatialPrimitivePair.from_list(objs)
                        self.edges.append(spp)
                        
                        cn1 = GraphNode(primitives=deepcopy(self.primitives),
                                        all_objects=self.all_objs,
                                        object=c1,
                                        parent=self,
                                        child_num=2*i,
                                        used_indices=self.used_indices.copy())
                        self.children.append(cn1)
                        cn1.score += spp.score

                        cn2 = GraphNode(primitives=deepcopy(self.primitives),
                                        all_objects=self.all_objs,
                                        object=c2,
                                        parent=self,
                                        child_num=2*i+1,
                                        used_indices=self.used_indices.copy())
                        self.children.append(cn2)
                        cn2.score += spp.score

                        cn1.expand()
                        cn2.expand()
                        cn1.score_check()
                        cn2.score_check()
                    self.score_check()  # Ensure to call after all children are expanded

                else:
                    children_instances, child_otype = self._create_children_instances(otype, prim)
                    for i, child in enumerate(children_instances):
                        child_index = self.all_objs.index(child)
                        if child_index in self.used_indices:
                            continue  # Skip already used objects
                        objs = [(otype, self.object), (child_otype, child)]
                        spp = SpatialPrimitivePair.from_list(objs, prim)
                        self.edges.append(spp)
                        cn = GraphNode(primitives=deepcopy(self.primitives),
                                       all_objects=self.all_objs,
                                       object=child,
                                       parent=self,
                                       child_num=i,
                                       used_indices=self.used_indices.copy())
                        self.children.append(cn)
                        cn.score += spp.score
                        # print(f'Current primitive score: {spp.score}, child score: {cn.score}')
                        cn.expand()
                        cn.score_check()
                    self.score_check()  # Ensure to call after all children are expanded
            else:
                return
                
    def _create_children_instances(self, otype: ObjectType, primitive: Dict):
        prim = get_primitive(primitive['primitive'])
        if prim == SpatialPrimitive.BETWEEN:
            child_otypes, child_labels = get_remaining_labels(otype, primitive)
            ci1 = get_instances(child_labels[0], self.all_objs)
            ci2 = get_instances(child_labels[1], self.all_objs)
            return ci1, child_otypes[0], ci2, child_otypes[2]
        else:
            child_otype, child_label = get_remaining_labels(otype, primitive)
            child_instances = get_instances(child_label, self.all_objs)
        return child_instances, child_otype

    def _check_if_in_primitives(self, label) -> int | None:
        p_idx = None
        for i, p in enumerate(self.primitives):
            if (label in p['target_object']) or (label in p['reference_object']) or ((get_primitive(p['primitive']) == SpatialPrimitive.BETWEEN) and (label['reference_object_2'])):
                p_idx = i
                break
        return p_idx

    def score_check(self):
        if self.parent is None:
            return
        # print(f'In score_check. Parent: {self.parent.label}_{self.parent.id}, Child: {self.label}_{self.id}, child_num: {self.child_num}, child score: {self.score}, parent best score: {self.parent.best_score}')
        
        if self.score > self.parent.best_score:
            self.parent.best_score = self.score  # Use the setter to update best_score
            self.parent.best_child_index = self.child_num  # Update the best child index


def retrieve_best_action_object(instruction: dict, objects: List[Object3D]) -> Object3D:
    """
    Attributes:
        instruction (dict): dict extracted from the json with the instructions from GPT4
        objects (List[Object3D]): list of object_3d

    Return:
        best_action_object (Object3D): best action object
    """


    root = GraphNode(primitives=instruction['spatial_primitives'],
                    all_objects=objects,
                    root=True)
    action_instances = get_instances(instruction['action_object'], objects)
    root.expand(action_instances)

    node = root
    if len(node.children) == 0:
        return None
    best_action_object_idx = np.argmax([child.best_score for child in node.children])
    best_action_object = node.children[best_child].object
    return best_action_object
    