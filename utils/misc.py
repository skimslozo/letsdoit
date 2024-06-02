import numpy as np
from typing import List
#from letsdoit.utils.object_instance import ObjectInstance  # removing this because of Circular Import error

def inverseRigid(H):
    H_R = H[0:3, 0:3]
    H_T = H[0:3, 3]

    invH = np.eye(4)
    invH[0:3, 0:3] = H_R.T
    invH[0:3, 3] = -H_R.T @ H_T

    return invH

def select_ids(array, ids):
    return [array[idx] for idx in ids]

def sample_points(points, n=1000):
    # points have shape (3, n_points)
    return points[:, np.random.randint(0, points.shape[1], size=n)]

#def get_instances(object: str, all_objects: List[ObjectInstance]) -> List[ObjectInstance]:
def get_instances(object: str, all_objects: List) -> List:  # modified because of Circular Import Error
    # all_objects is a list of object instances
    return [obj for obj in all_objects if obj.label==object]