from typing import List
import numpy as np

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
