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


def number2str(i: int, n_digits: int=3):

    assert isinstance(i, int), 'i must be an int!'

    # count how many digits does i already have
    k = i
    i_digits = 1
    while k >= 10:
        k /= 10
        i_digits += 1

    n_digits = max(i_digits, n_digits)
    diff_digits = n_digits - i_digits

    str_number = '0' * diff_digits + str(i)

    return str_number