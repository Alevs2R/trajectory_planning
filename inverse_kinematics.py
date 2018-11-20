import numpy as np

def inv_kin(pos, L):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    q1 = np.arctan2(y, x)
    c3 = (x ** 2 + y ** 2 + (z - L[0]) ** 2 - L[1] ** 2 - L[2] ** 2) / (2 * L[1] * L[2])
    s3 = - (np.sqrt(1 - c3 ** 2))
    q2 = np.arctan2(z - L[0], np.sqrt(x ** 2 + y ** 2)) - np.arctan2(L[2] * s3, L[1] + L[2] * c3)
    q3 = np.arctan2(s3, c3)
    return [q1, q2, q3]