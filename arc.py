import numpy as np


def getArcBy3Points(A, B, C):
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)
    s = (a + b + c) / 2
    R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
    P /= b1 + b2 + b3


def rotation_matrix_3d(axis, theta):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = axis * np.sin(theta / 2.0)
    rot = np.array([[a*a+b*b-c*c-d*d,     2*(b*c-a*d),     2*(b*d+a*c)],
                    [    2*(b*c+a*d), a*a+c*c-b*b-d*d,     2*(c*d-a*b)],
                    [    2*(b*d-a*c),     2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    return rot