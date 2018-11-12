from sympy import *
import numpy as np
from elementary_transormations import *
import matplotlib.pyplot as plt


init_printing(use_unicode=True)

max_freq = 100
max_joint_velocity = 1
max_cartesian_velocty = 1
max_joint_acceleration = 1
max_cartesian_acceleration = 1
junction = 5 / max_freq


def jacobian_revolute(T_i, T_n):
    Z_i = T_i[0:3, 2]
    O_i = T_i[0:3, 3]
    O_n = T_n[0:3, 3]

    return simplify(Matrix(Z_i.cross(O_n - O_i).col_join(Z_i)))


def jacobian_prismatic(T_i, T_n):
    Z_i = T_i[0:3, 2]
    return simplify(Matrix(Z_i.col_join(Matrix([[0], [0], [0]]))))


# for artuculated RRR robot
def jacobian():
    q1, q2, q3, l1, l2, l3 = symbols('q1 q2 q3 l1 l2 l3')
    T0 = eye(4)
    T0_1 = rz(q1)
    T0_2 = T0_1 * tz(l1)
    T0_3 = T0_2 * rx(q2)
    T0_4 = T0_3 * tz(l2)
    T0_5 = T0_4 * rx(q3)
    T0_6 = T0_5 * tz(l3)

    J_1 = jacobian_revolute(T0, T0_6)
    J_2 = jacobian_revolute(T0_2, T0_6)
    J_3 = jacobian_revolute(T0_4, T0_6)

    return Matrix().row_join(J_1).row_join(J_2).row_join(J_3)


def ptp_trajectory(q0, qf):
    dq = qf - q0
    # dq is a figure area, max_joint_velocity is a height
    t_ba = np.around(dq / max_joint_velocity, 2)
    # t_b is a time for which acceleration > 0 and constant
    t_b = np.around(max_joint_velocity / max_joint_acceleration, 2)

    print(t_ba)
    print(t_b)

    if np.any(t_b < t_ba):
        # trapezium
        t_a = t_ba - t_b  # t_a is a time of constant velocity
        t_a_max = np.amax(t_a)

        # we need the velocity plots of each joint to look like trapeziums of equal length
        each_joint_velocity = dq / (t_a_max + t_b)

        return each_joint_velocity, t_b, t_a_max
    else:
        # triangle
        # dq = t_b * v_max
        # t_b = v_max / acc => dq = t_b ^ 2 * acc
        t_b = np.around(np.sqrt(dq/max_joint_acceleration), 2)
        t_b_max = np.amax(t_b)
        each_joint_velocity = dq / t_b_max
        return each_joint_velocity, t_b_max, 0


# draws 3 plots since articulated RRR robot has 3 joints
def velocity_plot(each_joint_velocity, t_b, t_a):
    t_a *= 100  # to present in milliseconds
    t_b *= 100

    for [i], v in np.ndenumerate(each_joint_velocity):
        plt.figure(i+1)
        plt.ylabel('velocity, rad/s')
        plt.xlabel('time, ms')
        plt.title('joint '+str(i+1))

        if t_a > 0:  # trapezium
            x1, y1 = [0, t_b, t_a+t_b], [0, v, v]
            x2, y2 = [t_b, t_a+t_b, t_a+2*t_b], [v, v, 0]
        else:  # triangle
            x1, y1 = [0, t_b], [0, v]
            x2, y2 = [t_b, 2 * t_b], [v, 0]
        plt.plot(x1, y1, x2, y2, marker = 'o')
        plt.show()

# J = jacobian()
# print(J)

q0 = np.array([0.1, 0.2, 0.3])
qf = np.array([0.2, 4, 0.6])

velocity_plot(*ptp_trajectory(q0, qf))
