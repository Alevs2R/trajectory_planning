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
    dq_abs = np.abs(dq)
    # dq is a figure area, max_joint_velocity is a height
    t_ba = np.around(dq_abs / max_joint_velocity, 2)
    # t_b is a time for which acceleration > 0 and constant
    t_b = np.around(max_joint_velocity / max_joint_acceleration, 2)

    if np.any(t_b < t_ba):
        # trapezium
        t_a = t_ba - t_b  # t_a is a time of constant velocity
        t_a_max = np.amax(t_a)

        # we need the velocity plots of each joint to look like trapeziums of equal length
        each_joint_velocity = dq / (t_a_max + t_b)

        time = np.zeros(shape=(dq.shape[0], 4))
        time[:, 0] = 0
        time[:, 1] = t_b
        time[:, 2] = t_a_max + t_b
        time[:, 3] = t_a_max + 2 * t_b

        v = np.zeros(shape=(dq.shape[0], 4))
        v[:, 0] = 0
        v[:, 1] = each_joint_velocity
        v[:, 2] = each_joint_velocity
        v[:, 3] = 0

        return time, v
    else:
        # triangle
        # dq = t_b * v_max
        # t_b = v_max / acc => dq = t_b ^ 2 * acc
        t_b = np.around(np.sqrt(dq_abs/max_joint_acceleration), 2)
        t_b_max = np.amax(t_b)
        each_joint_velocity = dq / t_b_max

        time = np.zeros(shape=(dq.shape[0], 3))
        time[:, 0] = 0
        time[:, 1] = t_b_max
        time[:, 2] = 2 * t_b_max

        v = np.zeros(shape=(dq.shape[0], 3))
        v[:, 0] = 0
        v[:, 1] = each_joint_velocity
        v[:, 2] = 0

        return time, v


# draws 3 plots since articulated RRR robot has 3 joints
def velocity_plot(time, v):
    plt.ylabel('velocity, rad/s')
    plt.xlabel('time, ms')
    plt.title('joint velocities')

    for i in range(0, time.shape[0]):
        x1 = time[i, 0:-1]
        x2 = time[i, 1:]
        y1 = v[i, 0:-1]
        y2 = v[i, 1:]
        plt.plot(x1, y1, x2, y2, label='Joint '+str(i+1),)
    plt.legend()
    plt.show()


# def junction(each_joint_velocity,)

# J = jacobian()
# print(J)

q0 = np.array([0.1, -0.2, 0.3])
qf = np.array([0.2, -2, 0.6])

velocity_plot(*ptp_trajectory(q0, qf))
