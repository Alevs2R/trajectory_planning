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

q1, q2, q3, l1, l2, l3 = symbols('q1 q2 q3 l1 l2 l3')

robot_l1 = 1
robot_l2 = 4
robot_l3 = 3


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
        t_f = t_a_max + 2 * t_b

        # we need the velocity plots of each joint to look like trapeziums of equal length
        each_joint_velocity = dq / (t_a_max + t_b)
        each_joint_acceleration = each_joint_velocity / t_b

        # time = np.zeros(shape=(dq.shape[0], 4))
        # time[:, 0] = 0
        # time[:, 1] = t_b
        # time[:, 2] = t_a_max + t_b
        # time[:, 3] = t_a_max + 2 * t_b

        time = np.arange(0, t_f+0.005, 0.01)
        v = np.zeros(shape=(dq.shape[0], time.shape[0]))
        for (i,), cur_time in np.ndenumerate(time):
            if cur_time < t_b:
                v[:, i] = each_joint_acceleration * cur_time
            elif cur_time < t_a_max + t_b:
                v[:, i] = each_joint_velocity
            else:
                v[:, i] = each_joint_acceleration * (t_f - cur_time)

        return v
    else:
        # triangle
        # dq = t_b * v_max
        # t_b = v_max / acc => dq = t_b ^ 2 * acc
        t_b = np.around(np.sqrt(dq_abs/max_joint_acceleration), 2)
        t_b_max = np.amax(t_b)
        t_f = 2 * t_b_max
        each_joint_velocity = dq / t_b_max
        each_joint_acceleration = each_joint_velocity / t_b_max
        print(t_f)
        time = np.arange(0, t_f + 0.005, 0.01)
        v = np.zeros(shape=(dq.shape[0], time.shape[0]))
        for (i,), cur_time in np.ndenumerate(time):
            if cur_time < t_b_max:
                v[:, i] = each_joint_acceleration * cur_time
            else:
                v[:, i] = each_joint_acceleration * (t_f - cur_time)

        return v

def lin_trajectory(x0, xf):
    dist = np.linalg.norm(xf - x0)
    t_ba = np.around(dist / max_cartesian_velocty, 2)
    t_b = np.around(max_cartesian_velocty / max_cartesian_acceleration, 2)
    if t_b < t_ba:
        t_a = t_ba - t_b




# draws 3 plots since articulated RRR robot has 3 joints
def velocity_plot(v):
    plt.ylabel('velocity, rad/s')
    plt.xlabel('time, ms')
    plt.title('joint velocities')

    time = np.arange(0, v.shape[1] * 0.01 - 0.005, 0.01)

    for i in range(0, v.shape[0]):
        x1 = time[0:-1]
        x2 = time[1:]
        y1 = v[i, 0:-1]
        y2 = v[i, 1:]
        plt.plot(x1, y1, x2, y2)
    plt.show()


# def junction(each_joint_velocity,)

# J = jacobian()
# print(J)

q0 = np.array([0.1, -0.2, 0.3])
qf = np.array([0.2, 2, 0.6])

velocity_plot(ptp_trajectory(q0, qf))
