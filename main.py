from sympy import *
import numpy as np
from elementary_transormations import *
import matplotlib.pyplot as plt

from inverse_kinematics import inv_kin

init_printing(use_unicode=True)

max_freq = 100
max_joint_velocity = 1
max_cartesian_velocity = 1
max_joint_acceleration = 1
max_cartesian_acceleration = 1
junction = 5 / max_freq

q1, q2, q3, l1, l2, l3 = symbols('q1 q2 q3 l1 l2 l3')

robot_l1 = 1
robot_l2 = 4
robot_l3 = 3

L = np.array([robot_l1, robot_l2, robot_l3])

# for artuculated RRR robot
def jacobian():
    T0 = eye(4)
    T0_1 = rz(q1)
    T0_2 = T0_1 * tz(l1)
    T0_3 = T0_2 * ry(q2)
    T0_4 = T0_3 * tx(l2)
    T0_5 = T0_4 * ry(q3)
    T0_6 = T0_5 * tx(l3)

    x = simplify(T0_6[0, 3])
    y = simplify(T0_6[1, 3])
    z = simplify(T0_6[2, 3])

    J = Matrix([
        [diff(x, q1), diff(x, q2), diff(x, q3)],
        [diff(y, q1), diff(y, q2), diff(y, q3)],
        [diff(z, q1), diff(z, q2), diff(z, q3)],
    ])

    return J

# already calculated inverse jacobian
def jacobian_inverse(q1, q2, q3):
    inverse = np.array([
        [-np.sin(q1) / (robot_l2 * np.cos(q2) + robot_l3 * np.cos(q2 + q3)),
         np.cos(q1) / (robot_l2 * np.cos(q2) + robot_l3 * np.cos(q2 + q3)), 0],
        [np.cos(q1) * np.cos(q2 + q3) / (robot_l2 * np.sin(q3)), np.sin(q1) * np.cos(q2 + q3) / (robot_l2 * np.sin(q3)),
         -np.sin(q2 + q3) / (robot_l2 * np.sin(q3))],
        [-(robot_l2 * np.cos(q2) + robot_l3 * np.cos(q2 + q3)) * np.cos(q1) / (robot_l2 * robot_l3 * np.sin(q3)),
         -(robot_l2 * np.cos(q2) + robot_l3 * np.cos(q2 + q3)) * np.sin(q1) / (robot_l2 * robot_l3 * np.sin(q3)),
         (robot_l2 * np.sin(q2) + robot_l3 * np.sin(q2 + q3)) / (robot_l2 * robot_l3 * np.sin(q3))]
    ])
    return inverse


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

        time = np.arange(0, t_f+0.005, 0.01)  # add 0.005 to create array with correct size
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
    dist_z = xf[2] - x0[2]
    dist_y = xf[1] - x0[1]
    dist_x = xf[0] - x0[0]
    dist_xy = np.sqrt(dist_x**2 + dist_y**2)
    sin_a = dist_z / dist
    cos_a = dist_xy / dist
    cos_b = dist_x / dist_xy
    sin_b = dist_y / dist_xy

    q0 = inv_kin(x0, L)
    qf = inv_kin(xf, L)

    def get_velocity_components(module_v):
        return np.array([module_v * cos_a * cos_b, module_v * cos_a * sin_b, module_v * sin_a])

    t_ba = np.around(dist / max_cartesian_velocity, 2)
    t_b = np.around(max_cartesian_velocity / max_cartesian_acceleration, 2)
    if t_b < t_ba:
        t_a = t_ba - t_b
        t_f = t_ba + t_b
        time = np.arange(0, t_f + 0.005, 0.01)  # add 0.005 to create array with correct size
        v = np.zeros(shape=(3, time.shape[0]))  # 3 because there are 3 joints
        pos = np.zeros(shape=(3, time.shape[0]))
        for (i,), cur_time in np.ndenumerate(time):
            if cur_time < t_b:
                cur_v = max_cartesian_acceleration * cur_time
                v[:, i] = jacobian_inverse()get_velocity_components(cur_v)
            elif cur_time < t_a + t_b:
                v[:, i] = max_cartesian_velocity
            else:
                v[:, i] = max_cartesian_acceleration * (t_f - cur_time)
    else:
        t_b = np.around(np.sqrt(dist/max_joint_acceleration), 2)
        actual_cartesian_velocity = dist / t_b
        x_v =




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

J = jacobian()
print(J)

print(jacobian_inverse(1,2,3))


# J_real = J.subs([(l1, robot_l1), (l2, robot_l2), (l3, robot_l3)]);
#
# print(simplify(J_real**-1))
# print(J_real)
#

# q0 = np.array([0.1, -0.2, 0.3])
# qf = np.array([0.2, 2, 0.6])
#
# velocity_plot(ptp_trajectory(q0, qf))
