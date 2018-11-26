import numpy as np
from robot_configuration import *
import matplotlib.pyplot as plt


def get_v_cartesian(x0, xf):
    dist = np.linalg.norm(xf - x0)

    t_ba = np.around(dist / max_cartesian_velocity, 2)
    t_b = np.around(max_cartesian_velocity / max_cartesian_acceleration, 2)

    time = 0
    v_cartesian = 0

    if t_b < t_ba:
        t_a = t_ba - t_b
        t_f = t_ba + t_b
        time = np.arange(0, t_f + 0.005, 0.01)  # add 0.005 to create array with correct size
        v_cartesian = np.zeros(time.shape[0])  # 3 because there are 3 joints

        for (i,), cur_time in np.ndenumerate(time):
            if cur_time < t_b:
                v_cartesian[i] = max_cartesian_acceleration * cur_time
            elif cur_time < t_a + t_b:
                v_cartesian[i] = max_cartesian_velocity
            else:
                v_cartesian[i] = max_cartesian_acceleration * (t_f - cur_time)

    else:
        t_b = np.around(np.sqrt(dist/max_cartesian_acceleration), 2)
        t_f = 2 * t_b
        time = np.arange(0, t_f + 0.005, 0.01)  # add 0.005 to create array with correct size
        v_cartesian = np.zeros(time.shape[0])  # 3 because there are 3 joints
        for (i,), cur_time in np.ndenumerate(time):
            if cur_time < t_b:
                v_cartesian[i] = max_cartesian_acceleration * cur_time
            else:
                v_cartesian[i] = max_cartesian_acceleration * (t_f - cur_time)

    return v_cartesian
    # plt.figure(0)
    # plt.ylabel('velocity, m')
    # plt.xlabel('time, s')
    # plt.title('cartesian velocity')
    # x1 = time[0:-1]
    # x2 = time[1:]
    # y1 = v_cartesian[0:-1]
    # y2 = v_cartesian[1:]
    # plt.plot(x1, y1, x2, y2)
    # plt.show()