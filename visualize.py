import matplotlib.pyplot as plt
import numpy as np

from kinematics import forw_kin
from robot_configuration import L


def motion_plot(pos, v, acc):
        time = np.arange(0, v.shape[1] * 0.01 - 0.005, 0.01)

        plt.figure(1)
        plt.ylabel('position, rad')
        plt.xlabel('time, ms')
        plt.title('joint positions')

        for i in range(0, v.shape[0]):
            x1 = time[0:-1]
            x2 = time[1:]
            y1 = pos[i, 0:-1]
            y2 = pos[i, 1:]
            plt.plot(x1, y1, x2, y2)
        plt.show()

        plt.figure(2)
        plt.ylabel('velocity, rad')
        plt.xlabel('time, ms')
        plt.title('joint velocities')

        for i in range(0, v.shape[0]):
            x1 = time[0:-1]
            x2 = time[1:]
            y1 = v[i, 0:-1]
            y2 = v[i, 1:]
            plt.plot(x1, y1, x2, y2)
        plt.show()

        plt.figure(3)
        plt.ylabel('position, rad')
        plt.xlabel('time, ms')
        plt.title('joint accelerations')

        cartesian_pos = np.zeros(shape=(3, pos.shape[1]))

        for i in range(0, v.shape[0]):
            x1 = time[0:-1]
            x2 = time[1:]
            y1 = acc[i, 0:-1]
            y2 = acc[i, 1:]
            plt.plot(x1, y1, x2, y2)
        plt.show()

        for i in range(0, pos.shape[1]):
            cartesian_pos[:, i] = forw_kin(pos[:, i].flatten(), L)
            print(cartesian_pos[:, i])

        # plt.figure(4)
        # ax = plt.axes(projection='3d')
        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Z axis')
        # ax.plot3D(cartesian_pos[0, :].flatten(), cartesian_pos[1, :].flatten(), cartesian_pos[2:, ].flatten(), 'gray')
        # plt.show()

        plt.figure(4)
        plt.ylabel('position, m')
        plt.xlabel('time, s')
        plt.title('X cartesian axis')

        x1 = time[0:-1]
        x2 = time[1:]
        y1 = cartesian_pos[0, 0:-1]
        y2 = cartesian_pos[0, 1:]
        plt.plot(x1, y1, x2, y2)
        plt.ylim([0, 7])

        plt.show()


        plt.figure(5)
        plt.ylabel('position, m')
        plt.xlabel('time, s')
        plt.title('Y cartesian axis')

        x1 = time[0:-1]
        x2 = time[1:]
        y1 = cartesian_pos[1, 0:-1]
        y2 = cartesian_pos[1, 1:]
        plt.plot(x1, y1, x2, y2)
        plt.ylim([0, 7])
        plt.show()


        plt.figure(6)
        plt.ylabel('position, m')
        plt.xlabel('time, s')
        plt.title('Z cartesian axis')

        x1 = time[0:-1]
        x2 = time[1:]
        y1 = cartesian_pos[2, 0:-1]
        y2 = cartesian_pos[2, 1:]
        plt.plot(x1, y1, x2, y2)
        plt.ylim([0, 2])
        plt.show()