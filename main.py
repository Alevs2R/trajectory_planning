from sympy import *
from elementary_transormations import *

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


J = jacobian()
print(J)
