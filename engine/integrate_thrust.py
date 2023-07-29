import matplotlib.pyplot as plt
import numpy as np
from math import pi

from scipy import integrate
from numpy import linalg

mu = 398_600
Re = 6378
g0 = 9.807
deg = 2 * pi / 360

r0 = np.array([Re + 480, 0, 0])
v0 = np.array([0, 7.7102, 0])
t0 = 0
t_burn = 261.1127


m0 = 2000
T = 10  # kN
Isp = 300

y0 = np.concatenate((r0, v0, (m0,)))





def manuever_with_engines(t, y):
    r_vec, v_vec, mass = y[0:3], y[3:6], y[6]

    r, v = linalg.norm(r_vec), linalg.norm(v_vec)

    a_vec = - mu * r_vec / r ** 3 + T / mass * v_vec / v
    m_dot = - T * 1000 / g0 / Isp

    return np.concatenate((v_vec, a_vec, (m_dot,)))


def trajectory_no_engine(t, y):
    r_vec, v_vec, mass = y[0:3], y[3:6], y[6]

    r = linalg.norm(r_vec)
    a_vec = - mu * r_vec / r ** 3

    return np.concatenate((v_vec, a_vec, (mass,)))


solution = integrate.RK45(
    manuever_with_engines,
    t0, y0, t_burn, 1e-1
)

t_values, y_values = [], []

while (solution.status != 'finished'):
    solution.step()
    t_values.append(solution.t)
    y_values.append(solution.y)

print(y_values[-1])
solution = integrate.RK45(
    trajectory_no_engine,
    t_values[-1], y_values[-1],
    t_values[-1] + 5000, 1e-1,
)

while (solution.status != 'finished'):
    solution.step()
    t_values.append(solution.t)
    y_values.append(solution.y)

plt.plot([val[0] for val in y_values], [val[1]
         for val in y_values], 'r', label='manuever engines')
plt.xlim([-9000, 9000])
plt.ylim([-9000, 9000])


# plt.plot([val[0] for val in y_values], [np.sqrt((Re + 480) ** 2 - val[0] ** 2)
#          for val in y_values], 'b', label='nonmanuever')




plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
# data = zip(t_values, y_values)
