import sys
from datetime import datetime

import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from engine.oblateness_time_solver import OblatenessTimeSolver
from engine.projection_vector import ProjectionVector
from engine.state_vector import OrbitStateVector

initial_state = OrbitStateVector([ 6.55156074e+03, 2.18584740e+03, -2612.9], [-2.42229184e+00, 9.07201796e+00, 0])

if (initial_state.eccentricity_module > 1.0):
    print(f"orbit must be elliptic. Current ecc.: {initial_state.eccentricity_module}")
    sys.exit(1)

solver = OblatenessTimeSolver(initial_state)
final_state = solver.state_after(96 * 60 * 60)

fig = plt.figure(figsize=(7, 8))
ax = fig.add_subplot(211, projection='3d')
ax.view_init(elev=17., azim=-145, roll=0)
pr_ax = fig.add_subplot(212)

t_min, t_max = 0, 96 * 1 * 3600
dt = 125
t = np.arange(t_min, t_max, dt)

data = np.zeros((3, int(t_max / dt) + 1))
projection_data = np.zeros((2, int(t_max / dt) + 1))
projection_limiters = [0]

pr = ProjectionVector(np.array([3212.6, -2250.5, 5568.6]))
pr.projection_after(2700)


def update(i):
    ax.cla()
    pr_ax.cla()

    img = plt.imread('C:\\Users\\Андрей\\Desktop\\projection3.png')
    pr_ax.imshow(img, interpolation='nearest', aspect='auto', extent=(0, 360, -70, 70))

    pr_ax.set_xlim([0, 360])
    pr_ax.set_ylim([-70, 70])

    r = solver.earth_radius
    u, v = np.mgrid[0: 2 * np.pi: 25j, 0: np.pi: 25j]
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)

    # Plot sphere
    ax.plot_surface(x, y, z, color='b', zorder=1)
    art3d.pathpatch_2d_to_3d(plt.Circle((0, 0), r), z=0, zdir='z')

    time_str = datetime.utcfromtimestamp(i * dt).strftime('day, %H:%M:%S')
    day = datetime.utcfromtimestamp(i * dt).strftime('%d')

    fig.suptitle(f"Time elapsed: {int(day) - 1} {time_str}")

    ax.set_xlim([-11000, 11000])
    ax.set_ylim([-11000, 11000])
    ax.set_zlim([-11000, 11000])

    ax.set_box_aspect([1, 1, 1])

    ticks = np.arange(-9000, 11000, 9000)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Animation')

    r = solver.state_after(dt * i).radius
    for j in range(3):
        data[j][i] = r[j]

    ax.plot(data[0][:i + 1], data[1][:i + 1], data[2][:i + 1], c='r', zorder=15)
    ax.scatter(r[0], r[1], r[2], c='black', marker='o', zorder=30)

    pr_ax.set_title('Earth Mercator Projection')

    plt.show()

    pr = ProjectionVector(r)
    pr.projection_after(dt * i)
    projection_data[0][i] = np.degrees(pr.asc)
    projection_data[1][i] = np.degrees(pr.decl)

    if (i > 2):
        if (projection_data[0][i] < projection_data[0][i - 1]):
            projection_limiters.append(i)
        for j in range(len(projection_limiters) - 1):
            pr_ax.plot(projection_data[0][projection_limiters[j]:projection_limiters[j + 1]], projection_data[1][projection_limiters[j]:projection_limiters[j + 1]], c='r', zorder=15)
        pr_ax.plot(projection_data[0][projection_limiters[len(projection_limiters) - 1]:i + 1], projection_data[1][projection_limiters[len(projection_limiters) - 1]:i + 1], c='r', zorder=15)
        
    pr_ax.scatter(projection_data[0][0],projection_data[1][0], c='red', marker='o', zorder=30)
    pr_ax.scatter(np.degrees(pr.asc), np.degrees(pr.decl), c='black', marker='o', zorder=30)


# Create the animation
ani = FuncAnimation(fig, update, frames=len(t), interval=5)

# Display the animation
plt.show()
