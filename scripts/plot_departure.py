# with open('imp_test.npy', 'rb') as f:
# points = np.load(f)
import dill
import numpy as np
from astropy.time import Time
from matplotlib import pyplot as plt
from poliastro.bodies import Earth, Mars, Sun, Venus
from scipy.constants import gravitational_constant
from astropy import units as u

from engine.ksp_planet import Kerbol, KspPlanet
from engine.lambert_problem import LambertProblem
from engine.mga import mu_sun, starting_domain, Planet, LastState, FlybyDomain, InitialDomain
from engine.state_vector import OrbitStateVector
from engine.universal_trajectory import UniversalTimeSolver




mu = 398_600
r0 = 600 + 100
mu_sun = 1.327 * (10 ** 11)

mu = 3.5316000 * 10 ** 12 / 10 ** 9
mu_sun = 1.1723328e18 / 10 ** 9  # kerbol
Sun = Kerbol

Earth = "Kerbin"

points = dill.load(open("scripts/ksp_to_duna.pickle", "rb"))

points = [
    LastState(i['total_flight_time'], i['total_delta_v'], i['velocity'],
              [
                  InitialDomain.DomainPoint(
                    x["initial_time"],
                    KspPlanet(x["departure_planet"]),
                    KspPlanet(x["arrival_planet"]),
                    x["v_start"],
                    x["launch_time"],
                    x["alpha"],
                    x["flight_period"],
                    x["incl"],
                    x["decl"],
                    ) if 'incl' in x else
                  FlybyDomain.DomainPoint(
                    x["initial_time"],
                    KspPlanet(x["departure_planet"]),
                    KspPlanet(x["arrival_planet"]),
                    x["gamma"],
                    x["periapsis"],
                    x["alpha"],
                    x["flight_period"],
                  )
                  for x in i['points_sequence']
              ]
              )
    for i in points
]



point = min(points, key=lambda x: x.total_delta_v)

print(point.points_sequence[0], point.total_delta_v)
data = point.points_sequence[0]
print(len(point.points_sequence))
print(
    data.initial_time,
    data.departure_planet,
    data.arrival_planet,
    data.v_start,
    data.launch_time / (24 * 60 * 60),
    data.alpha,
    data.flight_period / (24 * 60 * 60),
    data.incl,
    data.decl,
)

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(projection='3d')

start_vel = data.v_start * np.array([
    np.cos(data.incl) * np.cos(data.decl),
    np.sin(data.incl) * np.cos(data.decl),
    np.sin(data.decl)
])

departure_planet_state = data.departure_planet.ephemeris_at_time(starting_domain.initial_time, data.launch_time)
print(departure_planet_state.velocity)
print("norm of planet:", np.linalg.norm(departure_planet_state.velocity * 1000))


spacecraft_state_sun = OrbitStateVector(
    departure_planet_state.radius,
    start_vel + departure_planet_state.velocity,
    mu_sun
)
print(start_vel * 1000, departure_planet_state.velocity * 1000)
print("velocity around sun:", spacecraft_state_sun.velocity * 1000, np.linalg.norm(spacecraft_state_sun.velocity * 1000))

ax.scatter(departure_planet_state.radius[0], departure_planet_state.radius[1], departure_planet_state.radius[2],
           c='blue', marker='o', zorder=30, s=80)

# sun_eph = Planet(Kerbol).ephemeris_at_time(starting_domain.initial_time, data.launch_time)
# ax.scatter(sun_eph.radius[0], sun_eph.radius[1], sun_eph.radius[2], c='yellow', marker='o', zorder=30, s=200)
ax.scatter(0, 0, 0, c='yellow', marker='o', zorder=30, s=200)

departure_planet_state = KspPlanet(Earth).ephemeris_at_time(starting_domain.initial_time, data.launch_time)

solver = UniversalTimeSolver(spacecraft_state_sun, Planet(Sun))

# fulll traj drawing
# time = np.linspace(0, 4 * data.flight_period, 500)
# x, y, z = [], [], []
# for t in time:
#     r = solver.state_after(t).radius
#     x.append(r[0])
#     y.append(r[1])
#     z.append(r[2])
# ax.plot(x, y, z)

time = np.linspace(0, data.alpha * data.flight_period, 500)
x, y, z = [], [], []
for t in time:
    r = solver.state_after(t).radius
    x.append(r[0])
    y.append(r[1])
    z.append(r[2])

mid_state = solver.state_after(data.alpha * data.flight_period)

ax.scatter(mid_state.radius[0], mid_state.radius[1], mid_state.radius[2], c='black', marker='x', s=30)

departure_planet_state = data.arrival_planet.ephemeris_at_time(
    starting_domain.initial_time, data.launch_time + data.flight_period
)
problem = LambertProblem(mid_state.radius, departure_planet_state.radius, (1 - data.alpha) * data.flight_period,
                         mu_sun)
start_state, end_state = problem.solution()

t = ax.text(mid_state.radius[0] + 25_000_000 / 4, mid_state.radius[1], mid_state.radius[2],
            f'Маневр (delta-v: {np.linalg.norm(start_state.velocity - mid_state.velocity):.2f} km/s)', zdir=None)
print("----- norm: ", np.linalg.norm(start_state.velocity - mid_state.velocity))
t.set_bbox(dict(facecolor='lightcyan', alpha=0.9, edgecolor='grey'))

print(start_state.eccentricity_module)
solver = UniversalTimeSolver(start_state, Planet(Sun))
time = np.linspace(0, (1 - data.alpha) * data.flight_period, 200)
for t in time:
    r = solver.state_after(t).radius
    x.append(r[0])
    y.append(r[1])
    z.append(r[2])

ax.scatter(departure_planet_state.radius[0], departure_planet_state.radius[1], departure_planet_state.radius[2],
           c='orange', marker='o', zorder=30, s=80)

ax.plot(x, y, z, c='magenta')

departure_planet_state = data.departure_planet.ephemeris_at_time(starting_domain.initial_time, data.launch_time)
print(departure_planet_state.velocity)


v = solver.state_after((1 - data.alpha) * data.flight_period).velocity

#
# data = point.points_sequence[1]
# state = point.points_sequence[1]
#
#
# last_state: LastState = LastState(point.points_sequence[0].flight_period + point.points_sequence[0].launch_time, 0, v, None)
#
# flyby_planet_state = state.departure_planet.ephemeris_at_time(
#     state.initial_time, last_state.total_flight_time
# )
# spacecraft_flyby_excess_velocity = last_state.velocity - flyby_planet_state.velocity
#
# u_p = spacecraft_flyby_excess_velocity / np.linalg.norm(spacecraft_flyby_excess_velocity)
# rotation_matrix = FlybyDomain.generate_rotation_matrix(u_p, state.gamma)
#
# n = np.cross(spacecraft_flyby_excess_velocity, flyby_planet_state.velocity)
# n = n / np.linalg.norm(n)
#
# n_trajectory_plane = rotation_matrix @ n
# mu = gravitational_constant * float(state.departure_planet.mass / u.kg) / 1000 ** 3
#
# betta = np.arccos(1 / (1 + state.periapsis * np.linalg.norm(spacecraft_flyby_excess_velocity) ** 2 / mu))
#
# leading_flyby = True
# if np.dot(spacecraft_flyby_excess_velocity, flyby_planet_state.velocity) < 0:
#     leading_flyby = False
#
# turn_angle = 2 * betta
# if leading_flyby:
#     turn_angle *= -1
#
# rotation_matrix = FlybyDomain.generate_rotation_matrix(n_trajectory_plane, turn_angle)
# spacecraft_flyby_excess_velocity = rotation_matrix @ spacecraft_flyby_excess_velocity
#
# spacecraft_departure_velocity = spacecraft_flyby_excess_velocity + flyby_planet_state.velocity
# spacecraft_departure_state = OrbitStateVector(
#     flyby_planet_state.radius,
#     spacecraft_departure_velocity
# )
#
# solver = UniversalTimeSolver(spacecraft_departure_state, Planet(Sun))
# time = np.linspace(0, data.alpha * data.flight_period, 500)
# x, y, z = [], [], []
# for t in time:
#     r = solver.state_after(t).radius
#     x.append(r[0])
#     y.append(r[1])
#     z.append(r[2])
# ax.plot(x, y, z, c='magenta')
#
# mid_state = solver.state_after(state.alpha * state.flight_period)
#
# arrival_planet_state = state.arrival_planet.ephemeris_at_time(
#     state.initial_time, last_state.total_flight_time + state.flight_period
# )
#
#
# problem = LambertProblem(
#     mid_state.radius, arrival_planet_state.radius,
#     (1 - state.alpha) * state.flight_period,
#     mu_sun
# )
# start_state, end_state = problem.solution()
#
# t = ax.text(mid_state.radius[0] + 25_000_000 / 4, mid_state.radius[1], mid_state.radius[2],
#             f'Маневр (delta-v: {np.linalg.norm(start_state.velocity - mid_state.velocity):.2f} km/s)', zdir=None)
# t.set_bbox(dict(facecolor='lightcyan', alpha=0.9, edgecolor='grey'))
#
# ax.scatter(mid_state.radius[0], mid_state.radius[1], mid_state.radius[2], c='black', marker='x', s=30)
#
# solver = UniversalTimeSolver(start_state, Planet(Sun))
# time = np.linspace(0, (1 - state.alpha) * state.flight_period, 500)
# x, y, z = [], [], []
# for t in time:
#     r = solver.state_after(t).radius
#     x.append(r[0])
#     y.append(r[1])
#     z.append(r[2])
# ax.plot(x, y, z, c='magenta')
#
# r_p = data.arrival_planet.ephemeris_at_time(
#     starting_domain.initial_time, last_state.total_flight_time + data.flight_period
# ).radius
#
#
# ax.scatter(r_p[0], r_p[1], r_p[2],
#            c='green', marker='o', zorder=30, s=80)




def planet_trajectory(ax, planet, period):
    time_steps = np.linspace(0, period * 6 * 60 * 60, 300)

    x_p, y_p, z_p = [], [], []
    for t_planet in time_steps:
        r_p = planet.ephemeris_at_time(
            0, t_planet
        ).radius

        x_p.append(r_p[0])
        y_p.append(r_p[1])
        z_p.append(r_p[2])
    print(np.linalg.norm([x_p[0], y_p[0], z_p[0]]))
    ax.plot(x_p, y_p, z_p, c='grey', linestyle='dashed')


planet_trajectory(ax, KspPlanet("Kerbin"), 427)
planet_trajectory(ax, KspPlanet("Duna"), 802)
planet_trajectory(ax, KspPlanet("Jool"), 4_846)
planet_trajectory(ax, KspPlanet("Eve"), 262)


ax.set_xlim([-65353911, 65353911])
ax.set_ylim([-65353911, 65353911])
ax.set_zlim([-65353911 / 3, 65353911 / 3])

ax.set_box_aspect([1, 1, 1])
plt.show()
