# with open('imp_test.npy', 'rb') as f:
# points = np.load(f)
import dill
import numpy as np
from matplotlib import pyplot as plt
from poliastro.bodies import Sun, Earth, Mars, Venus

from engine.lambert.custom import LambertProblem
from engine.mga import LastState
from engine.planets.solar import SolarPlanet
from engine.propagator.universal import UniversalPropagator
from engine.state_vector import StateVector
from engine.utils import points_json_decoder, generate_rotation_matrix
from examples.solar_system_mga import initial_time

raw_points = dill.load(open("data_simulations/solar_mga.pickle", "rb"))

# TODO make it specified in json data
planet_type = SolarPlanet
current_system_sun = planet_type(Sun)

points = points_json_decoder(raw_points, planet_type)
point = min(points, key=lambda p: p.total_delta_v)

data = point.points_sequence[0]

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(projection='3d')

start_vel = data.v_start * np.array([
    np.cos(data.incl) * np.cos(data.decl),
    np.sin(data.incl) * np.cos(data.decl),
    np.sin(data.decl)
])

departure_planet_state = data.departure_planet.ephemeris_at_time(
    initial_time, data.launch_time
)

spacecraft_state_sun = StateVector(
    departure_planet_state.radius,
    start_vel + departure_planet_state.velocity,
    current_system_sun
)

ax.scatter(
    departure_planet_state.radius[0],
    departure_planet_state.radius[1],
    departure_planet_state.radius[2],
    c='blue', marker='o', zorder=30, s=80
)

# Drawing sun trajectory
sun_eph = current_system_sun.ephemeris_at_time(initial_time, data.launch_time)
ax.scatter(
    sun_eph.radius[0], sun_eph.radius[1], sun_eph.radius[2],
    c='yellow', marker='o', zorder=30, s=200
)

departure_planet_state = data.departure_planet.ephemeris_at_time(initial_time, data.launch_time)
solver = UniversalPropagator(spacecraft_state_sun, current_system_sun)


time = np.linspace(0, data.alpha * data.flight_period, 500)
x, y, z = [], [], []
for t in time:
    r = solver.state_after(t).radius
    x.append(r[0])
    y.append(r[1])
    z.append(r[2])
ax.plot(x, y, z)

mid_state = solver.state_after(data.alpha * data.flight_period)
ax.scatter(mid_state.radius[0], mid_state.radius[1], mid_state.radius[2], c='black', marker='x', s=30)

departure_planet_state = data.arrival_planet.ephemeris_at_time(
    initial_time, data.launch_time + data.flight_period
)
problem = LambertProblem(
    mid_state.radius, departure_planet_state.radius,
    (1 - data.alpha) * data.flight_period, current_system_sun
)
start_state, end_state = problem.solution()

t = ax.text(mid_state.radius[0] + 25_000_000 / 4, mid_state.radius[1], mid_state.radius[2],
            f'Maneuver (delta-v: {np.linalg.norm(start_state.velocity - mid_state.velocity):.2f} km/s)', zdir=None)
t.set_bbox(dict(facecolor='lightcyan', alpha=0.9, edgecolor='grey'))

solver = UniversalPropagator(start_state, SolarPlanet(Sun))
time = np.linspace(0, (1 - data.alpha) * data.flight_period, 200)
for t in time:
    r = solver.state_after(t).radius
    x.append(r[0])
    y.append(r[1])
    z.append(r[2])
ax.plot(x, y, z, c='magenta')

ax.scatter(departure_planet_state.radius[0], departure_planet_state.radius[1], departure_planet_state.radius[2],
           c='orange', marker='o', zorder=30, s=80)

departure_planet_state = data.departure_planet.ephemeris_at_time(initial_time, data.launch_time)
print(departure_planet_state.velocity)

v = solver.state_after((1 - data.alpha) * data.flight_period).velocity

data = point.points_sequence[1]
state = point.points_sequence[1]

last_state: LastState = LastState(
    point.points_sequence[0].flight_period + point.points_sequence[0].launch_time,
    0, v, None
)

flyby_planet_state = state.departure_planet.ephemeris_at_time(
    state.initial_time, last_state.total_flight_time
)
spacecraft_flyby_excess_velocity = last_state.velocity - flyby_planet_state.velocity

u_p = spacecraft_flyby_excess_velocity / np.linalg.norm(spacecraft_flyby_excess_velocity)
rotation_matrix = generate_rotation_matrix(u_p, state.gamma)

n = np.cross(spacecraft_flyby_excess_velocity, flyby_planet_state.velocity)
n = n / np.linalg.norm(n)

n_trajectory_plane = rotation_matrix @ n
mu = state.departure_planet.mu

betta = np.arccos(1 / (1 + state.periapsis * np.linalg.norm(spacecraft_flyby_excess_velocity) ** 2 / mu))
turn_angle = 2 * betta

rotation_matrix = generate_rotation_matrix(n_trajectory_plane, turn_angle)
spacecraft_flyby_excess_velocity = rotation_matrix @ spacecraft_flyby_excess_velocity

spacecraft_departure_velocity = spacecraft_flyby_excess_velocity + flyby_planet_state.velocity
spacecraft_departure_state = StateVector(
    flyby_planet_state.radius,
    spacecraft_departure_velocity,
    current_system_sun,
)

solver = UniversalPropagator(spacecraft_departure_state, current_system_sun)
time = np.linspace(0, data.alpha * data.flight_period, 500)
x, y, z = [], [], []
for t in time:
    r = solver.state_after(t).radius
    x.append(r[0])
    y.append(r[1])
    z.append(r[2])
ax.plot(x, y, z, c='magenta')

mid_state = solver.state_after(state.alpha * state.flight_period)

arrival_planet_state = state.arrival_planet.ephemeris_at_time(
    state.initial_time, last_state.total_flight_time + state.flight_period
)


problem = LambertProblem(
    mid_state.radius, arrival_planet_state.radius,
    (1 - state.alpha) * state.flight_period,
    current_system_sun
)
start_state, end_state = problem.solution()

t = ax.text(mid_state.radius[0] + 25_000_000 / 4, mid_state.radius[1], mid_state.radius[2],
            f'Маневр (delta-v: {np.linalg.norm(start_state.velocity - mid_state.velocity):.2f} km/s)', zdir=None)
t.set_bbox(dict(facecolor='lightcyan', alpha=0.9, edgecolor='grey'))

ax.scatter(mid_state.radius[0], mid_state.radius[1], mid_state.radius[2], c='black', marker='x', s=30)

solver = UniversalPropagator(start_state, current_system_sun)
time = np.linspace(0, (1 - state.alpha) * state.flight_period, 500)
x, y, z = [], [], []
for t in time:
    r = solver.state_after(t).radius
    x.append(r[0])
    y.append(r[1])
    z.append(r[2])
ax.plot(x, y, z, c='magenta')

r_p = data.arrival_planet.ephemeris_at_time(
    initial_time, last_state.total_flight_time + data.flight_period
).radius


ax.scatter(
    r_p[0], r_p[1], r_p[2],
    c='green', marker='o', zorder=30, s=80
)


def planet_trajectory(ax, planet, period):
    time_steps = np.linspace(0, period * 24 * 60 * 60, 300)

    x_p, y_p, z_p = [], [], []
    for t_planet in time_steps:
        r_p = planet.ephemeris_at_time(
            initial_time, t_planet
        ).radius

        x_p.append(r_p[0])
        y_p.append(r_p[1])
        z_p.append(r_p[2])
    print(np.linalg.norm([x_p[0], y_p[0], z_p[0]]))
    ax.plot(x_p, y_p, z_p, c='grey', linestyle='dashed')


planet_trajectory(ax, SolarPlanet(Earth), 365)
planet_trajectory(ax, SolarPlanet(Mars), 687)
planet_trajectory(ax, SolarPlanet(Venus), 255)

# ax.set_xlim([-65353911, 65353911])
# ax.set_ylim([-65353911, 65353911])
# ax.set_zlim([-65353911 / 3, 65353911 / 3])

ax.set_box_aspect([1, 1, 1])
plt.show()
