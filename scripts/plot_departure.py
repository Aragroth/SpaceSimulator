import numpy as np
from matplotlib import pyplot as plt
from poliastro.bodies import Earth, Mars, Sun

from engine.ellipsoid_trajectory import EllipsoidTrajectoryTimeSolver
from engine.lambert_problem import LambertProblem
from engine.mga import InitialDomain, mu_sun, starting_domain, ManeuversSequence, Planet
from engine.state_vector import OrbitStateVector

# with open('imp_test.npy', 'rb') as f:
# points = np.load(f)


points = ManeuversSequence.domain_solver(starting_domain)

data = min(points, key=lambda a: starting_domain.cost_function(a.to_numpy_array(), *a.to_meta()))

print(starting_domain.cost_function(data.to_numpy_array(), *data.to_meta()))
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

departure_planet_state = Planet(Earth).ephemeris_at_time(starting_domain.initial_time, data.launch_time)
print(departure_planet_state.velocity)


solver = EllipsoidTrajectoryTimeSolver(
    OrbitStateVector(departure_planet_state.radius, departure_planet_state.velocity, mu_sun),
    mu_sun
)

time = np.linspace(0, 366 * 24 * 60 * 60, 500)
x, y, z = [], [], []
for t in time:
    r = solver.state_after(t).radius
    x.append(r[0])
    y.append(r[1])
    z.append(r[2])

ax.plot(x, y, z, c='black', linestyle='dashed')
spacecraft_state_sun = OrbitStateVector(
    departure_planet_state.radius,
    start_vel + departure_planet_state.velocity,
    mu_sun
)

t = ax.text(departure_planet_state.radius[0] + 25_000_000, departure_planet_state.radius[1], departure_planet_state.radius[2],
            f'$V^{{\\nu}}_{{sun}}=${np.linalg.norm(spacecraft_state_sun.velocity):.2f} km/s', zdir=None)
t.set_bbox(dict(facecolor='lightcyan', alpha=0.9, edgecolor='grey'))


ax.scatter(departure_planet_state.radius[0], departure_planet_state.radius[1], departure_planet_state.radius[2],
           c='blue', marker='o', zorder=30, s=80)

sun_eph = Planet(Sun).ephemeris_at_time(starting_domain.initial_time, data.launch_time)
ax.scatter(sun_eph.radius[0], sun_eph.radius[1], sun_eph.radius[2], c='yellow', marker='o', zorder=30, s=200)

departure_planet_state = Planet(Earth).ephemeris_at_time(starting_domain.initial_time, data.launch_time)

solver = EllipsoidTrajectoryTimeSolver(spacecraft_state_sun, mu_sun)
time = np.linspace(0, data.alpha * data.flight_period, 500)
x, y, z = [], [], []
for t in time:
    r = solver.state_after(t).radius
    x.append(r[0])
    y.append(r[1])
    z.append(r[2])

mid_state = solver.state_after(data.alpha * data.flight_period)

ax.scatter(mid_state.radius[0], mid_state.radius[1], mid_state.radius[2], c='black', marker='x', s=30)

departure_planet_state = Planet(Mars).ephemeris_at_time(
    starting_domain.initial_time, data.launch_time + data.flight_period
)
problem = LambertProblem(mid_state.radius, departure_planet_state.radius, (1 - data.alpha) * data.flight_period,
                         mu_sun)
start_state, end_state = problem.solution()

t = ax.text(mid_state.radius[0] + 25_000_000, mid_state.radius[1], mid_state.radius[2],
            f'Маневр (delta-v: {np.linalg.norm(start_state.velocity - mid_state.velocity):.2f} km/s)', zdir=None)
t.set_bbox(dict(facecolor='lightcyan', alpha=0.9, edgecolor='grey'))

print(start_state.eccentricity_module)
solver = EllipsoidTrajectoryTimeSolver(start_state, mu_sun)
time = np.linspace(0, (1 - data.alpha) * data.flight_period, 200)
for t in time:
    r = solver.state_after(t).radius
    x.append(r[0])
    y.append(r[1])
    z.append(r[2])

ax.scatter(departure_planet_state.radius[0], departure_planet_state.radius[1], departure_planet_state.radius[2],
           c='red', marker='o', zorder=30, s=80)

ax.plot(x, y, z, c='magenta')

departure_planet_state = Planet(Mars).ephemeris_at_time(
    starting_domain.initial_time, data.launch_time + data.flight_period
)
solver = EllipsoidTrajectoryTimeSolver(
    OrbitStateVector(departure_planet_state.radius, departure_planet_state.velocity, mu_sun),
    mu_sun
)
time = np.linspace(0, 688 * 24 * 60 * 60, 500)
x, y, z = [], [], []
for t in time:
    r = solver.state_after(t).radius
    x.append(r[0])
    y.append(r[1])
    z.append(r[2])

ax.plot(x, y, z, c='black', linestyle='dashed')
ax.set_box_aspect([1, 1, 1])
plt.show()
