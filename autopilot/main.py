import krpc
import numpy as np
from astropy import units as u
from poliastro.twobody import Orbit
from scipy import optimize

from autopilot.calculating import start_vel, data, start_state, mid_state
from autopilot.planets import Kerbin
from engine.mga import FlybyDomain
from engine.state_vector import OrbitStateVector


def ksp_vec_to_normal(vector):
    vector[2], vector[1] = vector[1], vector[2]


conn = krpc.connect()

cur_vessel = conn.space_center.active_vessel
current_planet = cur_vessel.orbit.body

angular_velocity = current_planet.angular_velocity(current_planet.non_rotating_reference_frame)
v = current_planet.rotation_angle

rotation_frame = current_planet.reference_frame

inertial_frame_kerbin = conn.space_center.ReferenceFrame.create_relative(
    rotation_frame,
    rotation=(0, 1 * np.sin(v / 2), 0, np.cos(v / 2)),  # tuple(rotation.as_quat())
    angular_velocity=tuple(-np.array(angular_velocity))
)

velocity = np.array(cur_vessel.velocity(inertial_frame_kerbin)) / 1000
radius = np.array(cur_vessel.position(inertial_frame_kerbin)) / 1000

ksp_vec_to_normal(velocity)
ksp_vec_to_normal(radius)

start_vel_u = start_vel / np.linalg.norm(start_vel)
h = np.cross(radius, velocity)
h_u = h / np.linalg.norm(h)

print("угол: ", np.degrees(np.arccos(np.clip(np.dot(start_vel_u, h_u), -1.0, 1.0))))

r = radius << u.km
v = velocity << u.km / u.s
solution = Orbit.from_vectors(Kerbin, r, v)

soi_radius = current_planet.sphere_of_influence

i = float((solution.inc << u.rad) / u.rad)
om = float((solution.raan << u.rad) / u.rad)


def finding_omega(x):
    # TODO возможно надо прибавлять np.pi/1 взависимости от того, h[2] куда направлен
    return start_vel_u[0] * np.cos(np.pi / 2 - x[0]) * np.cos(x[1] - np.pi / 2) + \
        start_vel_u[1] * np.cos(np.pi / 2 - x[0]) * np.sin(x[1] - np.pi / 2) + \
        start_vel_u[2] * np.sin(np.pi / 2 - x[0])


root = optimize.newton(finding_omega, np.array([i, om]), maxiter=500)
print("нужно сделать inc и omega:", root[0], root[1])
print("значение в такой точке:", finding_omega(root))

v_excess = np.linalg.norm(start_vel)
r_p = 100 + 600
mu = current_planet.gravitational_parameter / 1000 ** 3

betta = np.arccos(1 / (1 + r_p * v_excess ** 2 / mu))
print(betta, h_u)
rot = FlybyDomain.generate_rotation_matrix(h_u, -betta)  # TODO понять mкакой знак
r_p_pointing = -(rot @ start_vel_u)

ksp_vec_to_normal(r_p_pointing)
line1 = conn.drawing.add_direction_from_com(tuple(r_p_pointing), inertial_frame_kerbin)
ksp_vec_to_normal(r_p_pointing)


def finding_tetta(deg):
    r = np.array(solution.propagate_to_anomaly(deg[0] << u.deg).r / u.km)
    r_u = r / np.linalg.norm(r)
    return np.linalg.norm(r_u - r_p_pointing)


root = optimize.minimize(finding_tetta, np.array(180), bounds=[(0, 360)]).x
print("true anomaly for r_p departure:", root, finding_tetta(root))

at_beginning_of_time = solution.propagate((data.launch_time - conn.space_center.ut) << u.s)
till_periapsis = at_beginning_of_time.period / u.s - at_beginning_of_time.t_p / u.s
print(till_periapsis)

node_point = solution.propagate_to_anomaly(root[0] << u.deg)
time_to_wait = float(data.launch_time - conn.space_center.ut + till_periapsis + node_point.t_p / u.s)  # TODO CHANGE

print('wait', time_to_wait)
v_p_hyp = np.sqrt(v_excess ** 2 + 2 * mu / r_p)
delta_v = (v_p_hyp - np.linalg.norm(np.array(node_point.v / u.km * u.s))) * 1000
cur_vessel.control.add_node(time_to_wait + conn.space_center.ut, prograde=delta_v)

aiming_radius = r_p * np.sqrt(1 + 2 * mu / (r_p * v_excess ** 2))

ksp_vec_to_normal(start_vel_u)

line1 = conn.drawing.add_direction_from_com(tuple(start_vel_u), inertial_frame_kerbin)
line1.color = (1.0, 0, 0)

h = h / np.linalg.norm(h)
ksp_vec_to_normal(h)
line2 = conn.drawing.add_direction_from_com(tuple(h), inertial_frame_kerbin)



current_planet = conn.space_center.bodies['Sun']

angular_velocity = current_planet.angular_velocity(current_planet.non_rotating_reference_frame)
v = current_planet.rotation_angle

rotation_frame = current_planet.reference_frame

inertial_frame_sun = conn.space_center.ReferenceFrame.create_relative(
    rotation_frame,
    rotation=(0, 1 * np.sin(v / 2), 0, np.cos(v / 2)),  # tuple(rotation.as_quat())
    angular_velocity=tuple(-np.array(angular_velocity))
)

delta_v_sun = start_state.velocity - mid_state.velocity

velocity_u = mid_state.velocity / np.linalg.norm(mid_state.velocity)
h = OrbitStateVector(mid_state.radius, mid_state.velocity, mu=current_planet.gravitational_parameter / 1000 ** 3).angular_momentum
h_u = h / np.linalg.norm(h)
additional = np.cross(velocity_u, h_u)

q = np.array([
    velocity_u,
    h_u,
    additional
])
delta_v_velocity_frame = (q @ delta_v_sun) * 1000
ksp_vec_to_normal(delta_v_velocity_frame)

print(delta_v_velocity_frame, np.linalg.norm(delta_v_velocity_frame))

cur_vessel.control.add_node(conn.space_center.ut + time_to_wait + data.alpha * data.flight_period, radial=delta_v_velocity_frame[1],
                            prograde=delta_v_velocity_frame[0], normal=delta_v_velocity_frame[2])
while True:
    pass

# print()
# print(solution.nu << u.deg)
