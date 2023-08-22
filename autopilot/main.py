import copy

import krpc
import numpy as np
from astropy import units as u, time
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from scipy import optimize

from autopilot.calculating import MGATrajectoryParser
from autopilot.planets import Kerbin
from engine.ksp_planet import Kerbol
from engine.mga import FlybyDomain
from engine.state_vector import OrbitStateVector


# https://github.com/Ren0k/Project-Atmospheric-Drag

class AutopilotMGA:
    """
    Gets and returns only normal right-handed vectors
    """

    def __init__(self, mga_trajectory: MGATrajectoryParser):
        self.conn = krpc.connect()
        self.vessel = self.conn.space_center.active_vessel
        self.departure_planet = self.vessel.orbit.body

        self.data = mga_trajectory.point.points_sequence[0]
        self.mga_data = mga_trajectory

        self.excess_velocity = mga_trajectory.excess_velocity
        print(np.linalg.norm(self.excess_velocity))
        self.excess_velocity_u = self.excess_velocity / np.linalg.norm(self.excess_velocity)

        kerbin_body = self.connection().space_center.bodies['Kerbin']
        self.inertial_frame_kerbin = self.get_inertial_frame(kerbin_body)

        self.lines = []  # so lines are not deleted while exist here

        start_vel_u = self.vector_right_left_converter(self.excess_velocity_u)
        line1 = self.conn.drawing.add_direction_from_com(tuple(start_vel_u), self.inertial_frame_kerbin)
        line1.color = (1.0, 0, 0)
        self.lines.append(line1)

        sun_body = self.conn.space_center.bodies['Sun']
        self.inertial_frame_sun = self.get_inertial_frame(sun_body)

    def get_inertial_frame(self, planet):
        rotation_frame = planet.reference_frame

        angular_velocity = planet.angular_velocity(planet.non_rotating_reference_frame)
        rot_difference = planet.rotation_angle

        inertial_frame = self.conn.space_center.ReferenceFrame.create_relative(
            rotation_frame,
            rotation=(0, 1 * np.sin(rot_difference / 2), 0, np.cos(rot_difference / 2)),  # tuple(rotation.as_quat())
            angular_velocity=tuple(-np.array(angular_velocity))
        )
        return inertial_frame

    @staticmethod
    def vector_right_left_converter(vector):
        vector = copy.deepcopy(vector)
        vector[2], vector[1] = vector[1], vector[2]
        return vector

    def get_vessel_state(self, relative_to_frame):
        result = [
            np.array(self.vessel.position(relative_to_frame)) / 1000,
            np.array(self.vessel.velocity(relative_to_frame)) / 1000
        ]

        return [
            self.vector_right_left_converter(result[0]),
            self.vector_right_left_converter(result[1])
        ]

    def get_vessel_h(self, relative_to_frame):
        r, v = self.get_vessel_state(relative_to_frame)
        return np.cross(r, v)

    def calc_departure_orbit_params(self):
        kerbin_body = self.connection().space_center.bodies['Kerbin']
        inertial_frame_kerbin = self.get_inertial_frame(kerbin_body)

        radius, velocity = self.get_vessel_state(inertial_frame_kerbin)
        h = np.cross(radius, velocity)
        h_u = h / np.linalg.norm(h)

        print("угол: ", np.degrees(np.arccos(np.clip(np.dot(self.excess_velocity_u, h_u), -1.0, 1.0))))

        r = radius << u.km
        v = velocity << u.km / u.s
        solution = Orbit.from_vectors(Kerbin, r, v)

        i = float((solution.inc << u.rad) / u.rad)
        om = float((solution.raan << u.rad) / u.rad)

        # TODO использовать функцию минимазации топлива с условием Ax=b
        def finding_omega(x):
            # TODO возможно надо прибавлять np.pi/1 взависимости от того, h[2] куда направлен
            return self.excess_velocity_u[0] * np.cos(np.pi / 2 - x[0]) * np.cos(x[1] - np.pi / 2) + \
                self.excess_velocity_u[1] * np.cos(np.pi / 2 - x[0]) * np.sin(x[1] - np.pi / 2) + \
                self.excess_velocity_u[2] * np.sin(np.pi / 2 - x[0])

        root = optimize.newton(finding_omega, np.array([i, om]), maxiter=500)
        print("нужно сделать inc и omega:", np.degrees(root[0]), np.degrees(root[1]))
        print("значение в такой точке:", finding_omega(root))
        return np.degrees(root[0]), np.degrees(root[1])

    def tetta_for_escapment_manuever(self):
        v_excess = np.linalg.norm(self.excess_velocity)
        r_p = 100 + 600
        mu = self.departure_planet.gravitational_parameter / 1000 ** 3

        h = self.get_vessel_h(self.inertial_frame_kerbin)
        h_norm = np.linalg.norm(h)
        h_u = h / h_norm

        propagator_real = self.current_orbit_propagator(Kerbin, self.inertial_frame_kerbin)

        # ----------
        mu = self.mga_data.mu
        r_soi = 84_159
        a = mu * r_soi / (v_excess ** 2 * r_soi - 2 * mu)
        e = r_p / a + 1
        h_norm_hyp = np.sqrt(mu * r_p * (1 + e))
        tetta_soi = np.arccos((h_norm_hyp ** 2 / (r_soi * mu) - 1) / e)

        gamma = np.arctan(e * np.sin(tetta_soi) / (1 + e * np.cos(tetta_soi)))
        turn_angel = (np.pi / 2 - gamma) + tetta_soi
        # --------------

        betta = np.arccos(1 / (1 + r_p * v_excess ** 2 / mu))
        print('degrees betta:', np.degrees(betta), np.degrees(tetta_soi), np.degrees(gamma), np.degrees(turn_angel))
        rot = FlybyDomain.generate_rotation_matrix(h_u, -turn_angel)  # TODO понять mкакой знак
        r_p_pointing = (rot @ self.excess_velocity_u)

        r_p_pointing1 = self.vector_right_left_converter(r_p_pointing)
        line1 = self.conn.drawing.add_direction_from_com(tuple(r_p_pointing1), self.inertial_frame_kerbin)
        line1.color = (0.6, 0, 1)
        self.lines.append(line1)

        propagator = self.current_orbit_propagator(Kerbin, self.inertial_frame_kerbin)

        def finding_tetta(deg, printing=False):
            r = np.array(propagator.propagate_to_anomaly(deg[0] << u.deg).r / u.km)
            r_u = r / np.linalg.norm(r)
            if printing:
                print(r_u, r_p_pointing)

            return np.linalg.norm(r_u - r_p_pointing)

        root = optimize.minimize(finding_tetta, np.array(180), bounds=[(0, 360)], ).x
        print("true anomaly for r_p departure:", root[0], finding_tetta(root, True))

        return root[0]

    def current_orbit_propagator(self, planet, reference_frame):
        radius, velocity = self.get_vessel_state(reference_frame)

        r = radius << u.km
        v = velocity << u.km / u.s

        return Orbit.from_vectors(planet, r, v, epoch=time.Time(self.conn.space_center.ut, format='unix'))

    def departure_maneuver_nodes(self, multinode=False, multinode_count=3):
        propagator = self.current_orbit_propagator(Kerbin, self.inertial_frame_kerbin)

        # ----------
        excess_velocity_norm = np.linalg.norm(self.excess_velocity)
        r_p = 100 + 600

        mu = self.mga_data.mu
        r_soi = 84_159
        a = mu * r_soi / (excess_velocity_norm ** 2 * r_soi - 2 * mu)
        e = r_p / a + 1
        h_norm_hyp = np.sqrt(mu * r_p * (1 + e))
        tetta_soi = np.arccos((h_norm_hyp ** 2 / (r_soi * mu) - 1) / e)

        velocity = np.sqrt(
            2 * self.mga_data.mu * (1 / r_p + 1 / (2 * a))
        )

        # TODO орбиту делать circulirized, потому что иначе угол tetta_soi может оказаться больше угла ассимптоты
        cur_vel = propagator.v.to(u.km / u.s)
        time_to_reach_soi = Orbit.from_vectors(
            Kerbin, propagator.r,
            (cur_vel.value / np.linalg.norm(cur_vel.value) * velocity) * u.km / u.s
        ).propagate_to_anomaly(tetta_soi << u.rad).t_p / u.s
        print('время полета по гиперболе до soi:', time_to_reach_soi)
        # --------------

        # TODO здесь проблема с тем, что нужно кажется создавать маршрут заранее с запасом в время time_to_reach_soi??
        at_beginning_of_time = propagator.propagate(
            (
                    self.data.initial_time + self.data.launch_time - time_to_reach_soi.value - self.conn.space_center.ut) << u.s
        )
        print('at_beg_of_time:', self.data.initial_time + self.data.launch_time - time_to_reach_soi.value - self.conn.space_center.ut)
        till_periapsis = at_beginning_of_time.period / u.s - at_beginning_of_time.t_p / u.s

        tetta = pilot.tetta_for_escapment_manuever()
        node_point = at_beginning_of_time.propagate_to_anomaly(tetta << u.deg)

        maneuver_time_ut = float(
            self.data.initial_time + self.data.launch_time - time_to_reach_soi.value + till_periapsis + node_point.t_p / u.s
        )

        excess_velocity_norm = np.linalg.norm(self.excess_velocity)
        r_p = 100 + 600

        v_esc = np.sqrt(2 * self.mga_data.mu / r_p)  # old
        v_p_hyp = np.sqrt(excess_velocity_norm ** 2 + 2 * self.mga_data.mu / r_p)

        # ----------
        mu = self.mga_data.mu
        r_soi = 84_159
        a = mu * r_soi / (excess_velocity_norm ** 2 * r_soi - 2 * mu)
        e = r_p / a + 1
        h_norm_hyp = np.sqrt(mu * r_p * (1 + e))
        tetta_soi = np.arccos((h_norm_hyp ** 2 / (r_soi * mu) - 1) / e)

        velocity = np.sqrt(
            2 * self.mga_data.mu * (1 / r_p + 1 / (2 * a))
        )

        cur_vel = propagator.v.to(u.km / u.s)

        # --------------

        print(velocity)
        v_p_hyp = velocity

        if not multinode:
            delta = v_p_hyp - np.linalg.norm(np.array(node_point.v / u.km * u.s))
            self.vessel.control.add_node(maneuver_time_ut, prograde=delta * 1000)
            return

        propagator = self.current_orbit_propagator(Kerbin, self.inertial_frame_kerbin)
        safety_offset_delta = 40 / 1000
        part_delta = ((v_esc - safety_offset_delta - np.linalg.norm(np.array(node_point.v / u.km * u.s)))
                      / multinode_count * 1000)

        # TODO периоды полета, должны просходить до ut
        # TODO! make it proparly use unit system form astropy
        # https://docs.astropy.org/en/stable/units/quantity.html#creating-quantity-instances
        period_i, prop_time = 0 * u.s, (maneuver_time_ut - self.conn.space_center.ut) << u.s
        for i in range(multinode_count):
            self.vessel.control.add_node(maneuver_time_ut + period_i.value, prograde=part_delta)

            state = propagator.propagate(prop_time)

            # TODO боже, что это. Достаточно: v / norm(v)
            h = self.get_vessel_h(self.inertial_frame_kerbin)
            prograde = np.cross(h, state.r / u.km)
            prograde_u = prograde / np.linalg.norm(prograde)

            propagator = Orbit.from_vectors(Kerbin, state.r,
                                            state.v.to(u.km / u.s) + (prograde_u * part_delta) * u.m / u.s)
            prop_time = propagator.period
            period_i += propagator.period
            propagator = propagator.propagate(propagator.period)

        final_delta = v_p_hyp * 1000 - part_delta * 3 - np.linalg.norm(np.array(node_point.v / u.km * u.s)) * 1000
        self.vessel.control.add_node(maneuver_time_ut + period_i.value, prograde=final_delta)

    def correction_departure_maneuver(self):
        print(self.conn.space_center.ut,  self.data.initial_time + self.data.launch_time)

        propagator_real = self.current_orbit_propagator(Kerbol, self.inertial_frame_sun)
        calculation_started_at_ut = self.conn.space_center.ut

        departure_planet_state = self.data.departure_planet.ephemeris_at_time(
            self.data.initial_time, self.data.launch_time
        )
        spacecraft_state_sun = OrbitStateVector(
            departure_planet_state.radius, self.excess_velocity + departure_planet_state.velocity,
        )

        propagator_ideal = Orbit.from_vectors(
            Kerbol,
            spacecraft_state_sun.radius * u.km, spacecraft_state_sun.velocity * u.km / u.s,
            epoch=time.Time(self.data.initial_time + self.data.launch_time, format='unix')
        ).propagate((calculation_started_at_ut - self.data.initial_time - self.data.launch_time) * u.s)

        print("идеальная", propagator_ideal.v.value, np.linalg.norm(propagator_ideal.v.value), "\n",
              propagator_ideal.r.value, np.linalg.norm(propagator_ideal.r.value))

        real_r, real_v = self.get_vessel_state(self.inertial_frame_sun)
        print("реальная", real_v, np.linalg.norm(real_v), "\n", real_r, np.linalg.norm(real_r))

        def cost_function(data: np.array) -> float:
            maneuver_start, maneuver_end = data
            maneuver_time_length = maneuver_end - maneuver_start
            if maneuver_time_length < 0:
                return 10_000

            start = propagator_real.propagate(maneuver_start * u.s)
            end = propagator_ideal.propagate(
                maneuver_end * u.s
            )
            sol = Maneuver.lambert(start, end)

            return np.linalg.norm(sol[0][1].to(u.km / u.s).value) + np.linalg.norm(sol[1][1].to(u.km / u.s).value)

        print('cost_calc')
        upper_time_start_limit = self.data.alpha / 2 * self.data.flight_period
        root = optimize.minimize(cost_function, np.array([3.5 * 60 * 60, upper_time_start_limit]), bounds=[
            (3 * 60, self.data.alpha / 2 * self.data.flight_period),
            (3 * 60, self.data.alpha / 2 * self.data.flight_period),
        ], options={'maxiter': 200}).x

        print("true cost for maneuver:", root, cost_function(root) * 1000)

        maneuver_time_start, maneuver_time_end = root
        state_start = propagator_real.propagate(maneuver_time_start * u.s)
        state_end = propagator_ideal.propagate(
            maneuver_time_end * u.s
        )
        man = Maneuver.lambert(state_start, state_end)

        # man[1][1].to(u.km / u.s).value

        delta_v_sun = man[0][1].to(u.km / u.s).value

        velocity_u = state_start.v.value / np.linalg.norm(state_start.v.value)
        h = OrbitStateVector(state_start.r.value, state_start.v.value,
                             mu=self.mga_data.mu_sun).angular_momentum
        h_u = h / np.linalg.norm(h)
        additional = np.cross(velocity_u, h_u)

        q = np.array([
            velocity_u,
            h_u,
            additional
        ])

        delta_v_velocity_frame = (q @ delta_v_sun) * 1000

        print(delta_v_velocity_frame, np.linalg.norm(delta_v_velocity_frame))

        self.vessel.control.add_node(
            calculation_started_at_ut + maneuver_time_start,
            radial=delta_v_velocity_frame[2], prograde=delta_v_velocity_frame[0], normal=delta_v_velocity_frame[1]
        )

        # second correction:
        correction_state = Orbit.from_vectors(Kerbol, state_start.r, state_start.v + man[0][1])
        mid_state = correction_state.propagate(
            (maneuver_time_end - maneuver_time_start) * u.s
        )

        delta_v_sun = man[1][1].to(u.km / u.s).value

        velocity_u = mid_state.v.value / np.linalg.norm(mid_state.v.value)
        h = OrbitStateVector(mid_state.r.value, mid_state.v.value,
                             mu=self.mga_data.mu_sun).angular_momentum
        h_u = h / np.linalg.norm(h)
        additional = np.cross(velocity_u, h_u)

        q = np.array([
            velocity_u,
            h_u,
            additional
        ])

        delta_v_velocity_frame = (q @ delta_v_sun) * 1000

        print(delta_v_velocity_frame, np.linalg.norm(delta_v_velocity_frame))

        self.vessel.control.add_node(
            calculation_started_at_ut + maneuver_time_end,
            radial=delta_v_velocity_frame[2], prograde=delta_v_velocity_frame[0], normal=delta_v_velocity_frame[1]
        )

        propagator_ideal = propagator_ideal.propagate(maneuver_time_end * u.s)
        print("идеальная", propagator_ideal.v.value, np.linalg.norm(propagator_ideal.v.value))
        print("", propagator_ideal.r.value, np.linalg.norm(propagator_ideal.r.value))

        propagator_ideal = mid_state.from_vectors(Kerbol, mid_state.r, mid_state.v + man[1][1])
        print("реальная после ламберта", propagator_ideal.v.value, np.linalg.norm(propagator_ideal.v.value))
        print("", propagator_ideal.r.value, np.linalg.norm(propagator_ideal.r.value))

    def mid_arc_manuever(self):
        mid_state, start_state, end_state = self.mga_data.departure_arc()
        delta_v_sun = start_state.velocity - mid_state.velocity

        velocity_u = mid_state.velocity / np.linalg.norm(mid_state.velocity)
        h = OrbitStateVector(mid_state.radius, mid_state.velocity,
                             mu=self.mga_data.mu_sun).angular_momentum
        h_u = h / np.linalg.norm(h)
        additional = np.cross(velocity_u, h_u)

        q = np.array([
            velocity_u,
            h_u,
            additional
        ])
        delta_v_velocity_frame = (q @ delta_v_sun) * 1000

        print(delta_v_velocity_frame, np.linalg.norm(delta_v_velocity_frame))

        self.vessel.control.add_node(
            self.data.launch_time + self.data.initial_time + self.data.alpha * self.data.flight_period,
            radial=delta_v_velocity_frame[2],
            prograde=delta_v_velocity_frame[0], normal=delta_v_velocity_frame[1])

        propagator_ideal = Orbit.from_vectors(
            Kerbol,
            start_state.radius * u.km, start_state.velocity * u.km / u.s,
        ).propagate(((1 - self.data.alpha) * self.data.flight_period) * u.s)

        print("идеальная", propagator_ideal.v.value, np.linalg.norm(propagator_ideal.v.value), "\n",
              propagator_ideal.r.value, np.linalg.norm(propagator_ideal.r.value))

        real_r, real_v = self.get_vessel_state(self.inertial_frame_sun)
        print("реальная", real_v, np.linalg.norm(real_v), "\n", real_r, np.linalg.norm(real_r))


    def connection(self):
        return self.conn


# ----------------------
mga_trajectory = MGATrajectoryParser("autopilot/ksp_to_duna.pickle")
pilot = AutopilotMGA(mga_trajectory)

# inclination, omega = pilot.calc_departure_orbit_params()
# print('ГРАДУСЫ. incl:', inclination, 'right ascending node:', omega)
# input("CONTINUE?..")

# pilot.departure_maneuver_nodes()
# pilot.departure_maneuver_nodes(multinode=True, multinode_count=3)
# input("CONTINUE?..")

# pilot.correction_departure_maneuver()
# input("CONTINUE?..")

pilot.mid_arc_manuever()
input("CONTINUE?..")

exit(0)
# ----------------------

#
# conn = krpc.connect()
#
# cur_vessel = conn.space_center.active_vessel
# current_planet = kerbin_body = cur_vessel.orbit.body
#
# kerbin_body = current_planet = conn.space_center.bodies['Kerbin']
# print('mean:', kerbin_body.orbit.mean_anomaly_at_ut(0))
# angular_velocity = current_planet.angular_velocity(current_planet.non_rotating_reference_frame)
# v = current_planet.rotation_angle
#
# rotation_frame = current_planet.reference_frame
#
# inertial_frame_kerbin = conn.space_center.ReferenceFrame.create_relative(
#     rotation_frame,
#     rotation=(0, 1 * np.sin(v / 2), 0, np.cos(v / 2)),  # tuple(rotation.as_quat())
#     angular_velocity=tuple(-np.array(angular_velocity))
# )
#
# velocity = np.array(cur_vessel.velocity(inertial_frame_kerbin)) / 1000
# radius = np.array(cur_vessel.position(inertial_frame_kerbin)) / 1000
#
# ksp_vec_to_normal(velocity)
# ksp_vec_to_normal(radius)
#
# start_vel_u = start_vel / np.linalg.norm(start_vel)
# h = np.cross(radius, velocity)
# h_u = h / np.linalg.norm(h)
#
# print("угол: ", np.degrees(np.arccos(np.clip(np.dot(start_vel_u, h_u), -1.0, 1.0))))
#
# r = radius << u.km
# v = velocity << u.km / u.s
# solution = Orbit.from_vectors(Kerbin, r, v)
#
# soi_radius = 84_159_286  # TODO current_planet.sphere_of_influence <- gives inf
#
# i = float((solution.inc << u.rad) / u.rad)
# om = float((solution.raan << u.rad) / u.rad)
#
#
# def finding_omega(x):
#     # TODO возможно надо прибавлять np.pi/1 взависимости от того, h[2] куда направлен
#     return start_vel_u[0] * np.cos(np.pi / 2 - x[0]) * np.cos(x[1] - np.pi / 2) + \
#         start_vel_u[1] * np.cos(np.pi / 2 - x[0]) * np.sin(x[1] - np.pi / 2) + \
#         start_vel_u[2] * np.sin(np.pi / 2 - x[0])
#
#
# root = optimize.newton(finding_omega, np.array([i, om]), maxiter=500)
# print("нужно сделать inc и omega:", np.degrees(root[0]), np.degrees(root[1]))
# print("значение в такой точке:", finding_omega(root))
#
# # ---------------
#
# v_excess = np.linalg.norm(start_vel)
# print('excess: ', v_excess)
# r_p = 100 + 600
# mu = current_planet.gravitational_parameter / 1000 ** 3
#
# betta = np.arccos(1 / (1 + r_p * v_excess ** 2 / mu))
# print(betta, h_u)
# rot = FlybyDomain.generate_rotation_matrix(h_u, betta)  # TODO понять mкакой знак
# r_p_pointing = -(rot @ start_vel_u)
#
# ksp_vec_to_normal(r_p_pointing)
# line1 = conn.drawing.add_direction_from_com(tuple(r_p_pointing), inertial_frame_kerbin)
# ksp_vec_to_normal(r_p_pointing)
#
#
# def finding_tetta(deg):
#     r = np.array(solution.propagate_to_anomaly(deg[0] << u.deg).r / u.km)
#     r_u = r / np.linalg.norm(r)
#     return np.linalg.norm(r_u - r_p_pointing)
#
#
# root = optimize.minimize(finding_tetta, np.array(180), bounds=[(0, 360)]).x
# print("true anomaly for r_p departure:", root, finding_tetta(root))
#
# at_beginning_of_time = solution.propagate((data.launch_time - conn.space_center.ut) << u.s)
# till_periapsis = at_beginning_of_time.period / u.s - at_beginning_of_time.t_p / u.s
# print(till_periapsis)
#
# node_point = at_beginning_of_time.propagate_to_anomaly(root[0] << u.deg)
# time_to_wait = float(data.launch_time - conn.space_center.ut + till_periapsis + node_point.t_p / u.s)  # TODO CHANGE
#
# print('wait', time_to_wait)
# v_p_hyp = np.sqrt(v_excess ** 2 + 2 * mu / r_p)
# delta_v = (v_p_hyp - np.linalg.norm(np.array(node_point.v / u.km * u.s))) * 1000
# cur_vessel.control.add_node(time_to_wait + conn.space_center.ut, prograde=delta_v)
#
# aiming_radius = r_p * np.sqrt(1 + 2 * mu / (r_p * v_excess ** 2))
#
# ksp_vec_to_normal(start_vel_u)
#
# line1 = conn.drawing.add_direction_from_com(tuple(start_vel_u), inertial_frame_kerbin)
# line1.color = (1.0, 0, 0)
#
# h = h / np.linalg.norm(h)
# ksp_vec_to_normal(h)
# line2 = conn.drawing.add_direction_from_com(tuple(h), inertial_frame_kerbin)
#


# -------------------------------------------------------------------------------------------------------


while True:
    pass

# print()
# print(solution.nu << u.deg)
