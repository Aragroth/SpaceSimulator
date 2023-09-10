# with open('imp_test.npy', 'rb') as f:
# points = np.load(f)
import dill
import numpy as np

from engine.planets.ksp import Kerbol, KspPlanet
from engine.lambert.custom import LambertProblem
from engine.mga import starting_domain
from engine.mga.sequence import LastState
from engine.mga.flyby_domain import FlybyDomain
from engine.mga.initial_domain import InitialDomain
from engine.planets.solar import SolarPlanet
from engine.state_vector import StateVector
from engine.propagator.universal import UniversalPropagator


class MGATrajectoryParser:
    def __init__(self, filename):
        self.mu = 3.5316000 * 10 ** 12 / 10 ** 9
        self.mu_sun = 1.1723328e18 / 10 ** 9  # kerbol
        self.Sun = Kerbol

        self.Earth = "Kerbin"


        raw_points = dill.load(open(filename, 'rb'))
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
            for i in raw_points
        ]

        self.point = min(points, key=lambda x: x.total_delta_v)

        print(self.point.points_sequence[0], self.point.total_delta_v)
        print(len(self.point.points_sequence))

        self.departure_data = self.point.points_sequence[0]
        self.excess_velocity = self.departure_data.v_start * np.array([
            np.cos(self.departure_data.incl) * np.cos(self.departure_data.decl),
            np.sin(self.departure_data.incl) * np.cos(self.departure_data.decl),
            np.sin(self.departure_data.decl)
        ])

    def departure_arc(self):
        departure_planet_state = self.departure_data.departure_planet.ephemeris_at_time(
            starting_domain.initial_time,
            self.departure_data.launch_time
        )

        print(departure_planet_state.velocity)
        spacecraft_state_sun = StateVector(
            departure_planet_state.radius,
            self.excess_velocity + departure_planet_state.velocity,
            self.mu_sun
        )

        solver = UniversalPropagator(spacecraft_state_sun, SolarPlanet(self.Sun))

        mid_state = solver.state_after(self.departure_data.alpha * self.departure_data.flight_period)

        departure_planet_state = self.departure_data.arrival_planet.ephemeris_at_time(  # TODO change name because gives fake error
            starting_domain.initial_time, self.departure_data.launch_time + self.departure_data.flight_period
        )
        problem = LambertProblem(mid_state.radius, departure_planet_state.radius, (1 - self.departure_data.alpha) * self.departure_data.flight_period,
                                 self.mu_sun)

        start_state, end_state = problem.solution()

        return mid_state, start_state, end_state




# sun_eph = Planet(Kerbol).ephemeris_at_time(starting_domain.initial_time, data.launch_time)
# ax.scatter(sun_eph.radius[0], sun_eph.radius[1], sun_eph.radius[2], c='yellow', marker='o', zorder=30, s=200)



# solver = UniversalTimeSolver(start_state, Planet(Sun))
#
# departure_planet_state = data.departure_planet.ephemeris_at_time(starting_domain.initial_time, data.launch_time)
# print(departure_planet_state.velocity)
#
# v = solver.state_after((1 - data.alpha) * data.flight_period).velocity

# data = point.points_sequence[1]
# state = point.points_sequence[1]
#
# last_state: LastState = LastState(point.points_sequence[0].flight_period + point.points_sequence[0].launch_time, 0, v,
#                                   None)
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
# mid_state = solver.state_after(state.alpha * state.flight_period)
#
# arrival_planet_state = state.arrival_planet.ephemeris_at_time(
#     state.initial_time, last_state.total_flight_time + state.flight_period
# )
#
# problem = LambertProblem(
#     mid_state.radius, arrival_planet_state.radius,
#     (1 - state.alpha) * state.flight_period,
#     mu_sun
# )
# start_state, end_state = problem.solution()
#
# solver = UniversalTimeSolver(start_state, Planet(Sun))
# r_p = data.arrival_planet.ephemeris_at_time(
#     starting_domain.initial_time, last_state.total_flight_time + data.flight_period
# ).radius
#
