# with open('imp_test.npy', 'rb') as f:
# points = np.load(f)
import dill
import numpy as np

from engine.lambert.custom import LambertProblem
from engine.mga import starting_domain
from engine.mga.flyby_domain import FlybyDomain
from engine.mga.initial_domain import InitialDomain
from engine.mga.sequence import LastState
from engine.planets.ksp import Kerbol, KspPlanet
from engine.planets.solar import SolarPlanet
from engine.propagator.universal import UniversalPropagator
from engine.state_vector import StateVector
from engine.utils import points_json_decoder


class MGATrajectoryParser:
    def __init__(self, filename):
        self.mu = 3.5316000 * 10 ** 12 / 10 ** 9
        self.mu_sun = 1.1723328e18 / 10 ** 9  # kerbol
        self.Sun = Kerbol

        self.Earth = "Kerbin"

        raw_points = dill.load(open(filename, 'rb'))
        points = points_json_decoder(raw_points)

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

        departure_planet_state = self.departure_data.arrival_planet.ephemeris_at_time(
            # TODO change name because gives fake error
            starting_domain.initial_time, self.departure_data.launch_time + self.departure_data.flight_period
        )
        problem = LambertProblem(mid_state.radius, departure_planet_state.radius,
                                 (1 - self.departure_data.alpha) * self.departure_data.flight_period,
                                 self.mu_sun)

        start_state, end_state = problem.solution()

        return mid_state, start_state, end_state
