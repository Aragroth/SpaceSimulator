from __future__ import annotations

from copy import deepcopy
from typing import List, Tuple, Any

import numpy as np
from astropy import units as u
from astropy.time import Time
from numpy import random, ndarray
from poliastro.bodies import Mars, Earth, Venus, Sun, Body
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from scipy import optimize

from engine.ksp_planet import KspPlanet, Kerbol
from engine.lambert_problem import LambertProblem
from engine.state_vector import OrbitStateVector
from scipy.constants import gravitational_constant

from engine.universal_trajectory import UniversalTimeSolver

mu = 398_600
r0 = 600 + 100
mu_sun = 1.327 * (10 ** 11)

mu = 3.5316000 * 10 ** 12 / 10 ** 9
mu_sun = 1.1723328e18 / 10 ** 9  # kerbol
Sun = Kerbol

from engine.ellipsoid_trajectory import EllipsoidTrajectoryTimeSolver


class Constraint:
    def __init__(self, minimum, maximum):
        self.min = minimum
        self.max = maximum


class Planet:
    def __init__(self, planet):
        self.mass = planet.mass
        self.planet = planet

    def ephemeris_at_time(self, initial_time, delta_time, astropy_planet=True) -> OrbitStateVector:
        epoch = Time(initial_time.jd + delta_time / (60 * 60 * 24), format='jd')
        planet_ephem = Ephem.from_body(self.planet, epoch.tdb)

        r, v = np.array([0, 0, 0]), np.array([0, 0, 0])
        if astropy_planet:
            position = planet_ephem.rv()[0] << u.km
            speed = planet_ephem.rv()[1] << u.km / u.s
            r = np.array(position[0])
            v = np.array(speed[0])

        return OrbitStateVector(r, v)


class InitialDomain:
    def __init__(
            self, initial_time, departure_planet: Planet, arrival_planet: Planet,
            v_start_c: Constraint, launch_time_c: Constraint, alpha_c: Constraint, flight_period_c: Constraint,
            inclination_c: Constraint, declination_c: Constraint
    ):
        self.initial_time = initial_time
        self.departure_planet = departure_planet
        self.arrival_planet = arrival_planet

        self.v_start_constraint = v_start_c
        self.launch_time_constraint = launch_time_c
        self.alpha_constraint = alpha_c
        self.flight_period_constraint = flight_period_c
        self.incl_constraint = inclination_c
        self.decl_constraint = declination_c

    class DomainPoint:
        def __init__(
                self, initial_time, departure_planet: Planet, arrival_planet: Planet,
                v_start, launch_time, alpha, flight_period, incl, decl
        ):
            self.initial_time = initial_time
            self.departure_planet = departure_planet
            self.arrival_planet = arrival_planet

            self.v_start = v_start
            self.launch_time = launch_time
            self.alpha = alpha
            self.flight_period = flight_period
            self.incl = incl
            self.decl = decl

        @staticmethod
        def create_point_with_meta(meta: tuple, domain: np.array):
            return InitialDomain.DomainPoint(
                meta[0], meta[1], meta[2],
                domain[0], domain[1], domain[2], domain[3], domain[4], domain[5]
            )

        def to_numpy_array(self):
            return np.array([
                self.v_start, self.launch_time, self.alpha, self.flight_period, self.incl, self.decl
            ])

        def to_meta(self):
            return self.initial_time, self.departure_planet, self.arrival_planet

        def delta_v(self) -> float:
            return InitialDomain.cost_function(self.to_numpy_array(), *self.to_meta())

    def create_point(self, domain: np.array):
        return InitialDomain.DomainPoint(
            self.initial_time, self.departure_planet, self.arrival_planet,
            domain[0], domain[1], domain[2], domain[3], domain[4], domain[5]
        )

    def meta(self):
        return self.initial_time, self.departure_planet, self.arrival_planet

    def generate_point(self) -> InitialDomain.DomainPoint:
        v_start = random.default_rng().uniform(
            low=self.v_start_constraint.min, high=self.v_start_constraint.max
        )
        launch_time = int(random.default_rng().uniform(
            low=self.launch_time_constraint.min, high=self.launch_time_constraint.max
        ))
        alpha = random.default_rng().uniform(
            low=self.alpha_constraint.min, high=self.alpha_constraint.max
        )
        flight_period = random.default_rng().uniform(
            low=self.flight_period_constraint.min,
            high=self.flight_period_constraint.max
        )

        incl_normalized = random.default_rng().uniform(
            low=self.incl_constraint.min, high=self.incl_constraint.max
        )
        decl_normalized = random.default_rng().uniform(
            low=self.decl_constraint.min, high=self.decl_constraint.max
        )
        incl = 2 * np.pi * incl_normalized
        decl = np.arccos(2 * decl_normalized - 1) - np.pi / 2

        return InitialDomain.DomainPoint(
            self.initial_time, self.departure_planet, self.arrival_planet,
            v_start, launch_time, alpha, flight_period, incl, decl
        )

    def generate_bounds(self):
        return [
            (self.v_start_constraint.min, self.v_start_constraint.max),
            (self.launch_time_constraint.min, self.launch_time_constraint.max),
            (self.alpha_constraint.min, self.alpha_constraint.max),
            (self.flight_period_constraint.min, self.flight_period_constraint.max),
            (self.incl_constraint.min, self.incl_constraint.max),
            (self.decl_constraint.min, self.decl_constraint.max),
        ]

    @staticmethod
    def cost_function(domain: np.array, *args, **kwargs) -> float | tuple:
        """
        domain_array все параметры по которым считается, args - начальное время и планеты
        """
        state = InitialDomain.DomainPoint.create_point_with_meta(args, domain)

        excess_velocity = state.v_start * np.array([
            np.cos(state.incl) * np.cos(state.decl),
            np.sin(state.incl) * np.cos(state.decl),
            np.sin(state.decl)
        ])

        v_p_hyp = np.sqrt(np.linalg.norm(excess_velocity) ** 2 + 2 * mu / (100 + 600))

        departure_planet_state = state.departure_planet.ephemeris_at_time(state.initial_time, state.launch_time)
        spacecraft_state_sun = OrbitStateVector(
            departure_planet_state.radius,
            excess_velocity + departure_planet_state.velocity,
            mu_sun
        )

        solver = UniversalTimeSolver(spacecraft_state_sun, Planet(Sun))  # fix to universal mu or Planet class
        mid_state = solver.state_after(state.alpha * state.flight_period)

        arrival_planet_state = state.arrival_planet.ephemeris_at_time(
            state.initial_time, state.launch_time + state.flight_period
        )
        try:
            problem = LambertProblem(
                mid_state.radius, arrival_planet_state.radius,
                (1 - state.alpha) * state.flight_period,
                mu_sun
            )
            start_state, end_state = problem.solution()
        except RuntimeError:
            # some value if it fails, 1_000 delta-v is unreachable (nan + overflow errors)
            return 1_000

        delta_v = (v_p_hyp - np.sqrt(mu / (100 + 600))) + np.linalg.norm(start_state.velocity - mid_state.velocity)

        if 'last_state_generation' in kwargs and kwargs['last_state_generation']:
            return end_state.velocity, delta_v
        return delta_v


class FlybyDomain:

    def __init__(
            self, initial_time, departure_planet: Planet, arrival_planet: Planet,
            # total time flight and speed current
            gamma_c: Constraint, periapsis_c: Constraint, alpha_c: Constraint, flight_period_c: Constraint,
    ):
        self.initial_time = initial_time
        self.departure_planet = departure_planet
        self.arrival_planet = arrival_planet

        self.gamma_constraint = gamma_c
        self.periapsis_constraint = periapsis_c
        self.alpha_constraint = alpha_c
        self.flight_period_constraint = flight_period_c

    class DomainPoint:
        def __init__(
                self, initial_time, departure_planet: Planet, arrival_planet: Planet,
                gamma, periapsis, alpha, flight_period
        ):
            self.initial_time = initial_time
            self.departure_planet = departure_planet
            self.arrival_planet = arrival_planet

            self.gamma = gamma
            self.periapsis = periapsis
            self.alpha = alpha
            self.flight_period = flight_period

        @staticmethod
        def create_point_with_meta(meta: tuple, domain: np.array):
            return FlybyDomain.DomainPoint(
                meta[0], meta[1], meta[2],
                domain[0], domain[1], domain[2], domain[3],
            )

        def to_numpy_array(self):
            return np.array([
                self.gamma, self.periapsis, self.alpha, self.flight_period
            ])

        def to_meta(self):
            return self.initial_time, self.departure_planet, self.arrival_planet

        def delta_v(self) -> float:
            return FlybyDomain.cost_function(self.to_numpy_array(), *self.to_meta())

    def create_point(self, domain: np.array):
        return FlybyDomain.DomainPoint(
            self.initial_time, self.departure_planet, self.arrival_planet,
            domain[0], domain[1], domain[2], domain[3]
        )

    def meta(self):
        return self.initial_time, self.departure_planet, self.arrival_planet

    def generate_point(self) -> InitialDomain.DomainPoint:
        gamma_normalized = random.default_rng().uniform(
            low=self.gamma_constraint.min, high=self.gamma_constraint.max
        )
        gamma = 2 * np.pi * gamma_normalized

        periapsis = int(random.default_rng().uniform(
            low=self.periapsis_constraint.min, high=self.periapsis_constraint.max
        ))
        alpha = random.default_rng().uniform(
            low=self.alpha_constraint.min, high=self.alpha_constraint.max
        )
        flight_period = random.default_rng().uniform(
            low=self.flight_period_constraint.min,
            high=self.flight_period_constraint.max
        )

        return FlybyDomain.DomainPoint(
            self.initial_time, self.departure_planet, self.arrival_planet,
            gamma, periapsis, alpha, flight_period
        )

    def generate_bounds(self):
        return [
            (self.gamma_constraint.min, self.gamma_constraint.max),
            (self.periapsis_constraint.min, self.periapsis_constraint.max),
            (self.alpha_constraint.min, self.alpha_constraint.max),
            (self.flight_period_constraint.min, self.flight_period_constraint.max),
        ]

    @staticmethod
    def cost_function(domain: np.array, *args, **kwargs) -> int | tuple[Any, float | Any] | float | Any:
        """
        domain_array все параметры по которым считается, args - начальное время и планеты
        """
        state = FlybyDomain.DomainPoint.create_point_with_meta(args, domain)
        last_state: LastState = args[-1]

        flyby_planet_state = state.departure_planet.ephemeris_at_time(
            state.initial_time, last_state.total_flight_time
        )
        spacecraft_flyby_excess_velocity = last_state.velocity - flyby_planet_state.velocity

        u_p = spacecraft_flyby_excess_velocity / np.linalg.norm(spacecraft_flyby_excess_velocity)
        rotation_matrix = FlybyDomain.generate_rotation_matrix(u_p, state.gamma)

        n = np.cross(spacecraft_flyby_excess_velocity, flyby_planet_state.velocity)
        n = n / np.linalg.norm(n)

        n_trajectory_plane = rotation_matrix @ n
        mu = gravitational_constant * float(state.departure_planet.mass / u.kg) / 1000 ** 3

        betta = np.arccos(1 / (1 + state.periapsis * np.linalg.norm(spacecraft_flyby_excess_velocity) ** 2 / mu))

        leading_flyby = True
        if np.dot(spacecraft_flyby_excess_velocity, flyby_planet_state.velocity) < 0:
            leading_flyby = False

        turn_angle = 2 * betta
        if leading_flyby:
            turn_angle *= -1

        rotation_matrix = FlybyDomain.generate_rotation_matrix(n_trajectory_plane, turn_angle)
        spacecraft_flyby_excess_velocity = rotation_matrix @ spacecraft_flyby_excess_velocity

        spacecraft_departure_velocity = spacecraft_flyby_excess_velocity + flyby_planet_state.velocity
        spacecraft_departure_state = OrbitStateVector(
            flyby_planet_state.radius,
            spacecraft_departure_velocity
        )

        solver = UniversalTimeSolver(spacecraft_departure_state, Planet(Sun))
        mid_state = solver.state_after(state.alpha * state.flight_period)

        arrival_planet_state = state.arrival_planet.ephemeris_at_time(
            state.initial_time, last_state.total_flight_time + state.flight_period
        )
        try:
            problem = LambertProblem(
                mid_state.radius, arrival_planet_state.radius,
                (1 - state.alpha) * state.flight_period,
                mu_sun
            )
            start_state, end_state = problem.solution()
        except RuntimeError:
            # some value if it fails, 1_000 delta-v is unreachable (nan + overflow errors)
            return 1_000

        # if last planet -> add braking delta-velocity
        # (v_p_hyp - v_p_hyp / np.sqrt(2))
        # ------------------------------------------------ last planet ----------------------------
        flyby_planet_state = state.arrival_planet.ephemeris_at_time(
            state.initial_time, last_state.total_flight_time + state.flight_period
        )
        excess_velocity = end_state.velocity - flyby_planet_state.velocity

        v_p_hyp = np.sqrt(np.linalg.norm(excess_velocity) ** 2 + 2 *
                          (state.arrival_planet.mass / u.kg * gravitational_constant / 10 ** 9) / (220 + 6000))

        delta_v = (v_p_hyp - np.sqrt(
            (state.arrival_planet.mass / u.kg * gravitational_constant / 10 ** 9) / (220 + 6000))) + np.linalg.norm(
            start_state.velocity - mid_state.velocity)

        # delta_v = np.linalg.norm(start_state.velocity - mid_state.velocity) + last_state.total_delta_v
        if 'last_state_generation' in kwargs and kwargs['last_state_generation']:
            return end_state.velocity, delta_v

        return delta_v

    @staticmethod
    def generate_rotation_matrix(u, gamma):
        return np.array([
            [
                np.cos(gamma) + u[0] ** 2 * (1 - np.cos(gamma)),
                u[0] * u[1] * (1 - np.cos(gamma)) - u[2] * np.sin(gamma),
                u[0] * u[2] * (1 - np.cos(gamma)) + u[1] * np.sin(gamma)
            ],
            [
                u[1] * u[0] * (1 - np.cos(gamma)) + u[2] * np.sin(gamma),
                np.cos(gamma) + u[1] ** 2 * (1 - np.cos(gamma)),
                u[1] * u[2] * (1 - np.cos(gamma)) - u[0] * np.sin(gamma)
            ],
            [
                u[2] * u[0] * (1 - np.cos(gamma)) - u[1] * np.sin(gamma),
                u[2] * u[1] * (1 - np.cos(gamma)) + u[0] * np.sin(gamma),
                np.cos(gamma) + u[2] ** 2 * (1 - np.cos(gamma))
            ]
        ])


class LastState:
    def __init__(self, total_flight_time, total_delta_v, velocity, point):
        self.total_flight_time = total_flight_time
        self.total_delta_v = total_delta_v
        self.velocity = velocity
        self.points_sequence = point

    def updated(self, flight_time, delta_v, velocity, point: FlybyDomain.DomainPoint):
        new_seq = deepcopy(self.points_sequence)
        new_seq.append(point)

        return LastState(
            self.total_flight_time + flight_time,
            self.total_delta_v + delta_v,
            velocity,
            new_seq
        )

    def to_dict(self):
        return {
            'total_flight_time': self.total_flight_time,
            'total_delta_v': self.total_delta_v,
            'velocity': self.velocity,
            'points_sequence': [
                {
                    "initial_time": point.initial_time,
                    "departure_planet": point.departure_planet.planet,
                    "arrival_planet": point.arrival_planet.planet,

                    "v_start": point.v_start,
                    "launch_time": point.launch_time,
                    "alpha": point.alpha,
                    "flight_period": point.flight_period,
                    "incl": point.incl,
                    "decl": point.decl,
                }
                if isinstance(point, InitialDomain.DomainPoint) else
                {
                    "initial_time": point.initial_time,
                    "departure_planet": point.departure_planet.planet,
                    "arrival_planet": point.arrival_planet.planet,
                    "gamma": point.gamma,
                    "periapsis": point.periapsis,
                    "alpha": point.alpha,
                    "flight_period": point.flight_period,
                }
                for point in self.points_sequence
            ],
        }


class ManeuversSequence:
    def __init__(self, domains_sequence):
        self.domains_sequence = domains_sequence

    def run(self):
        initial_set = self.domain_solver(self.domains_sequence[0], is_flyby=False)

        print('second-----------------')
        for domain in self.domains_sequence[1:]:
            initial_set = self.domain_solver(domain, allowed_points=initial_set)

        return initial_set

    @staticmethod
    def domain_solver(domain: InitialDomain, allowed_points=None, is_flyby=True) -> list[LastState | None]:
        points_set = []
        n_eval = 0
        meta = domain.meta()
        while True:
            print('new_point_generation')

            prev_point_state = None
            if is_flyby:
                prev_point_state: LastState = random.choice(allowed_points)

            x0 = domain.generate_point().to_numpy_array()
            bounds = domain.generate_bounds()

            x_local_minimum = optimize.minimize(
                domain.cost_function, x0, args=meta + (prev_point_state,), bounds=bounds
            ).x
            n_trials = 0
            print(domain.cost_function(x0, *(meta + (prev_point_state,))))
            while True:
                x_candidate = ManeuversSequence.random_point_nearby_given_point(x_local_minimum, 0.15, bounds)

                n_eval += 1
                n_trials += 1

                if n_trials > n_trials_max:
                    break

                if domain.cost_function(x_candidate, *meta, prev_point_state) < 20:
                    p = domain.create_point(x_candidate)
                    velocity, delta_v = domain.cost_function(x_candidate, *meta, prev_point_state,
                                                             last_state_generation=True)
                    points_set.append(
                        prev_point_state.updated(p.flight_period, delta_v, velocity, p) if is_flyby else
                        LastState(p.flight_period + p.launch_time, delta_v, velocity, [p])
                    )
                    print(len(points_set))

                x_local_minimum_new = optimize.minimize(
                    domain.cost_function, x_candidate, args=meta + (prev_point_state,), bounds=bounds
                ).x
                n_eval += 1

                new_cost = domain.cost_function(x_local_minimum_new, *meta, prev_point_state)
                old_cost = domain.cost_function(x_local_minimum, *meta, prev_point_state)

                if new_cost < old_cost:
                    if new_cost < old_cost - 0.1:
                        n_trials = 0

                    x_local_minimum = x_local_minimum_new
                    if domain.cost_function(x_local_minimum_new, *meta, prev_point_state) < 20:
                        p = domain.create_point(x_local_minimum_new)
                        velocity, delta_v = domain.cost_function(x_local_minimum_new, *meta, prev_point_state,
                                                                 last_state_generation=True)
                        points_set.append(
                            prev_point_state.updated(p.flight_period, delta_v, velocity, p) if is_flyby else
                            LastState(p.flight_period + p.launch_time, delta_v, velocity, [p])
                        )

                if n_eval > n_max and len(points_set) > 1:
                    return points_set

    @staticmethod
    def random_point_nearby_given_point(base_point, r, constraints):
        if r >= 1:
            raise RuntimeError("Radius is too big")

        dim = len(base_point)
        random_range = random.default_rng().uniform(low=-r, high=r, size=(1, dim))[0]

        point_in_sphere = base_point.copy()
        for i in range(dim):
            lower_bound, upper_bound = constraints[i]
            point_in_sphere[i] += random_range[i] * (upper_bound - lower_bound)
            point_in_sphere[i] = max(min(point_in_sphere[i], upper_bound), lower_bound)

        return point_in_sphere


n_trials_max = 3
n_max = 150

flight_period_min = 1 * 31 * 6 * 60 * 60
flight_period_max = 12 * 31 * 6 * 60 * 60

starting_domain = InitialDomain(
    0,
    KspPlanet("Kerbin"),
    KspPlanet("Duna"),
    Constraint(0.1, 2),  # excess velocity
    Constraint(0, 12 * 31 * (6 * 60 * 60)),  # first maneuver time limit
    Constraint(0.01, 0.99),  # alpha
    Constraint(flight_period_min, flight_period_max),  # total flight time for arc
    Constraint(0, 1),  # inclination
    Constraint(0, 1),  # declination
)
#
flight_period_min = 1 * 31 * 24 * 60 * 60
flight_period_max = 5 * 12 * 31 * 24 * 60 * 60
#
first_flyby_domain = FlybyDomain(
    0,
    KspPlanet("Duna"),
    KspPlanet("Jool"),
    Constraint(0, 1),
    Constraint(52 + 320, 3000 + 320),
    Constraint(0.01, 0.99),
    Constraint(flight_period_min, flight_period_max),
)

if __name__ == "__main__":
    seq = ManeuversSequence([starting_domain, first_flyby_domain])  # , first_flyby_domain
    result = seq.run()

    point = min(result, key=lambda x: x.total_delta_v)

    print(point.points_sequence[0], point.total_delta_v)

    import dill

    # Save the file
    dill.dump([i.to_dict() for i in result], file=open("result_ksp.pickle", "wb"))

    # Reload the file
    # bounds = [
    #     (v_start_min, v_start_max),
    #     (t_launch_min, t_launch_max),
    #     (0.01, 0.99),
    #     (flight_period_min, flight_period_max),
    #     (0, 1),
    #     (0, 1),
    # ]

    # points = initial_domain(0, 8 * 31 * 24 * 60 * 60, 2, 5, Mars)
    # print(
    #     len(points),
    #     min(points, key=lambda x: f_velocity_cost(x)),
    #     f_velocity_cost(min(points, key=lambda x: f_velocity_cost(x)))
    # )
    #
    # with open('test.npy', 'wb') as f:
    #     np.save(f, points)
