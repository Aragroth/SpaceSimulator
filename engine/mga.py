# Set feasible set D to the whole domain, Start from level 1: i ← 1
# 2: While i < nlevels, Do
# 3: Run search for feasible solutions at level i on feasible set ¯Di−1 × DL,i
# 4: Cluster the feasible solutions
# 5: Prune the search space at level i and define new feasible set ¯DL,i at level i
# 6: Apply back-pruning and redefine feasible sets ¯DL,j at levels j = 1, . . . , i − 1
# 7: Define the new feasible set ¯Di =
# i(−1
# j=1
# ¯D
# L,j × ¯DL,i
# 8: i ← i + 1; End Do


# Select x in Di, initialize n_eval = 0, ntrials = 0
# 2: Run local optimizer from x to local minimum xl
# 3: Select a candidate point xc ∈ N (xl , ρl); update neval ;ntrials ← ntrials + 1
# 4: If ntrials > ntmax
# 5: goto Step 1
# 6: End If
# 7: If
# i(xi) Then
# 8: Af ← Af ∪ {xc} ; goto Step 3
# 9: End If
# 10: Run local optimizer from xc → ¯xl , update neval
# 11: If fi (¯xl) ≤ fi (xl) Then
# 12: xl ← ¯xl
# 13: If
# i(xi)
# 14: Af ← Af ∪ {xl}
# 15: End If
# 16: If fi (¯xl) < fi (xl)
# 17: ntrials = 0
# 18: End If
# 19: End If
# 20: Termination Unless neval ≥ nmax, goto Step 3

from __future__ import annotations

from typing import List

import numpy as np
from astropy import units as u
from astropy.time import Time
from numpy import random
from poliastro.bodies import Mars, Earth
from poliastro.ephem import Ephem
from scipy import optimize

from engine.lambert_problem import LambertProblem
from engine.state_vector import OrbitStateVector

mu = 398_600
r0 = 6_378 + 400
mu_sun = 1.327 * (10 ** 11)

from engine.ellipsoid_trajectory import EllipsoidTrajectoryTimeSolver


class Constraint:
    def __init__(self, minimum, maximum):
        self.min = minimum
        self.max = maximum


class Planet:
    def __init__(self, planet):
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
    def cost_function(domain: np.array, *args) -> float:
        """
        domain_array все параметры по которым считается, args - начальное время и планеты
        """
        state = InitialDomain.DomainPoint.create_point_with_meta(args, domain)

        excess_velocity = state.v_start * np.array([
            np.cos(state.incl) * np.cos(state.decl),
            np.sin(state.incl) * np.cos(state.decl),
            np.sin(state.decl)
        ])

        v_p_hyp = np.sqrt(np.linalg.norm(excess_velocity) ** 2 + 2 * mu / (400 + 6378))

        departure_planet_state = state.departure_planet.ephemeris_at_time(state.initial_time, state.launch_time)
        spacecraft_state_sun = OrbitStateVector(
            departure_planet_state.radius,
            excess_velocity + departure_planet_state.velocity,
            mu_sun
        )

        solver = EllipsoidTrajectoryTimeSolver(spacecraft_state_sun, mu_sun)
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

        return (v_p_hyp - v_p_hyp / np.sqrt(2)) + np.linalg.norm(start_state.velocity - mid_state.velocity)


class ManeuversSequence:

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

    @staticmethod
    def domain_solver(domain: InitialDomain) -> List[InitialDomain.DomainPoint]:
        points_set = []
        n_eval = 0
        meta = domain.meta()
        while True:
            print('new_point_generation')

            x0 = domain.generate_point().to_numpy_array()
            bounds = domain.generate_bounds()

            x_local_minimum = optimize.minimize(
                domain.cost_function, x0, args=meta, bounds=bounds
            ).x
            n_trials = 0

            while True:
                x_candidate = ManeuversSequence.random_point_nearby_given_point(x_local_minimum, 0.15, bounds)

                n_eval += 1
                n_trials += 1

                if n_trials > n_trials_max:
                    break

                if domain.cost_function(x_candidate, *meta) < 10:
                    points_set.append(
                        domain.create_point(x_candidate)
                    )
                    print(len(points_set))

                x_local_minimum_new = optimize.minimize(
                    domain.cost_function, x_candidate, args=meta, bounds=bounds
                ).x
                n_eval += 1

                new_cost = domain.cost_function(x_local_minimum_new, *meta)
                old_cost = domain.cost_function(x_local_minimum, *meta)

                if new_cost < old_cost:
                    if new_cost < old_cost - 0.1:
                        n_trials = 0

                    x_local_minimum = x_local_minimum_new
                    if domain.cost_function(x_local_minimum_new, *meta) < 10:
                        points_set.append(
                            domain.create_point(x_local_minimum_new)
                        )

                if n_eval > n_max and len(points_set) > 1:
                    return points_set


n_trials_max = 6
n_max = 500

flight_period_min = 7 * 31 * 24 * 60 * 60
flight_period_max = 9 * 31 * 24 * 60 * 60

starting_domain = InitialDomain(
    Time("2020-06-20 12:00"),
    Planet(Earth),
    Planet(Mars),
    Constraint(1, 4),  # excess velocity
    Constraint(0, 3 * 31 * (24 * 60 * 60)),  # first maneuver time limit
    Constraint(0.01, 0.99),  # alpha
    Constraint(flight_period_min, flight_period_max),  # total flight time for arc
    Constraint(0, 1),  # inclination
    Constraint(0, 1),  # declination
)

if __name__ == "__main__":
    print(
        ManeuversSequence.domain_solver(starting_domain)
    )
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
