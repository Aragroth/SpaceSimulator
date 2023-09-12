from __future__ import annotations

from copy import deepcopy

from numpy import random
from scipy import optimize

from engine.mga import InitialDomain, FlybyDomain


class ManeuversSequence:
    def __init__(self, domains_sequence, n_trials_max=4, n_max=100):
        self.domains_sequence = domains_sequence
        self.n_trials_max = n_trials_max
        self.n_max = n_max

    def run(self):
        initial_set = self.domain_solver(self.domains_sequence[0], is_flyby=False)

        print('second-----------------')
        for domain in self.domains_sequence[1:]:
            initial_set = self.domain_solver(domain, allowed_points=initial_set)

        return initial_set

    def domain_solver(self, domain: InitialDomain, allowed_points=None, is_flyby=True) -> list[LastState | None]:
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

                if n_trials > self.n_trials_max:
                    break

                if domain.cost_function(x_candidate, *meta, prev_point_state) < 30:
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
                    if domain.cost_function(x_local_minimum_new, *meta, prev_point_state) < 30:
                        p = domain.create_point(x_local_minimum_new)
                        velocity, delta_v = domain.cost_function(x_local_minimum_new, *meta, prev_point_state,
                                                                 last_state_generation=True)
                        points_set.append(
                            prev_point_state.updated(p.flight_period, delta_v, velocity, p) if is_flyby else
                            LastState(p.flight_period + p.launch_time, delta_v, velocity, [p])
                        )

                if n_eval > self.n_max and len(points_set) > 1:
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
                    "departure_planet": point.departure_planet.to_json(),
                    "arrival_planet": point.arrival_planet.to_json(),

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
                    "departure_planet": point.departure_planet.to_json(),
                    "arrival_planet": point.arrival_planet.to_json(),
                    "gamma": point.gamma,
                    "periapsis": point.periapsis,
                    "alpha": point.alpha,
                    "flight_period": point.flight_period,
                }
                for point in self.points_sequence
            ],
        }
