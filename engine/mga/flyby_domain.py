from __future__ import annotations

import numpy as np
from numpy import random

from engine.planets.solar import SolarPlanet
from engine.utils import Constraint


class FlybyDomain:

    def __init__(
            self, initial_time, departure_planet: SolarPlanet, arrival_planet: SolarPlanet,
            # total time flight and speed current
            gamma_c: Constraint, periapsis_c: Constraint, alpha_c: Constraint, flight_period_c: Constraint,
            cost_function
    ):
        self.initial_time = initial_time
        self.departure_planet = departure_planet
        self.arrival_planet = arrival_planet

        self.gamma_constraint = gamma_c
        self.periapsis_constraint = periapsis_c
        self.alpha_constraint = alpha_c
        self.flight_period_constraint = flight_period_c
        self.cost_function = cost_function

    class DomainPoint:
        def __init__(
                self, initial_time, departure_planet: SolarPlanet, arrival_planet: SolarPlanet,
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

    def create_point(self, domain: np.array):
        return FlybyDomain.DomainPoint(
            self.initial_time, self.departure_planet, self.arrival_planet,
            domain[0], domain[1], domain[2], domain[3]
        )

    def meta(self):
        return self.initial_time, self.departure_planet, self.arrival_planet

    def generate_point(self) -> FlybyDomain.DomainPoint:
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
