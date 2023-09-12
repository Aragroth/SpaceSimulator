from __future__ import annotations

import numpy as np
from numpy import random

from engine.planets.abstract import AbstractPlanet
from engine.planets.solar import SolarPlanet
from engine.mga.constraint import Constraint


class InitialDomain:
    def __init__(
            self, initial_time, departure_planet: AbstractPlanet, arrival_planet: AbstractPlanet,
            v_start_c: Constraint, launch_time_c: Constraint, alpha_c: Constraint, flight_period_c: Constraint,
            inclination_c: Constraint, declination_c: Constraint, cost_function
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

        self.cost_function = cost_function

    class DomainPoint:
        def __init__(
                self, initial_time, departure_planet: SolarPlanet, arrival_planet: SolarPlanet,
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
