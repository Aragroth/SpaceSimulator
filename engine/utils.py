from __future__ import annotations

import numpy as np

from engine.mga import InitialDomain, FlybyDomain, LastState
from engine.planets.abstract import AbstractPlanet


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


def points_json_decoder(points_json, planet: AbstractPlanet):
    return [
        LastState(
            i['total_flight_time'], i['total_delta_v'], i['velocity'],
            [
                InitialDomain.DomainPoint(
                    x["initial_time"],
                    planet.from_json(x["departure_planet"]),
                    planet.from_json(x["arrival_planet"]),
                    x["v_start"],
                    x["launch_time"],
                    x["alpha"],
                    x["flight_period"],
                    x["incl"],
                    x["decl"],
                ) if 'incl' in x else
                FlybyDomain.DomainPoint(
                    x["initial_time"],
                    planet.from_json(x["departure_planet"]),
                    planet.from_json(x["arrival_planet"]),
                    x["gamma"],
                    x["periapsis"],
                    x["alpha"],
                    x["flight_period"],
                )
                for x in i['points_sequence']
            ]
        )
        for i in points_json
    ]
