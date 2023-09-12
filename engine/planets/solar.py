from __future__ import annotations

import astropy
import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Body, Sun, Earth, Mars, Venus
from poliastro.ephem import Ephem
from scipy.constants import gravitational_constant

from .abstract import AbstractPlanet
from engine.state_vector import StateVector


class SolarPlanet(AbstractPlanet):
    def __init__(self, planet: Body):
        self.mass = planet.mass
        self.astropy_planet = planet

    def ephemeris_at_time(
            self, initial_time: astropy.time.Time, delta_time: float, f=True
    ) -> StateVector:
        epoch = Time(initial_time.jd + delta_time / (60 * 60 * 24), format='jd')
        planet_ephem = Ephem.from_body(self.astropy_planet, epoch.tdb)

        r, v = np.array([0, 0, 0]), np.array([0, 0, 0])
        if f:
            position = planet_ephem.rv()[0] << u.km
            speed = planet_ephem.rv()[1] << u.km / u.s
            r = np.array(position[0])
            v = np.array(speed[0])

        return StateVector(r, v, self)

    @classmethod
    def from_json(cls, planet_name):
        match planet_name:
            case 'Sun': return cls(Sun)
            case 'Earth': return cls(Earth)
            case 'Mars': return cls(Mars)
            case 'Venus': return cls(Venus)

        raise RuntimeError("Planet not found")
