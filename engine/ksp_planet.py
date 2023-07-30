import numpy as np
from astropy import units as u
from astropy.constants import Constant
from astropy.constants.iau2015 import IAU2015
from astropy.time import Time
from numpy import random, ndarray
from poliastro.bodies import Mars, Earth, Venus, Sun, Body
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.twobody.angles import (
    D_to_nu, E_to_nu, F_to_nu, M_to_D, M_to_E, M_to_F,
)

from engine.state_vector import OrbitStateVector


GM_kerbol = Constant(
    "GM_sun",
    "kerbol gravitational constant",
    1.1723328e18,
    "m3 / (s2)",
    0.0000000001e20,
    "IAU 2009 system of astronomical constants",
    system="si",
)

R_kerbol = Constant(
    "R_sun",
    "kerbol equatorial radius",
    261600000,
    "m",
    0,
    "IAU Working Group on Cartographic Coordinates and Rotational Elements: 2015",
    system="si",
)

Kerbol = Body(
    parent=None,
    k=GM_kerbol,
    name="Kerbol",
    symbol="\u2609",
    R=R_kerbol,
)

class KspPlanet:
    def __init__(self, planet_name):
        self.planet = planet_name
        if self.planet == "Kerbin":
            self.mass = 5.2915158 * 10 ** 22 << u.kg
        elif self.planet == "Duna":
            self.mass = 4.5154270 * 10 ** 21 << u.kg
        elif self.planet == "Jool":
            self.mass = 4.2332127 * 10 ** 24 << u.kg


    def ephemeris_at_time(self, initial_time, delta_time, astropy_planet=False) -> OrbitStateVector:
        if self.planet == "Kerbin":
            a = 13599840256 << u.m
            ecc = 0 << u.one
            inc = 0 << u.deg
            raan = 49.562 << u.deg
            argp = 0 << u.deg
            nu = self.true_anomaly_from_mean_poliastro(ecc, 3.140 * u.rad) << u.deg

        elif self.planet == "Duna":
            a = 20726155264 << u.m
            ecc = 0.051 << u.one
            inc = 0.060 << u.deg
            raan = 135.5 << u.deg
            argp = 0 << u.deg
            nu = self.true_anomaly_from_mean_poliastro(ecc, 3.140 * u.rad) << u.deg

        elif self.planet == "Jool":
            a = 68773560320 << u.m
            ecc = 0.05 << u.one
            inc = 1.304 << u.deg
            raan = 52 << u.deg
            argp = 0 << u.deg
            nu = self.true_anomaly_from_mean_poliastro(ecc, 0.100 * u.rad) << u.deg

        orb = Orbit.from_classical(Kerbol, a, ecc, inc, raan, argp, nu)
        planet_state = orb.propagate((initial_time + delta_time) << u.s)
        return OrbitStateVector(planet_state.r / u.km, planet_state.v / u.km * u.s)

    @staticmethod
    def true_anomaly_from_mean_poliastro(ecc, M):
        if ecc < 1:
            M = (M + np.pi * u.rad) % (2 * np.pi * u.rad) - np.pi * u.rad
            return E_to_nu(M_to_E(M, ecc), ecc).to(u.deg)
        elif ecc == 1:
            return D_to_nu(M_to_D(M)).to(u.deg)
        else:
            return F_to_nu(M_to_F(M, ecc), ecc).to(u.deg)
