import numpy as np
from astropy import units as u
from poliastro.twobody import Orbit
from poliastro.twobody.angles import (
    D_to_nu, E_to_nu, F_to_nu, M_to_D, M_to_E, M_to_F,
)

from autopilot.planets import Kerbin
from engine.planets.ksp_astropy import Duna, Kerbin, Kerbol
from engine.state_vector import StateVector
from .abstract import AbstractPlanet


class KspPlanet(AbstractPlanet):

    def __init__(self, astropy_planet):
        self.astropy_planet = astropy_planet

    def ephemeris_at_time(self, initial_time, delta_time, astropy_planet=False) -> StateVector:
        if self.astropy_planet == Kerbin:
            a = 13599840256 * u.m
            ecc = 0.0 * u.one
            inc = 0 * u.deg
            raan = 0 * u.deg
            argp = 0 * u.deg
            nu = self.true_anomaly_from_mean_poliastro(ecc, 3.140 << u.rad)

        elif self.astropy_planet == Duna:
            a = 20726155264 * u.m
            ecc = 0.051 * u.one
            inc = 0.060 * u.deg
            raan = 135.5 * u.deg
            argp = 0 * u.deg
            nu = self.true_anomaly_from_mean_poliastro(ecc, 3.140 << u.rad)

        elif self.astropy_planet == "Jool":
            a = 68773560320 * u.m
            ecc = 0.05 * u.one
            inc = 1.304 * u.deg
            raan = 52 * u.deg
            argp = 0 * u.deg
            nu = self.true_anomaly_from_mean_poliastro(ecc, 0.100 << u.rad)

        elif self.astropy_planet == "Dres":
            a = 40839348203 * u.m
            ecc = 0.145 * u.one
            inc = 5 * u.deg
            raan = 280 * u.deg
            argp = 90 * u.deg
            nu = self.true_anomaly_from_mean_poliastro(ecc, 3.140 << u.rad)

        # TODO check this data in ksp
        elif self.astropy_planet == "Eve":
            a = 9832684544 * u.m
            ecc = 0.01 * u.one
            inc = 2.1 * u.deg
            raan = 15 * u.deg
            argp = 0 * u.deg
            nu = self.true_anomaly_from_mean_poliastro(ecc, 3.140 << u.rad)

        orb = Orbit.from_classical(Kerbol, a, ecc, inc, raan, argp, nu)
        planet_state = orb.propagate((initial_time + delta_time) * u.s)
        return StateVector(planet_state.r / u.km, planet_state.v / u.km * u.s, self)

    @staticmethod
    def true_anomaly_from_mean_poliastro(ecc, M):
        if ecc < 1:
            M = (M + np.pi * u.rad) % (2 * np.pi * u.rad) - np.pi * u.rad
            return E_to_nu(M_to_E(M, ecc), ecc)
        elif ecc == 1:
            return D_to_nu(M_to_D(M))
        else:
            return F_to_nu(M_to_F(M, ecc), ecc)


if __name__ == '__main__':
    p = KspPlanet("Dres")
    print(
        p.ephemeris_at_time(0, 0).radius
    )
