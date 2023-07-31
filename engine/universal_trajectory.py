import numpy as np
import poliastro
from poliastro.bodies import Body, SolarSystemPlanet
from scipy import optimize

from engine.state_vector import OrbitStateVector
from astropy import units as u
from poliastro.twobody import Orbit


class UniversalTimeSolver:
    def __init__(self, initial: OrbitStateVector, planet):
        self.initial = initial

        r = initial.radius << u.km
        v = initial.velocity << u.km / u.s

        self.orb = Orbit.from_vectors(planet.planet, r, v)

    def state_after(self, seconds: float):
        after_seconds = self.orb.propagate(int(seconds) << u.s)
        # return after_seconds
        return OrbitStateVector(after_seconds.r / u.km, after_seconds.v / u.km * u.s) # TODO change back
