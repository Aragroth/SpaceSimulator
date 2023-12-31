from engine.planets.abstract import AbstractPlanet
from engine.propagator.abstract import AbstractUniversalPropagator
from engine.state_vector import StateVector
from astropy import units as u
from poliastro.twobody import Orbit


class UniversalPropagator(AbstractUniversalPropagator):
    def __init__(self, initial: StateVector, planet: AbstractPlanet):
        self.initial = initial
        self.planet = planet
        r = initial.radius << u.km
        v = initial.velocity << u.km / u.s

        self.orb = Orbit.from_vectors(planet.astropy_planet, r, v)

    def state_after(self, seconds: float) -> StateVector:
        after_seconds = self.orb.propagate(int(seconds) << u.s)
        # return after_seconds
        return StateVector(after_seconds.r / u.km, after_seconds.v / u.km * u.s, self.planet)
