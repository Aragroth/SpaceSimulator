from abc import ABC, abstractmethod

from astropy import units as u
from astropy.time import Time
from scipy.constants import gravitational_constant

from engine.state_vector import StateVector


class AbstractPlanet(ABC):
    @abstractmethod
    def __init__(self, astropy_planet):
        self.astropy_planet = astropy_planet

    @abstractmethod
    def ephemeris_at_time(
            self, initial_time: Time, delta_time: float, astropy_planet=True
    ) -> StateVector:
        pass

    def to_json(self) -> str:
        return self.astropy_planet.name

    @property
    def mu(self) -> float:
        return gravitational_constant * self.astropy_planet.mass.to_value(u.kg) / 10 ** 9

    @property
    def radius(self) -> float:
        return self.astropy_planet.R.to_value(u.km)
