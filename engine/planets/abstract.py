from abc import ABC, abstractmethod

from astropy.time import Time

from engine.state_vector import StateVector


class AbstractPlanet(ABC):
    """
    Should store `astropy_planet`
    """

    @abstractmethod
    def ephemeris_at_time(
            self, initial_time: Time, delta_time: float, astropy_planet=True
    ) -> StateVector:
        pass

    @property
    @abstractmethod
    def mu(self) -> float:
        pass

