from abc import ABC, abstractmethod

from engine.state_vector import StateVector


class AbstractUniversalPropagator(ABC):
    """
    Gets StateVector and Planet
    """

    @abstractmethod
    def state_after(self, seconds: float) -> StateVector:
        ...
