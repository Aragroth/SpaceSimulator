import numpy as np
from scipy import optimize

from typing import Tuple
from engine.state_vector import StateVector


class LambertProblem:
    def __init__(self, r1: np.array, r2: np.array, time_seconds: float, planet, is_prograde: bool = True):
        self.r1 = r1
        self.r2 = r2
        self.time_seconds = time_seconds
        self.mu = planet.mu
        self.planet = planet
        self.is_prograde = is_prograde

    def solution(self) -> Tuple[StateVector, StateVector]:
        self.r1_module = np.linalg.norm(self.r1)
        self.r2_module = np.linalg.norm(self.r2)

        self.tetta = self.calculate_tetta()
        self.A = self.calculate_A()

        self.calculate_lagrange()

        v1 = 1 / self.g * (self.r2 - self.f * self.r1)
        v2 = 1 / self.g * (self.g_dot * self.r2 - self.r1)

        return StateVector(self.r1, v1, self.planet), StateVector(self.r2, v2, self.planet)

    def calculate_lagrange(self):
        F, F_dot = self.get_optimize_function()
        z = optimize.newton(F, 6) # many errors here, including F_DOT

        y = self.r1_module + self.r2_module + self.A * \
            (z * self.stumpff_S(z) - 1) / np.sqrt(self.stumpff_C(z))

        self.f = 1 - y / self.r1_module
        self.g = self.A * np.sqrt(y / self.mu)
        self.g_dot = 1 - y / self.r2_module

    def calculate_tetta(self):
        res = np.dot(self.r1, self.r2) / (self.r1_module * self.r2_module)
        radians = np.arccos(res)
        if self.is_prograde:
            if np.cross(self.r1, self.r2)[2] >= 0:
                answer = radians
            else:
                answer = 2 * np.pi - radians
        else:
            if np.cross(self.r1, self.r2)[2] < 0:
                answer = radians
            else:
                answer = 2 * np.pi - radians
        return answer

    def calculate_A(self):
        return np.sin(self.tetta) * np.sqrt(self.r1_module * self.r2_module / (1 - np.cos(self.tetta)))

    @staticmethod
    def stumpff_C(z):
        if (z > 0):
            return (1 - np.cos(np.sqrt(z))) / z
        elif (z < 0):
            return (np.cosh(np.sqrt(-z)) - 1) / (-z)
        return 1/2

    @staticmethod
    def stumpff_S(z):
        if (z > 0):
            return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z) ** 3)
        elif (z < 0):
            return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z) ** 3)
        return 1/6

    def get_optimize_function(self):
        A = self.A
        r1, r2 = self.r1_module, self.r2_module
        S = self.stumpff_S
        C = self.stumpff_C
        mu = self.mu
        time = self.time_seconds

        def F(z):
            y = r1 + r2 + A * (z * S(z) - 1) / np.sqrt(C(z))
            return (y / C(z)) ** (3/2) * S(z) + A * \
                np.sqrt(y) - np.sqrt(mu) * time

        def F_dot(z):
            y = r1 + r2 + A * (z * S(z) - 1) / np.sqrt(C(z))
            if z == 0:
                np.sqrt(2) / 40 * y ** (3 / 2) + A / 8 * (np.sqrt(y) + A * np.sqrt(1 / 2 / y))
            
            return (y / C(z)) ** (3/2) * ( 1 / (2 * z) * (C(z) - 3/2 * (S(z) / C(z)) + 3 / 4 * (S(z)**2 / C(z)))) + A / 8  * (3 * S(z) / C(z) * np.sqrt(y) + A * np.sqrt(C(z) / y))
        
        return F, F_dot


if (__name__ == "__main__"):
    # test, example 5.2
    pr = LambertProblem(
        np.array([5_000, 10_000, 2_100]),
        np.array([-14_600, 2_500, 7_000]),
        60 * 60,
        398_600
    )
    state_start, state_end = pr.solution()
    print(state_start, state_end)

    z = -228581412.4848671
    print(LambertProblem.stumpff_S(z))
