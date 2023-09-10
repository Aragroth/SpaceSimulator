from __future__ import annotations

import numpy as np


class Constraint:
    def __init__(self, minimum, maximum):
        self.min = minimum
        self.max = maximum


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