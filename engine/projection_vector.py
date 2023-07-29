
import numpy as np


class ProjectionVector:
    def __init__(self, radius):
        self.radius = radius

    def projection_after(self, seconds: int):
        tetta = 2 * np.pi * (1 + 1/365.26) / (24 * 3600) * seconds
        rot_tr = np.array([
            [np.cos(tetta), np.sin(tetta), 0],
            [-np.sin(tetta), np.cos(tetta), 0],
            [0, 0, 1],
        ])
        res = rot_tr @ self.radius
        r = np.linalg.norm(res)
        self.decl = np.arcsin(res[2] / r)

        l = res[0] / r
        m = res[1] / r
        asc = np.arccos(l / np.cos(self.decl))
        self.asc = asc if m > 0 else 2 * np.pi - asc

