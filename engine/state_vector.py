import numpy as np


class OrbitStateVector:
    def __init__(self, r, v, mu = 398_600):
        self.radius = np.array(r)
        self.velocity = np.array(v)

        self.mu = mu

        self.angular_momentum = self.calculate_angular_momentum()
        self.inclination = self.calculate_inclination()
        self.node = self.calculate_node()
        self.right_ascention = self.calculate_right_ascention()
        self.eccentricity = self.calculate_eccentricity()
        self.argument_of_perigee = self.calculate_argument_of_perigee()
        # self.true_anomaly = self.calculate_true_anomaly() ----------- error if e = 0 #TODO

    def calculate_angular_momentum(self):
        return np.cross(self.radius, self.velocity)

    def calculate_inclination(self):
        return np.arccos(self.angular_momentum[2] / self.angular_momentum_module)

    def calculate_node(self):
        return np.cross([0, 0, 1], self.angular_momentum)

    def calculate_right_ascention(self):
        if self.node_module == 0:
            return 0

        res = np.arccos(self.node[0] / self.node_module)
        return res if self.node[1] >= 0 else (2 * np.pi - res)

    def calculate_eccentricity(self):
        res_r = (self.velocity_module ** 2 - self.mu /
                 self.radius_module) * self.radius
        res_v = self.radius_module * self.radial_velocity * self.velocity
        return (1 / self.mu) * (res_r - res_v)

    def calculate_argument_of_perigee(self):
        if self.eccentricity_module == 0 or self.node_module == 0:
            return 0
        res = np.arccos(self.node @ self.eccentricity /
                        (self.node_module * self.eccentricity_module))
        return res if self.eccentricity[2] >= 0 else (2 * np.pi - res)

    def calculate_true_anomaly(self):
        if (self.eccentricity_module == 0): return 0

        res = np.arccos(self.eccentricity @ self.radius /
                        (self.eccentricity_module * self.radius_module))
        return res if self.radial_velocity >= 0 else (2 * np.pi - res)

    @property
    def angular_momentum_module(self):
        return np.linalg.norm(self.angular_momentum)

    @property
    def radius_module(self):
        return np.linalg.norm(self.radius)

    @property
    def velocity_module(self):
        return np.linalg.norm(self.velocity)

    @property
    def radial_velocity(self):
        return self.radius @ self.velocity.T / self.radius_module

    @property
    def node_module(self):
        return np.linalg.norm(self.node)

    @property
    def eccentricity_module(self):
        return np.linalg.norm(self.eccentricity)
    
    def __str__(self):
        return f"({str(self.radius)}, {str(self.velocity)})"



