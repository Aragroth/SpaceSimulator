
import numpy as np
from scipy import optimize

from engine.state_vector import OrbitStateVector


class EllipsoidTrajectoryTimeSolver:
    def __init__(self, initial: OrbitStateVector, mu):
        self.initial = initial
        self.mu = mu

        self.e = self.initial.eccentricity_module
        self.radius_coeff = self.initial.angular_momentum_module ** 2 / self.mu
        self.momentum = self.initial.angular_momentum_module

        self.semimajor = self.calculate_semimajor()
        self.period = self.calculate_period()
        self.mean_motion = 2 * np.pi / self.period
        self.eccentric_anomaly = self.calculate_eccentric_anomaly()
        self.initial_epoch = self.calculate_initial_epoch()

    def calculate_semimajor(self):
        return self.radius_coeff * (1 / (1 - self.e ** 2))

    def calculate_period(self):
        return 2 * np.pi / np.sqrt(self.mu) * self.semimajor ** (3 / 2)

    def calculate_eccentric_anomaly(self):
        coeff = np.sqrt((1 - self.e) / (1 + self.e))
        return 2 * np.arctan(coeff * np.tan(self.initial.true_anomaly / 2))

    def calculate_initial_epoch(self):
        return (self.eccentric_anomaly - self.e * np.sin(self.eccentric_anomaly)) / self.mean_motion
    
    def state_after(self, seconds: float):
        time_final = self.initial_epoch + seconds
        periods_num = time_final / self.period
        time_position = self.period * (periods_num - int(periods_num))
        mean_anomaly = self.mean_motion * time_position

        def kepler_equation(E):
                return E - self.e * np.sin(E) - mean_anomaly
        
        eccentric_position =  optimize.root(kepler_equation, 2 * np.pi).x[0]
        coeff = np.sqrt((1 + self.e) / (1 - self.e))
        # np.mod - чтобы сделать угол от 0 до 2 * pi
        true_anomaly = np.mod(2 * np.arctan(coeff * np.tan(eccentric_position / 2)), 2 * np.pi)
        radius_peri_module = self.radius_coeff / (1 + self.e * np.cos(true_anomaly))

        radius_perifocal = radius_peri_module * np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0])
        vel_coeff = self.mu / self.initial.angular_momentum_module
        velocity_perifocal = vel_coeff * np.array([-np.sin(true_anomaly), self.e + np.cos(true_anomaly), 0])
        
        final_ascending = self.initial.right_ascention
        final_periapsis = self.initial.argument_of_perigee
        incl = self.initial.inclination

        first_tr = np.array([
            [np.cos(final_periapsis), np.sin(final_periapsis), 0],
            [-np.sin(final_periapsis), np.cos(final_periapsis), 0],
            [0, 0, 1],
        ])
        second_tr = np.array([
            [1, 0 , 0],
            [0, np.cos(incl), np.sin(incl)],
            [0, -np.sin(incl), np.cos(incl)],
        ])
        third_tr = np.array([
            [np.cos(final_ascending), np.sin(final_ascending), 0],
            [-np.sin(final_ascending), np.cos(final_ascending), 0],
            [0, 0, 1],
        ])
        transition_m = (first_tr @ second_tr @ third_tr).T
        return OrbitStateVector(transition_m @ radius_perifocal, transition_m @ velocity_perifocal)