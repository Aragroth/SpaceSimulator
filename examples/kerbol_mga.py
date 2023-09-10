from __future__ import annotations

from engine.mga.initial_domain import InitialDomain
from engine.mga.sequence import ManeuversSequence
from engine.planets.ksp import KspPlanet, Kerbol
from engine.utils import Constraint

mu = 398_600
r0 = 600 + 100
mu_sun = 1.327 * (10 ** 11)

mu = 3.5316000 * 10 ** 12 / 10 ** 9
mu_sun = 1.1723328e18 / 10 ** 9  # kerbol
Sun = Kerbol

flight_period_min = 2 * 31 * 6 * 60 * 60
flight_period_max = 9 * 31 * 6 * 60 * 60

starting_domain = InitialDomain(
    0,
    KspPlanet("Kerbin"),
    KspPlanet("Duna"),
    Constraint(0.7, 2.4),  # excess velocity
    Constraint(6 * (6 * 60 * 60), 12 * 31 * (6 * 60 * 60)),  # first maneuver time limit
    Constraint(0.01, 0.99),  # alpha
    Constraint(flight_period_min, flight_period_max),  # total flight time for arc
    Constraint(0, 1),  # inclination
    Constraint(0, 1),  # declination
)

# flight_period_min = 1 * 31 * 6 * 60 * 60
# flight_period_max = 12 * 31 * 6 * 60 * 60
#
# starting_domain = InitialDomain(
#     0,
#     KspPlanet("Kerbin"),
#     KspPlanet("Duna"),
#     Constraint(0.1, 2),  # excess velocity
#     Constraint(0, 12 * 31 * (6 * 60 * 60)),  # first maneuver time limit
#     Constraint(0.01, 0.99),  # alpha
#     Constraint(flight_period_min, flight_period_max),  # total flight time for arc
#     Constraint(0, 1),  # inclination
#     Constraint(0, 1),  # declination
# )
#

# flight_period_min = 1 * 31 * 24 * 60 * 60
# flight_period_max = 5 * 12 * 31 * 24 * 60 * 60
# first_flyby_domain = FlybyDomain(
#     0,
#     KspPlanet("Duna"),
#     KspPlanet("Jool"),
#     Constraint(0, 1),
#     Constraint(52 + 320, 3000 + 320),
#     Constraint(0.01, 0.99),
#     Constraint(flight_period_min, flight_period_max),
# )

if __name__ == "__main__":
    seq = ManeuversSequence([starting_domain])  # , first_flyby_domain
    result = seq.run()

    point = min(result, key=lambda x: x.total_delta_v)

    print(point.points_sequence[0], point.total_delta_v)

    import dill

    # Save the file
    dill.dump([i.to_dict() for i in result], file=open("ksp_to_duna.pickle", "wb"))

    # Reload the file
    # bounds = [
    #     (v_start_min, v_start_max),
    #     (t_launch_min, t_launch_max),
    #     (0.01, 0.99),
    #     (flight_period_min, flight_period_max),
    #     (0, 1),
    #     (0, 1),
    # ]

    # points = initial_domain(0, 8 * 31 * 24 * 60 * 60, 2, 5, Mars)
    # print(
    #     len(points),
    #     min(points, key=lambda x: f_velocity_cost(x)),
    #     f_velocity_cost(min(points, key=lambda x: f_velocity_cost(x)))
    # )
    #
    # with open('test.npy', 'wb') as f:
    #     np.save(f, points)
