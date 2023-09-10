from __future__ import annotations

import numpy as np

from engine.lambert.custom import LambertProblem
from engine.mga import InitialDomain, ManeuversSequence
from engine.planets import Kerbin, Duna, Kerbol
from engine.planets.ksp import KspPlanet
from engine.planets.solar import SolarPlanet
from engine.propagator.universal import UniversalPropagator
from engine.state_vector import StateVector
from engine.utils import Constraint


def cost_function_initial(domain: np.array, *args, **kwargs) -> float | tuple:
    """
    domain_array все параметры по которым считается, args - начальное время и планеты
    """
    state = InitialDomain.DomainPoint.create_point_with_meta(args, domain)

    excess_velocity = state.v_start * np.array([
        np.cos(state.incl) * np.cos(state.decl),
        np.sin(state.incl) * np.cos(state.decl),
        np.sin(state.decl)
    ])

    dep_planet_mu = state.departure_planet.mu
    dep_planet_radius = state.departure_planet.radius

    v_p_hyp = np.sqrt(np.linalg.norm(excess_velocity) ** 2 + 2 * dep_planet_mu / (100 + dep_planet_radius))

    departure_planet_state = state.departure_planet.ephemeris_at_time(state.initial_time, state.launch_time)
    spacecraft_state_sun = StateVector(
        departure_planet_state.radius,
        excess_velocity + departure_planet_state.velocity,
        SolarPlanet(Kerbol)
    )

    solver = UniversalPropagator(spacecraft_state_sun, SolarPlanet(Kerbol))  # fix to universal mu or Planet class
    mid_state = solver.state_after(state.alpha * state.flight_period)

    arrival_planet_state = state.arrival_planet.ephemeris_at_time(
        state.initial_time, state.launch_time + state.flight_period
    )
    try:
        problem = LambertProblem(
            mid_state.radius, arrival_planet_state.radius,
            (1 - state.alpha) * state.flight_period,
            SolarPlanet(Kerbol)
        )
        start_state, end_state = problem.solution()
    except RuntimeError:
        # some value if it fails, 1_000 delta-v is unreachable (nan + overflow errors)
        return 1_000

    escapement_delta_v = v_p_hyp - np.sqrt(dep_planet_mu / (100 + dep_planet_radius))
    delta_v = escapement_delta_v + np.linalg.norm(start_state.velocity - mid_state.velocity)

    flyby_planet_state = state.arrival_planet.ephemeris_at_time(
        state.initial_time, state.launch_time + state.flight_period
    )
    excess_velocity = end_state.velocity - flyby_planet_state.velocity

    v_p_hyp = np.sqrt(np.linalg.norm(excess_velocity) ** 2 + 2 * state.arrival_planet.mu / (220 + 320))

    delta_v += ((v_p_hyp - np.sqrt(state.arrival_planet.mu / (220 + 320))) +
                np.linalg.norm(start_state.velocity - mid_state.velocity))

    if 'last_state_generation' in kwargs and kwargs['last_state_generation']:
        return end_state.velocity, delta_v

    return delta_v


flight_period_min = 2 * 31 * 6 * 60 * 60
flight_period_max = 9 * 31 * 6 * 60 * 60

starting_domain = InitialDomain(
    0,
    KspPlanet(Kerbin),
    KspPlanet(Duna),
    Constraint(0.1, 1),  # excess velocity
    Constraint(6 * (6 * 60 * 60), 12 * 31 * (6 * 60 * 60)),  # first maneuver time limit
    Constraint(0.01, 0.99),  # alpha
    Constraint(flight_period_min, flight_period_max),  # total flight time for arc
    Constraint(0, 1),  # inclination
    Constraint(0, 1),  # declination
    cost_function_initial
)

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

import multiprocessing


if __name__ == "__main__":
    seq = ManeuversSequence([starting_domain], 2, 10)  # , first_flyby_domain
    result = seq.run()

    point = min(result, key=lambda x: x.total_delta_v)

    print(point.points_sequence[0], point.total_delta_v)

    import dill

    # Save the file
    # TODO use to_json()
    dill.dump([i.to_dict() for i in result], file=open("data_simulations/ksp_to_duna.pickle", "wb"))

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
