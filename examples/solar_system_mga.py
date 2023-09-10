from typing import Any

import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Venus, Mars, Sun
from scipy.constants import gravitational_constant

from engine.lambert.custom import LambertProblem
from engine.mga import FlybyDomain, InitialDomain, ManeuversSequence, LastState
from engine.planets.solar import SolarPlanet
from engine.propagator.universal import UniversalPropagator
from engine.state_vector import StateVector
from engine.utils import Constraint, generate_rotation_matrix


# --- cost for last planet -----------------
#
# flyby_planet_state = state.arrival_planet.ephemeris_at_time(
#     state.initial_time, state.launch_time + state.flight_period
# )
# excess_velocity = end_state.velocity - flyby_planet_state.velocity
#
# v_p_hyp = np.sqrt(np.linalg.norm(excess_velocity) ** 2 + 2 *
#                   (state.arrival_planet.mass / u.kg * gravitational_constant / 10 ** 9) / (220 + 320))
#
# delta_v += (v_p_hyp - np.sqrt(
#     (state.arrival_planet.mass / u.kg * gravitational_constant / 10 ** 9) / (220 + 320))) + np.linalg.norm(
#     start_state.velocity - mid_state.velocity)

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
        SolarPlanet(Sun)
    )

    solver = UniversalPropagator(spacecraft_state_sun, SolarPlanet(Sun))  # fix to universal mu or Planet class
    mid_state = solver.state_after(state.alpha * state.flight_period)

    arrival_planet_state = state.arrival_planet.ephemeris_at_time(
        state.initial_time, state.launch_time + state.flight_period
    )
    try:
        problem = LambertProblem(
            mid_state.radius, arrival_planet_state.radius,
            (1 - state.alpha) * state.flight_period,
            SolarPlanet(Sun)
        )
        start_state, end_state = problem.solution()
    except RuntimeError:
        # some value if it fails, 1_000 delta-v is unreachable (nan + overflow errors)
        return 1_000

    escapement_delta_v = v_p_hyp - np.sqrt(dep_planet_mu / (100 + dep_planet_radius))
    delta_v = escapement_delta_v + np.linalg.norm(start_state.velocity - mid_state.velocity)

    if 'last_state_generation' in kwargs and kwargs['last_state_generation']:
        return end_state.velocity, delta_v
    return delta_v


def cost_function_flyby(domain: np.array, *args, **kwargs) -> int | tuple[Any, float | Any] | float | Any:
    """
    domain_array все параметры по которым считается, args - начальное время и планеты
    """
    state = FlybyDomain.DomainPoint.create_point_with_meta(args, domain)
    last_state: LastState = args[-1]

    flyby_planet_state = state.departure_planet.ephemeris_at_time(
        state.initial_time, last_state.total_flight_time
    )
    spacecraft_flyby_excess_velocity = last_state.velocity - flyby_planet_state.velocity

    u_p = spacecraft_flyby_excess_velocity / np.linalg.norm(spacecraft_flyby_excess_velocity)
    rotation_matrix = generate_rotation_matrix(u_p, state.gamma)

    n = np.cross(spacecraft_flyby_excess_velocity, flyby_planet_state.velocity)
    n = n / np.linalg.norm(n)

    n_trajectory_plane = rotation_matrix @ n
    mu = gravitational_constant * float(state.departure_planet.mass / u.kg) / 1000 ** 3

    betta = np.arccos(1 / (1 + state.periapsis * np.linalg.norm(spacecraft_flyby_excess_velocity) ** 2 / mu))
    turn_angle = 2 * betta

    rotation_matrix = generate_rotation_matrix(n_trajectory_plane, turn_angle)
    spacecraft_flyby_excess_velocity = rotation_matrix @ spacecraft_flyby_excess_velocity

    spacecraft_departure_velocity = spacecraft_flyby_excess_velocity + flyby_planet_state.velocity
    spacecraft_departure_state = StateVector(
        flyby_planet_state.radius,
        spacecraft_departure_velocity,
        SolarPlanet(Sun)
    )

    solver = UniversalPropagator(spacecraft_departure_state, SolarPlanet(Sun))
    mid_state = solver.state_after(state.alpha * state.flight_period)

    arrival_planet_state = state.arrival_planet.ephemeris_at_time(
        state.initial_time, last_state.total_flight_time + state.flight_period
    )
    try:
        problem = LambertProblem(
            mid_state.radius, arrival_planet_state.radius,
            (1 - state.alpha) * state.flight_period,
            SolarPlanet(Sun)
        )
        start_state, end_state = problem.solution()
    except RuntimeError:
        # some value if it fails, 1_000 delta-v is unreachable (nan + overflow errors)
        return 1_000

    # if last planet -> add braking delta-velocity
    # (v_p_hyp - v_p_hyp / np.sqrt(2))
    # ------------------------------------------------ last planet ----------------------------
    flyby_planet_state = state.arrival_planet.ephemeris_at_time(
        state.initial_time, last_state.total_flight_time + state.flight_period
    )
    excess_velocity = end_state.velocity - flyby_planet_state.velocity

    v_p_hyp = np.sqrt(np.linalg.norm(excess_velocity) ** 2 + 2 *
                      (state.arrival_planet.mass / u.kg * gravitational_constant / 10 ** 9) / (220 + 6000))

    delta_v = (v_p_hyp - np.sqrt(
        (state.arrival_planet.mass / u.kg * gravitational_constant / 10 ** 9) / (220 + 6000))) + np.linalg.norm(
        start_state.velocity - mid_state.velocity)

    # delta_v = np.linalg.norm(start_state.velocity - mid_state.velocity) + last_state.total_delta_v
    if 'last_state_generation' in kwargs and kwargs['last_state_generation']:
        return end_state.velocity, delta_v

    return delta_v


flight_period_min = 2 * 31 * 24 * 60 * 60
flight_period_max = 10 * 31 * 24 * 60 * 60

starting_domain = InitialDomain(
    Time("2023-07-30 12:00"),
    SolarPlanet(Earth),
    SolarPlanet(Venus),
    Constraint(3, 8),  # excess velocity
    Constraint(0, 24 * 31 * (24 * 60 * 60)),  # first maneuver time limit
    Constraint(0.01, 0.99),  # alpha
    Constraint(flight_period_min, flight_period_max),  # total flight time for arc
    Constraint(0, 1),  # inclination
    Constraint(0, 1),  # declination
    cost_function_initial,
)

flight_period_min = 6 * 31 * 24 * 60 * 60
flight_period_max = 20 * 31 * 24 * 60 * 60

first_flyby_domain = FlybyDomain(
    Time("2023-07-30 12:00"),
    SolarPlanet(Venus),
    SolarPlanet(Mars),
    Constraint(0, 1),
    Constraint(400 + 6_051, 2500 + 6_051),
    Constraint(0.01, 0.99),
    Constraint(flight_period_min, flight_period_max),
    cost_function_flyby,
)

if __name__ == "__main__":
    seq = ManeuversSequence([starting_domain, first_flyby_domain], 4, 10)
    result = seq.run()

    import dill

    # Save the file
    dill.dump(result, file=open("result.pickle", "wb"))
