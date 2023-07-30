from astropy.time import Time
from poliastro.bodies import Earth

from engine.mga import InitialDomain, Planet

flight_period_min = 2 * 31 * 24 * 60 * 60
flight_period_max = 10 * 31 * 24 * 60 * 60

starting_domain = InitialDomain(
    Time("2023-07-30 12:00"),
    Planet(Earth),
    Planet(Venus),
    Constraint(3, 8),  # excess velocity
    Constraint(0, 24 * 31 * (24 * 60 * 60)),  # first maneuver time limit
    Constraint(0.01, 0.99),  # alpha
    Constraint(flight_period_min, flight_period_max),  # total flight time for arc
    Constraint(0, 1),  # inclination
    Constraint(0, 1),  # declination
)

flight_period_min = 6 * 31 * 24 * 60 * 60
flight_period_max = 20 * 31 * 24 * 60 * 60

first_flyby_domain = FlybyDomain(
    Time("2023-07-30 12:00"),
    Planet(Venus),
    Planet(Mars),
    Constraint(0, 1),
    Constraint(400 + 6_051, 2500 + 6_051),
    Constraint(0.01, 0.99),
    Constraint(flight_period_min, flight_period_max),
)

if __name__ == "__main__":
    seq = ManeuversSequence([starting_domain, first_flyby_domain])
    result = seq.run()

    import dill

    # Save the file
    dill.dump(result, file=open("result.pickle", "wb"))
