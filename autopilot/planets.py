from astropy.constants import Constant
from poliastro.bodies import Body

GM_kerbin = Constant(
    "GM_sun",
    "kerbin gravitational constant",
    3.5316000 * 10 ** 12,
    "m3 / (s2)",
    0.0000000001e20,
    "IAU 2009 system of astronomical constants",
    system="si",
)

R_kerbin = Constant(
    "R_kerbin",
    "kerbol equatorial radius",
    600_000,
    "m",
    0,
    "IAU Working Group on Cartographic Coordinates and Rotational Elements: 2015",
    system="si",
)

Kerbin = Body(
    parent=None,
    k=GM_kerbin,
    name="Kerbin",
    symbol="\u2609",
    R=R_kerbin,
)
