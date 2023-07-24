import pathlib

import agama
import astropy.units as u
import gala.potential as gp
from gala.units import galactic

this_path = pathlib.Path(__file__).absolute().parent
tex_path = this_path.parent
data_path = tex_path / "data"

agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)


gala_pot = gp.CCompositePotential(
    disk=gp.MiyamotoNagaiPotential(m=6.91e10, a=3, b=0.25, units=galactic),
    halo=gp.NFWPotential(m=5.4e11, r_s=15.0, units=galactic),
)

agama_pot = agama.Potential(
    dict(
        type="miyamotonagai",
        mass=gala_pot["disk"].parameters["m"].value,
        scaleradius=gala_pot["disk"].parameters["a"].value,
        scaleheight=gala_pot["disk"].parameters["b"].value,
    ),
    dict(
        type="nfw",
        mass=gala_pot["halo"].parameters["m"].value,
        scaleradius=gala_pot["halo"].parameters["r_s"].value,
    ),
)

R0 = 8.275 * u.kpc
