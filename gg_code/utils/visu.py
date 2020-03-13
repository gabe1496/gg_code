import numpy as np
from fury import window, actor
from dipy.core.sphere import Sphere



WINDOW_SIZE = (600, 600)

def my_visu(sf, sphere, rot=True, norm=True, scale=True, title="Modeling"):
    ren = window.renderer(background=window.colors.white)

    sf_actor = actor.odf_slicer(sf, sphere=sphere, colormap='jet', scale = 0.4,
                                norm=norm, radial_scale=scale)
    if rot :
        sf_actor.RotateX(90)
    ren.add(sf_actor)
    window.show(ren, title=title, size=WINDOW_SIZE)

    ren.rm(sf_actor)
    window.rm_all(ren)


def visu_odf(bvec, odf_sf):
    sph_gtab = Sphere(xyz=np.vstack(bvec))
    my_visu(odf_sf, sph_gtab, rot=False, norm=True)
