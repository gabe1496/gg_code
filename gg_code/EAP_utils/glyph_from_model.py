import numpy as np
from dipy.core.geometry import sphere2cart, cart2sphere


def compute_pdf(model, data, sphere, radius):
    model_fit = model.fit(data)

    vertices = sphere.vertices
    r, theta, phi = cart2sphere(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    x, y, z = sphere2cart(radius, theta, phi)

    vertices_new = np.vstack((x, y, z)).T
    pdf = model_fit.pdf(vertices_new)

    return pdf


def compute_odf_mapmri(model, data, sphere, s=2):
    model_fit = model.fit(data)

    odf = model_fit.odf(sphere, s)
    return odf


def compute_odf_shore(model, data, sphere):
    model_fit = model.fit(data)

    odf = model_fit.odf(sphere)
    return odf
