from .utils import fillna


def t0f(r0, h0, gamma=1.):
    return (1. / (18. * gamma)) * ((7. / 4.) ** 3.) * (r0 ** 4. / h0 ** 7.)


def halfar_ice_analytical(r, t, h0, r0, gamma=1.):
    t0 = t0f(r0, h0, gamma)

    if t.min() < t0:
        raise ValueError(f"Behavior undefined when querying points before t0."
                         f"t0={t0}, t={t.min()}")

    r = r.abs()

    hterm = (h0 * (t0 / t) ** (1. / 9.))
    rterm = (1. - ((t0 / t) ** (1. / 18.) * (r / r0)) ** (4. / 3.)) ** (3. / 7.)

    return fillna(hterm * rterm)
