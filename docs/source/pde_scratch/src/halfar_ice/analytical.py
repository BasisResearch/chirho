from .utils import fillna


def t0f(r0, h0, gamma=1.):
    return (1. / (18. * gamma)) * ((7. / 4.) ** 3.) * (r0 ** 4. / h0 ** 7.)


def halfar_ice_analytical(r, t, h0, r0, gamma=1.):
    t0 = t0f(r0, h0, gamma)

    # We assume the user's t0 == 0, but the analytical solution calculates a t0 depending on the parameters.
    # So add that value to the user's t to get the correct time.
    t = t + t0

    r = r.abs()

    hterm = (h0 * (t0 / t) ** (1. / 9.))
    rterm = (1. - ((t0 / t) ** (1. / 18.) * (r / r0)) ** (4. / 3.)) ** (3. / 7.)

    return fillna(hterm * rterm)
