from .utils import fillna
import torch


def t0f(r0, h0, gamma=1.):
    return (1. / (18. * gamma)) * ((7. / 4.) ** 3.) * (r0 ** 4. / h0 ** 7.)


def _halfar_ice_analytical(r, t, h0, r0, gamma=1.):
    t0 = t0f(r0, h0, gamma)

    # We assume the user's t0 == 0, but the analytical solution calculates a t0 depending on the parameters.
    # So add that value to the user's t to get the correct time.
    t = t + t0

    r = r.abs()

    hterm = (h0 * (t0 / t) ** (1. / 9.))
    rterm = (1. - ((t0 / t) ** (1. / 18.) * (r / r0)) ** (4. / 3.)) ** (3. / 7.)

    return hterm * rterm


def halfar_ice_analytical(r, t, h0, r0, gamma=1., differentiable=False):
    # TODO HACK for this to be differentiable, need to run it first to figure out where the nans lie, then
    # run it again only in the region where the nans aren't, then add that to a tensor of zeros of the original shape.
    # Unclear rn how to make this happy, but TDLR if the return vector has any nans in it, the gradients will be nans.

    res1 = _halfar_ice_analytical(r, t, h0, r0, gamma)

    if not differentiable:
        return fillna(res1)

    # Figure out where the nans aren't, and re-run those radii.
    nonnan = ~torch.isnan(res1)
    r_nonnan = r[nonnan]
    nonnan_res = _halfar_ice_analytical(r_nonnan, t, h0, r0, gamma)
    assert not torch.isnan(nonnan_res).any()

    # Create a zero-tensor of the same shape as the original, then add the nonnan_res to it in the proper place.
    ret = torch.zeros_like(res1)
    ret[nonnan] = nonnan_res

    return ret

