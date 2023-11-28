from typing import Callable
from ..typedecs import KWType
import torch
from torch import Tensor as TT
import warnings
from .expectation_atom import ExpectationAtom
from .composed_expectation import ComposedExpectation


def _build_df_dd(dparams: TT, di: int, part: ExpectationAtom) -> Callable[[KWType], TT]:
    def df_dd(stochastics: KWType) -> TT:
        y: TT = part.f(stochastics)

        if y.ndim != 0:
            raise ValueError(f"Argument f to {ExpectationAtom.__name__} with name {part.name} must return a scalar,"
                             f" but got {y} instead.")

        assert dparams[di].ndim == 0, "This shouldn't be possible due to the outer check of 1 dimension."

        try:
            df_ddparam, = torch.autograd.grad(
                outputs=(y,),
                # FIXME HACK Have to grad wrt the whole tensor apparently, and then index after.
                inputs=(dparams,),
                create_graph=True
            )
            df_ddparam = df_ddparam[di]
        except RuntimeError as e:
            if "does not require grad and does not have a grad_fn" in str(e):
                # FIXME FIXME kf2801dgi1 this is only correct when this particular atom is a mul or div of one
                #  that does require grad. It would be nice to not have to repro autodiff here but
                #  somehow this needs to how the parent incorporates this. Maybe we could autodiff
                #  parent's op and see how this atom relates, then use that to determine what
                #  should be returned here?
                warnings.warn(f"The gradient of atom named {part.name} with respect to dparam {di}"
                              f" is 0.0, but returning the original atom's value for now because"
                              f" it's probably a scaling factor. This is a hack and should be fixed."
                              f" See FIXME tagged kf2801dgi1.")
                return y
            else:
                raise

        assert df_ddparam.ndim == 0, "This shouldn't be possible due to out and in being 0 dimensional."

        return df_ddparam

    return df_dd


# FIXME 7301ykd0sk See below. Want to conver to proper in place operation with no return value.
def gradify_in_place_but_need_return(
        output: ComposedExpectation, dparams: TT, split_atoms=False) -> ComposedExpectation:

    if dparams.ndim != 1:
        raise ValueError(f"Argument dparams to {gradify_in_place_but_need_return.__name__} must be a 1d tensor, "
                         f"but got ndim {dparams.ndim} instead.")

    assert len(output.parts) >= 1, "This shouldn't be possible due to composites always having at least one " \
                                   "part (themselves)."

    if not len(dparams) >= 1:
        raise ValueError(f"Argument dparams to {gradify_in_place_but_need_return.__name__} must have at least one "
                         f"element, but got {len(dparams)} instead.")

    # Only relevant if output is an atom. Just defining outside of loop so type checking is happy below.
    sub_atom_composite = None

    for part in output.parts:

        sub_atoms = []

        # Create a new atom for each of the old atoms.
        for di, _ in enumerate(dparams):

            # Create a new atom just for just this element of the gradient vector.
            ea = ExpectationAtom(
                f=_build_df_dd(dparams, di, part),
                name=f"d{part.name}_dd{di}",
                log_fac_eps=part.log_fac_eps
                # TODO maybe seed a new guide with the original guide (if present)?
            )

            if split_atoms:
                ea = ea.split_into_positive_components()

            sub_atoms.append(ea)

        # Create a composite that simply concatenates the new atoms into one tensor.
        sub_atom_composite = ComposedExpectation(
            children=sub_atoms,
            op=lambda *v: torch.stack(v, dim=0),
            # Note bm72gdi1: This will be updated before the return of this function.
            parts=[]
        )

        for parent in part.parents:
            positions_as_child = [i for i, child in enumerate(parent.children) if child is part]
            assert len(positions_as_child) >= 1, "This shouldn't be possible." \
                                                 " There's a reference mismatch with parents."
            # Now, swap out the old atom with the new composite.
            for pac in positions_as_child:
                parent.children[pac] = sub_atom_composite
                sub_atom_composite.parents.append(parent)

    if isinstance(output, ExpectationAtom):
        assert output.parts[0] and len(output.parts) == 1, "This shouldn't be possible: atom code broken?"
        # FIXME 7301ykd0sk this is why you have to take the return value here...
        output = sub_atom_composite

    # Note bm72gdi1 this ensures that the cached part list is up-to-date.
    output.recursively_refresh_parts()

    return output
