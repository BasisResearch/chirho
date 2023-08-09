import pyro
from typing import Callable, List, Tuple, Type, Optional
import torch
from pyro.infer.autoguide import AutoGuide
from ..typedecs import ModelType, KWType
from ..composeable_expectation.composed_expectation import ComposedExpectation
from ..composeable_expectation.expectation_atom import ExpectationAtom


class _GuideRegistrationMixin:
    def __init__(self):
        self.pseudo_densities = dict()
        self.guides = dict()
        self.registered_model = None

    def optimize_guides(self, lr: float, n_steps: int,
                        adjust_grads_: Callable[[torch.nn.Parameter, ...], None] = None,
                        callback: Optional[Callable[[str, int], None]] = None,
                        start_scale=1.0):
        if not len(self.keys()):
            raise ValueError("No guides registered. Did you call "
                             f"{_GuideRegistrationMixin.__name__}.register_guides?")

        for k in self.keys():
            pseudo_density = self.pseudo_densities[k]
            guide = self.guides[k]

            current_scale = start_scale

            def scaling_pseudo_density() -> KWType:
                return pyro.poutine.scale(pseudo_density, scale=current_scale)()

            # noinspection PyTypeChecker
            elbo = pyro.infer.Trace_ELBO()(scaling_pseudo_density, guide)
            elbo()  # Call to surface parameters for optimizer.
            # optim = torch.optim.ASGD(elbo.parameters(), lr=lr)
            optim = torch.optim.Adam(elbo.parameters(), lr=lr)

            for i in range(n_steps):

                # Scale from start_scale up to 1.
                current_scale = start_scale + (1. - start_scale) * (i / n_steps)

                for param in elbo.parameters():
                    param.grad = None
                optim.zero_grad()

                loss = elbo()
                loss.backward()

                if adjust_grads_ is not None:
                    adjust_grads_(*tuple(elbo.parameters()))

                if callback is not None:
                    callback(k, i)

                optim.step()

    def register_guides(self, ce: ComposedExpectation, model: ModelType,
                        auto_guide: Optional[Type[AutoGuide]], auto_guide_kwargs=None,
                        allow_repeated_names=False):
        self.clear_guides()

        if auto_guide_kwargs is None:
            auto_guide_kwargs = dict()
        else:
            if auto_guide is None:
                raise ValueError("auto_guide_kwargs provided but no auto_guide class provided. Did you mean to "
                                 "provide an auto_guide class?")

        for part in ce.parts:
            pseudo_density = part.build_pseudo_density(model)
            if part.guide is not None:
                guide = part.guide

                if tuple(guide().keys()) != tuple(model().keys()):
                    raise ValueError("A preset guide must return the same variables as the model, but got "
                                     f"{tuple(guide().keys())} and {tuple(model().keys())} instead.")

            else:
                if auto_guide is None:
                    raise ValueError("No guide preregistered and no no auto guide class provided.")
                guide = auto_guide(pseudo_density, **auto_guide_kwargs)

            if not allow_repeated_names:
                if part.name in self.pseudo_densities:
                    raise ValueError(f"Repeated part name {part.name}.")
                if part.name in self.guides:
                    raise ValueError(f"Repeated part name {part.name}.")

            if part.name not in self.pseudo_densities:
                self.pseudo_densities[part.name] = pseudo_density

            if part.name not in self.guides:
                self.guides[part.name] = guide

        self.registered_model = model

    def keys(self) -> frozenset:
        assert tuple(self.pseudo_densities.keys()) == tuple(self.guides.keys()), "Should not be possible b/c these" \
                                                                                 " are added to at the same time."
        return frozenset(self.pseudo_densities.keys())

    def clear_guides(self):
        self.pseudo_densities = dict()
        self.guides = dict()
        self.registered_model = None

    def _get_pq(self, ea: "ExpectationAtom", p: ModelType) -> Tuple[ModelType, ModelType]:
        try:
            q: ModelType = self.guides[ea.name]
        except KeyError:
            raise KeyError(f"No guide registered for {ea.name}. "
                           f"Did you call {_GuideRegistrationMixin.__name__}.register_guides?")

        if p is not self.registered_model:
            raise ValueError("The probability distribution registered with the guides does not match the "
                             "probability distribution called to compute the expectation. In other words,"
                             f"the same p must be used in {_GuideRegistrationMixin.__name__}"
                             f".register_guides and {ComposedExpectation.__name__}.__call__.")

        return p, q

    def plot_guide_pseudo_likelihood(
            self, rv_name: str, guide_kde_kwargs, pseudo_density_plot_kwargs, keys: List[str] = None):
        # TODO move this to a separate class and inherit or something, just so plotting code doesn't clutter
        #  up functional code.
        import seaborn as sns
        import matplotlib.pyplot as plt

        if not len(self.keys()):
            raise ValueError("No guides registered. Did you call "
                             f"{_GuideRegistrationMixin.__name__}.register_guides?")

        figs = []

        if keys is None:
            keys = self.keys()

        for k in keys:
            pseudo_density = self.pseudo_densities[k]
            guide = self.guides[k]

            if self.registered_model()[rv_name].ndim != 0:
                raise ValueError("Can only plot pseudo likelihood/guide comparisons for univariates.")

            fig, ax = plt.subplots(1, 1)
            sns.kdeplot([guide()[rv_name].item() for _ in range(1000)], label="guide", **guide_kde_kwargs)

            tax = ax.twinx()

            model_samples = torch.tensor([self.registered_model()[rv_name] for _ in range(1000)])
            xx = torch.linspace(model_samples.min(), model_samples.max(), 1000).detach()

            lps = []
            for x in xx:
                cm = pyro.poutine.condition(pseudo_density, data={rv_name: x})
                lp = pyro.poutine.trace(cm).get_trace().log_prob_sum()
                lps.append(lp)
            lps = torch.tensor(lps).exp().detach().numpy()

            # This will be squiggly if there are other latents. TODO smooth?
            tax.plot(xx, lps, label="pseudo-density", **pseudo_density_plot_kwargs)

            ax.set_title(f"q and pseudo-p for {rv_name} and part {k}")

            # Add single legend for both.
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = tax.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc=0)

            figs.append(fig)

        return figs
