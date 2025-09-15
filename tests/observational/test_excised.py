import pytest
import torch

from chirho.observational.ops import ExcisedCategorical, ExcisedNormal


# needed for testing interval CDFs
class ECDF(torch.nn.Module):
    def __init__(self, x, side="right"):
        super(ECDF, self).__init__()

        if side.lower() not in ["right", "left"]:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        if len(x.shape) != 1:
            msg = "x must be 1-dimensional"
            raise ValueError(msg)

        x = x.sort()[0]
        nobs = len(x)
        y = torch.linspace(1.0 / nobs, 1, nobs, device=x.device)

        self.x = torch.cat((torch.tensor([-torch.inf], device=x.device), x))
        self.y = torch.cat((torch.tensor([0], device=y.device), y))
        self.n = self.x.shape[0]

    def forward(self, time):
        tind = torch.searchsorted(self.x, time, side=self.side) - 1
        return self.y[tind]


@pytest.fixture
def true_parameters():
    mean = torch.tensor([[[1.0]], [[3.0]]], requires_grad=True)
    stddev = torch.tensor([[[2.0]], [[3.0]]], requires_grad=True)
    return mean, stddev


@pytest.mark.parametrize(
    "interval_key, intervals",
    [
        ("flat", [(torch.tensor(-1.0), torch.tensor(1.0))]),
        ("flat_intervals", [(-2.0, -1.0), (1.0, 2.0)]),
        ("shaped", None),  # will fill in fixture
        ("shaped_intervals", None),
    ],
)
def test_excised_normal_shapes_and_sampling(true_parameters, interval_key, intervals):
    mean, stddev = true_parameters

    shaped_intervals = [(mean - stddev, mean + stddev)]
    shaped_intervals_multiple = [
        (mean - 2 * stddev, mean - stddev),
        (mean + stddev, mean + 2 * stddev),
    ]

    if interval_key == "shaped":
        intervals = shaped_intervals
    elif interval_key == "shaped_intervals":
        intervals = shaped_intervals_multiple

    excised_normal = ExcisedNormal(mean, stddev, intervals)

    new_batch_shape = (2, 1, 3)
    excised_normal_expanded = excised_normal.expand(new_batch_shape)

    # --- Basic shape checks ---
    for dist_obj in [excised_normal, excised_normal_expanded]:
        assert dist_obj.mean.shape == dist_obj.loc.shape
        assert dist_obj.stddev.shape == dist_obj.scale.shape
        assert dist_obj.variance.shape == dist_obj.scale.shape
        for low, high in dist_obj._intervals:
            assert low.shape == dist_obj.loc.shape
            assert high.shape == dist_obj.loc.shape

    # --- Interval masses as expected ---
    if interval_key == "shaped":
        for (
            lcdf
        ) in excised_normal._lcdfs:  # these are expected to be *base* normal lcdfs
            assert torch.allclose(
                lcdf, torch.tensor(0.1587), atol=1e-4
            )  # below mean - 1 stddev
        for im in excised_normal._interval_masses:
            assert torch.allclose(
                im, torch.tensor(0.6827), atol=1e-4
            )  # between mean - 1 stddev and mean + 1 stddev

    # --- Sampling ---
    sample = excised_normal.sample(sample_shape=(400,))
    sample_expanded = excised_normal_expanded.sample(sample_shape=(400,))

    assert sample.shape == (400, 2, 1, 1)
    assert sample_expanded.shape == (400, 2, 1, 3)

    # --- samples avoid intervals ---
    for low, high in excised_normal._intervals:
        assert torch.all((sample <= low) | (sample >= high))
    for low, high in excised_normal_expanded._intervals:
        assert torch.all((sample_expanded <= low) | (sample_expanded >= high))

    # --- Log probability checks ---
    candidates = (torch.rand(sample.shape) - 0.5) * 40
    log_probs = excised_normal.log_prob(candidates)
    mask = torch.zeros_like(candidates, dtype=torch.bool)
    for low, high in excised_normal._intervals:
        mask |= (candidates >= low) & (candidates <= high)
    assert torch.all(torch.where(mask, log_probs == -float("inf"), True))

    # --- CDF vs ECDF  ---
    ecdf_sample_module = ECDF(sample[:, 0, 0, 0])
    ecdf_sample = ecdf_sample_module(sample[:, 0, 0, 0])
    cdf_sample = excised_normal.cdf(sample)[:, 0, 0, 0].detach()
    assert torch.allclose(ecdf_sample, cdf_sample, atol=0.1)

    ecdf_sample_exp_module = ECDF(sample_expanded[:, 0, 0, 0])
    ecdf_sample_exp = ecdf_sample_exp_module(sample_expanded[:, 0, 0, 0])
    cdf_sample_exp = excised_normal_expanded.cdf(sample_expanded)[:, 0, 0, 0].detach()
    assert torch.allclose(ecdf_sample_exp, cdf_sample_exp, atol=0.1)

    # --- rsample and backpropagation ---
    cloned_mean = mean.detach().clone().requires_grad_(True)
    cloned_stddev = stddev.detach().clone().requires_grad_(True)
    new_interval = [(torch.tensor(-1.0), torch.tensor(1.0))]
    excised_normal_copy = ExcisedNormal(cloned_mean, cloned_stddev, new_interval)
    r_sample = excised_normal_copy.rsample(sample_shape=(400,))
    loss = ((r_sample - 1.0) ** 2).mean()
    optimizer = torch.optim.SGD(
        [excised_normal_copy.loc, excised_normal_copy.scale], lr=0.1
    )
    loc_before, scale_before = (
        excised_normal_copy.loc.clone(),
        excised_normal_copy.scale.clone(),
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loc_after, scale_after = (
        excised_normal_copy.loc.clone(),
        excised_normal_copy.scale.clone(),
    )
    assert not torch.allclose(loc_before, loc_after)
    assert not torch.allclose(scale_before, scale_after)


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([0.1, 1.0, 2.0, 3.0]),  # shape (4,)
        torch.tensor([[0.1, 1.0, 2.0, 3.0]]),  # shape (1, 4)
        torch.tensor([[0.1, 1.0, 2.0, 3.0], [0.3, 0.2, 0.1, 0.4]]),  # shape (2, 4)
        torch.tensor(
            [[[0.1, 1.0, 2.0, 3.0]], [[0.3, 0.2, 0.1, 0.4]]]
        ),  # shape (2, 1, 4)
    ],
)
@pytest.mark.parametrize(
    "intervals",
    [
        [(torch.tensor(1), torch.tensor(2))],  # excise categories 1 and 2
        [(torch.tensor(0), torch.tensor(0))],  # excise first category
        [(torch.tensor(3), torch.tensor(3))],  # excise last category
        [
            (torch.tensor(0), torch.tensor(1)),
            (torch.tensor(2), torch.tensor(2)),
        ],  # multiple disjoint
        [],  # no excision
    ],
)
def test_excised_categorical_shapes_and_probs(logits, intervals):
    excised = ExcisedCategorical(logits=logits, intervals=intervals)

    # --- Basic shape checks ---
    assert excised.probs.shape == logits.shape
    assert excised.logits.shape == logits.shape

    # --- Excised categories have zero prob ---
    for low, high in intervals:
        low_i = int(torch.clamp(torch.ceil(low), 0, logits.size(-1) - 1))
        high_i = int(torch.clamp(torch.floor(high), 0, logits.size(-1) - 1))
        for i in range(low_i, high_i + 1):
            assert torch.all(excised.probs[..., i] == 0.0)

    # --- Remaining probs renormalize ---
    assert torch.allclose(
        excised.probs.sum(-1), torch.tensor(1.0).expand_as(excised.probs.sum(-1))
    )

    # --- Sampling avoids excised categories ---
    samples = excised.sample((5000,))
    for low, high in intervals:
        assert not torch.any((samples >= low) & (samples <= high))

    # --- Log prob checks ---
    num_categories = logits.size(-1)
    for i in range(num_categories):
        lp = excised.log_prob(torch.tensor(i))
        if any((low <= i) & (i <= high) for (low, high) in intervals):
            assert torch.all(lp == -float("inf"))
        else:
            assert torch.all(lp > -float("inf"))


def test_excised_categorical_empirical_frequencies():
    logits = torch.tensor([0.1, 1.0, 2.0, 3.0])
    intervals = [(torch.tensor(1), torch.tensor(2))]  # drop categories 1 and 2
    excised = ExcisedCategorical(logits=logits, intervals=intervals)

    N = 20000
    samples = excised.sample((N,))
    freqs = torch.bincount(samples, minlength=logits.size(-1)) / N

    # compare only on non-excised categories
    mask = excised.probs > 0
    assert torch.allclose(freqs[mask], excised.probs[mask], atol=0.02)
    # excised categories have zero empirical freq
    assert torch.allclose(freqs[~mask], torch.zeros_like(freqs[~mask]), atol=1e-3)
