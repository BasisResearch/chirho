from heat_over_time import HeatOverTime
import pyro
import pyro.distributions as dist
import torch
import matplotlib.pyplot as plt
from torch import tensor as tt
from torch import float64 as f64

pyro.settings.set(module_local_params=True)


N_ELEMENTS = 32
SUBSELECT_N_ELEMENTS = 3


def obs_model(temps: torch.Tensor):
    # Apply independent normal noise to each element of the obs_temp vector.
    return pyro.sample('obs_temps', dist.Normal(temps, 0.01).to_event(1))


def diffusivity_model(diffusivity: torch.Tensor, obs_time: torch.Tensor, hot: HeatOverTime):

    # Just a single execution.
    all_obs_temp = hot(*torch.atleast_2d(diffusivity, obs_time))
    subselected_elements = all_obs_temp[0, ::N_ELEMENTS // (SUBSELECT_N_ELEMENTS + 2)][1:-1]

    return obs_model(subselected_elements)


LOW_DIFFUSIVITY = 0.01
HIGH_DIFFUSIVITY = 1.


def prior():
    return pyro.sample('diffusivity', dist.Uniform(
        tt(LOW_DIFFUSIVITY, dtype=f64),
        tt(HIGH_DIFFUSIVITY, dtype=f64)
    ))


def model(obs_time: torch.Tensor, hot: HeatOverTime):

    # diffusivity = pyro.sample('diffusivity', dist.Uniform(0.01, 2.))
    # As before but float64.
    diffusivity = prior()

    return diffusivity_model(diffusivity, obs_time, hot)


def run_inference(obs_time: torch.Tensor, hot: HeatOverTime, steps=2000):

    # FIXME passing the model in here gives a "obs_temp" must be in the trace error when obs_temp is observed.
    guide = pyro.infer.autoguide.AutoNormal(prior)

    elbo = pyro.infer.Trace_ELBO()(lambda: model(obs_time, hot), guide)
    elbo()
    optim = torch.optim.Adam(elbo.parameters(), lr=1e-3)

    losses = []

    for i in range(steps):
        for param in elbo.parameters():
            param.grad = None
        optim.zero_grad()

        loss = elbo()
        losses.append(loss.clone().detach())
        loss.backward()

        if i % 500 == 0:
            print(f"Step {i}, loss {loss}")

        optim.step()

    return losses, guide


def main():
    """
    Define a very simple pyro model that infers the diffusivity of heat in a unit-length, 1D rod by observing a noisy
    measurement of the heat at a few points across the rod.
    """

    heat_over_time = HeatOverTime(n_elements=N_ELEMENTS)
    obs_time = tt(4., dtype=f64)

    # Generate data from one execution.
    true_diffusivity = tt(0.3, dtype=f64)
    obs_temps = diffusivity_model(true_diffusivity, obs_time, heat_over_time)
    true_temps = heat_over_time(*torch.atleast_2d(true_diffusivity, obs_time))[0]
    obs_locs = torch.linspace(0., 1., N_ELEMENTS)[::N_ELEMENTS // (SUBSELECT_N_ELEMENTS + 2)][1:-1]

    # Run inference.
    with pyro.condition(data={'obs_temps': obs_temps}):
        losses, guide = run_inference(obs_time, heat_over_time)

    plt.figure()
    plt.plot(losses)

    def plot_cis(diffusivities, color, alpha):
        assert diffusivities.ndim == 1

        diffusivities = diffusivities[:, None]
        tempss = heat_over_time(diffusivities, obs_time.expand(*diffusivities.shape))

        # Fill between confidence levels.
        temps_low = tempss.quantile(0.1, dim=0)
        temps_high = tempss.quantile(0.9, dim=0)

        plt.fill_between(
            torch.linspace(0., 1., N_ELEMENTS+1).numpy(),
            temps_low.detach().numpy(),
            temps_high.detach().numpy(),
            color=color,
            alpha=alpha
        )

    def plot_stuff(include_prior):

        plt.figure()

        if include_prior:
            plot_cis(tt([prior() for _ in range(500)], dtype=f64), color='blue', alpha=0.05)
            plt.plot([], [], color='blue', alpha=0.05, label='prior')

        # Plot the inferred diffusivity.
        inferred_diffusivity_med = guide.median()['diffusivity']
        inferred_temps_med = heat_over_time(*torch.atleast_2d(inferred_diffusivity_med, obs_time))[0]

        plt.plot(torch.linspace(0., 1., N_ELEMENTS+1).numpy(), inferred_temps_med.detach().numpy(),
                 label='inferred', color='orange')
        plot_cis(tt([guide()['diffusivity'] for _ in range(500)], dtype=f64), color='orange', alpha=0.3)

        plt.scatter(obs_locs.numpy(), obs_temps.detach().numpy(),
                    label='observed', color='purple', marker='x')
        plt.plot(torch.linspace(0., 1., N_ELEMENTS + 1).numpy(), true_temps.detach().numpy(),
                 label='true', color='purple')

        plt.suptitle(f"Inferring Diffusivity with Known Initial \n "
                     f"True: {true_diffusivity:.3f} \n"
                     f"Inferred: {inferred_diffusivity_med:.3f}")

        plt.ylabel("Temperature")
        plt.xlabel("Unit Length Rod")

        plt.legend()
        plt.show()

    plot_stuff(include_prior=True)


if __name__ == '__main__':
    import time
    t0 = time.time()
    main()
    print(f"Time taken: {time.time() - t0:.2f}s")
