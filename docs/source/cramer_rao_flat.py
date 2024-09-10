from typing import Any, Callable, TypeVar, ParamSpec, List
import torch
import pyro
from tqdm import tqdm
import pyro.distributions as dist


P = ParamSpec("P")
T = TypeVar("T")



# def compute_crb(target: Functional[P]) -> Functional[P]:
#     def _crb_functional(model: Model[P]) -> Callable[P, torch.Tensor]:
#         def _crb_function(*args: P.args, **kwargs: P.kwargs) -> torch.Tensor:
#             ...  # TODO implement me
#         return _crb_function
    
#     return _crb_functional

from chirho.robust.internals.nmc import BatchedNMCLogMarginalLikelihood
from chirho.robust.internals.linearize import make_empirical_fisher_vp, conjugate_gradient_solve

from chirho.robust.internals.utils import (
    ParamDict,
    make_flatten_unflatten,
    make_functional_call,
    reset_rng_state,
)
from chirho.robust.ops import Point

Model = Callable[P, ParamDict]
Functional = Callable[[Model[P]], Callable[P, torch.Tensor]]

# class SinusoidalModel(pyro.nn.PyroModule):
#     def __init__(self, f0, N=10, snr=1.0):
#         super().__init__()
#         self.f0 = torch.nn.Parameter(torch.tensor(f0))
#         self.n_vec = torch.arange(N)
#         self.N = N
#         self.sigma = 1.0/torch.sqrt(torch.tensor(snr))

#     def forward(self):
#         s = pyro.sample("s", dist.Normal(torch.cos(2*torch.pi * self.f0 * self.n_vec), self.sigma).to_event(1))
#         return {'f0': self.f0}

class SinusoidalModel(pyro.nn.PyroModule):
    def __init__(self, f0, N=10, snr=1.0):
        super().__init__()
        self.f0 = torch.nn.Parameter(torch.tensor([f0]))
        self.n_vec = torch.arange(N)
        self.N = N
        self.sigma = 1.0/torch.sqrt(torch.tensor(snr))

    def forward(self):
        s = pyro.sample("s", dist.Normal(torch.cos(2*torch.pi * self.f0 * self.n_vec), self.sigma).to_event(1)) 
        return s
    
class TrivialFunctional(torch.nn.Module):
    def __init__(self, model : Model[P]):
        super().__init__()
        self.model = model
        
    def forward(self):
        return self.model.f0

    
def do_the_thing(
        model: torch.nn.Module,
        functional: Callable[[torch.nn.Module], Callable[..., torch.Tensor]], 
        num_samples_outer: int = 50000,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> torch.Tensor:
    
    assert isinstance(model, torch.nn.Module)

    with torch.no_grad():
        data: Point[T] = pyro.infer.Predictive(model, num_samples=num_samples_outer, parallel=True)() # (*args, **kwargs)

    ######## Option 1 ########            
    # def target_func(params: Any, *args, **kwargs) -> torch.Tensor:
    #     return functional(lambda *args, **kwargs: model_func(params, *args, **kwargs))(*args, **kwargs)
    
    # def batched_func_log_prob(params, *args, **kwargs):
    #     return BatchedNMCLogMarginalLikelihood(
    #         lambda *args, **kwargs: model_func(params, *args, **kwargs),
    #         num_samples=1,
    #         max_plate_nesting=None
    #     )(*args, **kwargs)
    
    # fvp = make_empirical_fisher_vp(
    #     batched_func_log_prob, model_params, data
    # ) # (*args, **kwargs)
    
    # target_grad = torch.func.grad(target_func)(model_params, *args, **kwargs)

    ######## Option 2 ########
    # model_params, model_func = make_functional_call(model)

    # def target_func(params: Any, *args, **kwargs) -> torch.Tensor:
    #     for p in params:
    #         model[p.name] = p.value
    #     return lambda *args, **kwargs : functional(model)(*args, **kwargs)
        
    batched_log_prob = BatchedNMCLogMarginalLikelihood(
        model, num_samples=1, max_plate_nesting=None
    )
    log_prob_params, batched_func_log_prob = make_functional_call(batched_log_prob)

    fvp = make_empirical_fisher_vp(
        batched_func_log_prob, log_prob_params, data, *args, **kwargs
    )
    pinned_fvp = reset_rng_state(pyro.util.get_rng_state())(fvp)
    pinned_fvp_batched = torch.func.vmap(
        lambda v: pinned_fvp(v), randomness="different"
    )
    # target_grad = torch.func.grad(target_func)(model_params, *args, **kwargs)
    target_grad = {'model.f0': torch.tensor([1.0])}

    inverse_fisher_target_product = conjugate_gradient_solve(pinned_fvp_batched, target_grad) # nested dict
    # inverse_fisher_target_product = conjugate_gradient_solve(pinned_fvp, target_grad) # nested dict

    flat_target_grad: torch.Tensor = torch.tensor([1.0]) # torch.flatten(target_grad)
    flat_inverse_fisher_target_product: torch.Tensor = torch.flatten(inverse_fisher_target_product['model.f0'])
    return torch.einsum("s,t->st", flat_target_grad, flat_inverse_fisher_target_product)    

# the analytic CRB
def CRLB_analytic(model,f0):
    return model.sigma**2/(
        (2*torch.pi*model.n_vec.expand(f0.shape[0],-1)* \
            torch.sin(2*torch.pi*f0.unsqueeze(-1)*model.n_vec.unsqueeze(-2)
                    )
            )**2
            ).sum(axis=1)
    
if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    
    f0_vec = torch.linspace(0.05,0.45,1000)
    
    N = 10
    f0 = 0.3
    sigma = 2.0
    model = SinusoidalModel(f0=f0,N=N,snr=1.0/sigma**2)

    res_vec = []

    for f0 in tqdm(f0_vec):
        model.f0.data = f0
        res = do_the_thing(model, TrivialFunctional, 50000)
        res_vec.append(res)
        
    plt.scatter(f0_vec,torch.tensor(res_vec),s=4,label="empirical")
    plt.plot(f0_vec,CRLB_analytic(model,f0_vec),color="red", linestyle="--",label="analytic")
    plt.legend()
    plt.xlabel("f0")
    plt.title("Empirical CRB vs Analytic CRB")
    plt.show()
