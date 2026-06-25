---
title: Batched Interventions
toc: true
---

> Status: **draft / under design** — companion to PR
> [#594](https://github.com/BasisResearch/chirho/pull/594).


## 1. Motivation

In search for explanation (and many other causal workflows), we need to
evaluate **large batches of interventions**. These interventions are
*heterogeneous* — different elements of the batch may intervene on different
variables, with different values.

ChiRho already provides two counterfactual handlers living at two edges of a spectrum:

- **`TwinWorldCounterfactual` (TWC)** — factual world plus **one** alternative
  world. All `split`s reuse a single shared dimension name, so the world axis
  has size 2. Computationally/memory-wise cheap, but one can track only one counterfactual scenario.

- **`MultiWorldCounterfactual` (MWC)** — factual world plus **every** combination
  of interventions. Each `split` allocates its **own** index-plate dimension, so
  the represented worlds are the full **Cartesian product** of all interventions.
  Memory grows **exponentially** in the number of intervened sites.

A *batched* intervention handler is the natural third point on this spectrum:
**many alternative scenarios laid side-by-side on a single shared axis**. This
gives access to many coordinated interventions without the compute and memory cost
of materializing every *combination* of them — combinations that, in practice,
are usually never inspected and are discarded anyway.


## 2. Goals & requirements

1. **Linear memory** in the batch size `N` — a single shared world axis
   (conceptually, TWC scaled to `N+1` worlds).

2. **Faster than explicitly looping** (with `do` or TWC) over the interventions —
   one vectorized forward pass instead of `N` separate model evaluations.

3. **Correct downstream propagation** 

4. **An always present factual world**, addressable like TWC/MWC via
   `gather(value, IndexSet(<name>={0}))`.

5. **Works with non-trivial event dimensions** (`to_event(k)`).


6. **Citizen status alongside TWC/MWC** — a `Messenger` class usable like the
   other handlers (as a context manager or decorator), composing with
   `FactualConditioningMessenger`, whose outputs are read with the standard
   `chirho.indexed` operations (`gather`, `IndexSet`, `indices_of`).

7. **Less memory and faster than MWC or looped TWC**

## 3. Key design decision: `where`-semantics, not `split`

A key question was whether the batched handler should route through the existing
`split` primitive (as TWC/MWC do) with a shared dimension name, or use a
`torch.where(mask, act, obs)` mechanism. Prototyping showed that `split` is the
wrong primitive here.

Recall how `split` builds worlds. `split(obs, acts, name)` puts the observed value
at index 0 (the factual world) and, for each alternative `act_i`, stores
`intervene(factual, act_i)` at index `i+1`. Every counterfactual
world is rebuilt from the factual, index-0 value, which breaks things if one wants to keep track of counterfactual worlds on a single axis.




On a single shared axis, suppose a scenario
intervenes on `z` but leaves `y` alone, in a model where `y` depends on `z`. By
the time the handler reaches `y`, that scenario's slot on the shared axis already
holds a value of `y` that has propagated the upstream `z` intervention. Leaving
`y` alone should keep that propagated value — but `split` rebuilds the slot
from the index-0 factual `y` (the world where `z` was never touched), erasing the
propagation. 



The proposed solution is to use `torch.where(mask, act, obs)`. This, by contrast, keeps `obs` wherever a
scenario does not intervene, which is exactly that propagated value.

Concretely, as an example, for the model `z -> {x, y}`, `y <- 0.8 x + 0.3 z`, with scenario 0 =
`{z: 100}` (and `y` untouched):

- **Ground truth (MWC, `gather(z={1}, y={0})`):** `z = 100`, `y = 108.66`
  (the `z = 100` intervention propagates into `y`).
- **Shared-name `split` prototype:** `z = 100`, `y = 0.06` — `y` was reset to the
  factual value. Wrong.

- **`torch.where` (original PR mechanism):** `z = 100`, `y ≈ 109` — propagation
  preserved. Desired.

The reason MWC gets propagation right is because it gives each site its
own axis. This, however, causes the memory blow-up in larger models with non-trivial explanation search.


A second, orthogonal decision concerns the exogenous noise of latent sites.
The question is whether we draw `BatchedLatents` over all latents, giving each scenario an independent draw. This is a modeling decision that should be left to the user.
Holding noise fixed across worlds (true counterfactuals) is achieved by
not batching the latents and instead letting the intervention's `where`
create the shared axis at intervened sites and propagate it downstream — exactly
as MWC does. 

Both can be supported via a **`shared_noise` flag (default `True`)**:

- `shared_noise=True` (default): drop `BatchedLatents`; exogenous noise shared; this should be the  MWC counterpart.
 
- `shared_noise=False`: keep `BatchedLatents`; independent per-scenario draws.
 

## 4. Mechanism

**Named batch dimension**

The single shared world axis is a named index-plate dimension of size `N+1`
tracked by `chirho.indexed`. With `shared_noise=True` (default) it is created
lazily by the intervention itself (Section 4.2) at the first intervened site and propagates downstream, leaving upstream latents shared/scalar. With
`shared_noise=False` it is instead created eagerly on every latent site by
`BatchedLatents(N+1, name=...)`, which already handles event dimensions
correctly (a `to_event(1)` latent is placed at the registered plate position.
For each intervened site, the handler computes

```
value = torch.where(mask, act, obs)
```

where  `act` and `mask` are first moved onto the named batch
dimension with

```
act  = unbind_leftmost_dim(act,  name, size=N+1, event_dim=event_dim)
mask = unbind_leftmost_dim(mask, name, size=N+1, event_dim=event_dim)
```

`event_dim` is the site's event dimensionality, which already flows through
`msg["kwargs"]` from `Interventions._pyro_post_sample`
(`intervene(..., event_dim=len(fn.event_shape), name=...)`).

this ensures that the leftmost axis coincides with the named plate
dimension.


We reserve index 0 of the batch axis as the factual world: an extra batch slice whose `mask` is `False` for every site, so the factual value propagates
through it unchanged. Scenarios occupy indices `1..N`.
This matches the TWC/MWC convention (`gather(value, IndexSet(name={0}))` is the factual world).

## 6. Intended semantics 

For batch size `N`, world axis of size `N + 1`:

- **Index 0** — factual world; no interventions applied.
- **Index `i+1`** — scenario `i`: for each site `s`,
  `value[i+1] = act_{s,i}` if scenario `i` intervenes on `s`, else the
  propagated `obs[i+1]` (i.e. the value implied by scenario `i`'s other
  interventions; factual if it has none upstream).
- Reading results: `gather(value, IndexSet(<name>={k}), event_dim=...)` selects world `k`; `indices_of(value, event_dim=...)` reports the populated axis.

**Equivalence property.** With `shared_noise=True`, a batched
world equals the corresponding `gather`ed MWC slice. For multi-site
interventions with sampled descendants, the batched single axis (size `N+1`)
and MWC's Cartesian axes have different shapes, so a shared-noise downstream
`sample` draws different noise per cell: the worlds are then **distributionally**
equivalent but not sample-path identical. 


