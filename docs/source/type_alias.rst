.. _type-alias-R:

.. py:data:: R
    :annotation: = Union[numbers.Real, torch.Tensor]

    Represents either a real number or a tensor that is typically assumed to be a scalar (e.g. torch.tensor(1.0)).

.. _type-alias-State:

.. py:data:: State
    :annotation: = Mapping[str, T]

    Represents the state of a system as a mapping. The keys are strings
    representing state variable names, and the values are of type T, which is
    a generic placeholder for the state variable type. Importantly, this can also
    represent a mapping from state variable names to
    their instantaneous rates of change (dstate/dt).

.. _type-alias-Dynamics:

.. py:data:: Dynamics
    :annotation: = Callable[[State[T]], State[T]]

    Represents the dynamics of a system. It's a function type that takes a
    `State[T]` and returns a new `State[T]`, where the returned value is a
    mapping from state variable names to their instantaneous rates of change
    dstate/dt.
