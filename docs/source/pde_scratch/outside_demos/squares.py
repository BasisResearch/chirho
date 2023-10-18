import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from fenics import *
from fenics_adjoint import *

import torch
import numpy as np

import torch_fenics


class Squares(torch_fenics.FEniCSModule):
    def __init__(self):
        super(Squares, self).__init__()
        mesh = IntervalMesh(4, 0, 1)
        self.V = FunctionSpace(mesh, 'DG', 0)

    def solve(self, f1, f2):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        a = u * v * dx
        L = f1**2 * f2**2 * v * dx

        u_ = Function(self.V)
        solve(a == L, u_)

        return u_

    def input_templates(self):
        return Function(self.V), Function(self.V)


def test_squares():
    f1 = torch.autograd.Variable(torch.tensor([[1, 2, 3, 4],
                                               [2, 3, 5, 6]]).double(), requires_grad=True)
    f2 = torch.autograd.Variable(torch.tensor([[2, 3, 5, 6],
                                               [1, 2, 2, 1]]).double(), requires_grad=True)

    rank = MPI.comm_world.Get_rank()
    size = MPI.comm_world.Get_size()
    f1 = f1[:,rank::size]
    f2 = f2[:,rank::size]

    squares = Squares()

    squares(f1, f2)

    assert np.all((squares(f1, f2) == f1**2 * f2**2).detach().numpy())
    assert torch.autograd.gradcheck(squares, (f1, f2))


if __name__ == '__main__':
    test_squares()
