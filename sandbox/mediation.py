from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.contrib.autoname import scope
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.poutine import condition, reparam

import causal_pyro
from causal_pyro.counterfactual.handlers import (Factual,
                                                 MultiWorldCounterfactual,
                                                 TwinWorldCounterfactual)
from causal_pyro.query.do_messenger import do

# The csv data is stored at http://data.mxnet.io/data/personality.csv
# Let's load the data from the url
data_url = 'https://statsnotebook.io/blog/data_management/example_data/substance.csv'
data = pd.read_csv(data_url)

# Show the data
data.head()

# Check if there are any NaNs
data.isnull().values.any()

# For the moment let's just remove any rows that have missing data
ddata = data.dropna()

# Check if there are any NaNs
data.isnull().values.any()

# Let's check the data types
data.dtypes

keys_ = data.keys()
# Now let's get a key for first column
key_ = keys_[0]


# Variables

# dev_peer represents engagement with deviant peer groups and it was coded as “0: No” and “1: Yes”;
# sub_exp represents experimentation with drugs and it was coded as “0: No” and “1: Yes”;
# fam_int represents participation in family intervention during adolescence and it was coded as “0: No” and “1: Yes”;
# sub_disorder represents diagnosis of substance use disorder in young adulthood and it was coded as “0: No: and “1: Yes”.
# conflict represents level of family conflict. It will be used as a covariate in this analysis.

# Does reduced engagement with deviant peer groups and reduced experimentation with drugs mediate the effect of the family intervention during adolescence on future substance use disorder?
# In the example, we examine the causal chain “Participation in family intervention -> reduced engagement with deviant peer groups and experimentation with drugs -> substance use disorder in young adulthood”.
# In other words, we expect that participation in family intervention during adolescence reduces engagement with deviant peer groups and experimentation with drugs, which in turns reduces the likelihood for substance use disorder in young adulthood.

# Let's plot the distribution over the field Agreeableness

# Let's do logistic regression in Pyro
# Assumine that 
class LogisticRegression(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(LogisticRegression, self).__init__()
        self.linear = PyroModule[torch.nn.Linear](D_in, D_out)
        self.sigmoid = PyroModule[torch.nn.Sigmoid]()

    def forward(self, x):
        y_pred = self.sigmoid(self.linear(x))
        return y_pred

# Now let's make the causal model for the data
class CausalModel(PyroModule):
  def __init__(self):
    super().__init__()
    self.lr1 = PyroModule[LogisticRegression](1, 1)
    self.lr1.weight = PyroSample(dist.Normal(0., 1.).expand([1, 1]).to_event(2))
    self.lr1.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))
    self.lr2 = PyroModule[LogisticRegression](1, 1)
    self.lr2.weight = PyroSample(dist.Normal(0., 1.).expand([1, 1]).to_event(2))
    self.lr2.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))
    self.lr3 = PyroModule[LogisticRegression](2, 1)
    self.lr3.weight = PyroSample(dist.Normal(0., 1.).expand([1, 2]).to_event(2))
    self.lr3.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))

  def forward(self):
    # Family intervention is exogenous
    with pyro.plate("obs", len(ddata), dim = -2) as ind:
      fam_int = pyro.sample("fam_int", dist.Bernoulli(torch.tensor([0.5])))
      # print("fam_int", fam_int.shape)
      dev_peer = pyro.sample("dev_peer", dist.Bernoulli(self.lr1(fam_int)))
      # print("dev_peer", dev_peer.shape)
      sub_exp = pyro.sample("sub_exp", dist.Bernoulli(self.lr2(fam_int)))
      sub_disorder_ = self.lr3(torch.cat((dev_peer, sub_exp), dim = -1))
      sub_disorder = pyro.sample("sub_disorder", dist.Bernoulli(sub_disorder_))
      return fam_int, dev_peer, sub_exp, sub_disorder

class CausalModel2(PyroModule):
  def __init__(self):
    super().__init__()
    self.lr1 = LogisticRegression(1, 1)
    self.lr1.weight = PyroSample(dist.Normal(0., 1.).expand([1, 1]).to_event(2))
    self.lr1.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))
    self.lr2 = LogisticRegression(1, 1)
    self.lr2.weight = PyroSample(dist.Normal(0., 1.).expand([1, 1]).to_event(2))
    self.lr2.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))
    self.lr3 = LogisticRegression(2, 1)
    self.lr3.weight = PyroSample(dist.Normal(0., 1.).expand([1, 2]).to_event(2))
    self.lr3.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))

  def forward(self, fam_int_data = None, dev_peer_data = None, sub_exp_data = None, sub_disorder_data = None):
    # Family intervention is exogenous
    with pyro.plate("obs", len(ddata), dim = -2) as ind:
      fam_int = pyro.sample("fam_int", dist.Bernoulli(torch.tensor([0.5])), obs = fam_int_data)
      # print("fam_int", fam_int.shape)
      dev_peer = pyro.sample("dev_peer", dist.Bernoulli(self.lr1(fam_int)), obs = dev_peer_data)
      # print("dev_peer", dev_peer.shape)
      sub_exp = pyro.sample("sub_exp", dist.Bernoulli(self.lr2(fam_int)), obs = sub_exp_data)
      sub_disorder_ = self.lr3(torch.cat((dev_peer, sub_exp), dim = -1))
      sub_disorder = pyro.sample("sub_disorder", dist.Bernoulli(sub_disorder_), obs = sub_disorder_data)
      return fam_int, dev_peer, sub_exp, sub_disorder

# Inference with data
# Now let's do inference with the data
# We will use the data from the csv file

# Now let's use pyro.plate to do inference
# with the data:
# fam_int, dev_peer, sub_exp, sub_disorder
def execute_model():
  m = CausalModel()
  with pyro.plate("data", len(ddata), dim = -2) as ind:
    return m()

# converet panda row into torch tensor
def p_t(row):
  return torch.tensor(row.values).reshape(-1, 1).float()

def conditioned_model():
  return pyro.condition(execute_model, data={"fam_int": p_t(ddata["fam_int"])})

# Inference
def train(cm, *args):
  from pyro.infer import SVI, Trace_ELBO
  from pyro.infer.autoguide import AutoDiagonalNormal
  guide = pyro.infer.autoguide.AutoDelta(cm)
  # guide = pyro.infer.autoguide.AutoDiagonalNormal(cm)
  adam = pyro.optim.Adam({"lr": 0.03})
  svi = SVI(cm, guide, adam, loss=Trace_ELBO())
  num_iterations = 10000
  for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(*args)
    if j % 10 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))
  l = svi.step(*args)
  return l

m = CausalModel()
cm = pyro.condition(m, data={"fam_int": p_t(ddata["fam_int"]), \
                              "dev_peer": p_t(ddata["dev_peer"]), \
                              "sub_exp": p_t(ddata["sub_exp"]), \
                              "sub_disorder": p_t(ddata["sub_disorder"])})


result = train(cm)


# cm2 = CausalModel2()
# result = train(cm2, p_t(ddata["fam_int"]), p_t(ddata["dev_peer"]), p_t(ddata["sub_exp"]), p_t(ddata["sub_disorder"]))


# num_iterations = 1000
# pyro.clear_param_store()
# svi.step()
# for j in range(num_iterations):
#     # calculate the loss and take a gradient step
#     loss = svi.step(x_data, y_data)
#     if j % 100 == 0:
#         print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))


# Ok now for the mediation