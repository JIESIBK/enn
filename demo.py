from cgi import test
from cmath import exp, log
from multiprocessing import dummy
from telnetlib import EXOPL
import warnings

warnings.filterwarnings('ignore')

#@title Development imports
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import plotnine as gg

import dataclasses
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

## ENN-demo
#@title ENN imports
import enn
from enn import losses
from enn import networks
from enn import supervised
from enn import base
from enn import data_noise
from enn import utils
from enn import datasets
from enn.loggers import TerminalLogger
from enn.supervised import classification_data
from enn.supervised import regression_data

import random

class MLPEnsembleMatchedPriorModified(networks.base.EnnArray):
  """Ensemble of MLPs with matched prior functions."""
  from typing import Sequence, Optional
  def __init__(self,
               output_sizes: Sequence[int],
               dummy_input: chex.Array,
               num_ensemble: int,
               prior_scale: float = 1.,
               seed: int = 0,
               w_init: Optional[hk.initializers.Initializer] = None,
               b_init: Optional[hk.initializers.Initializer] = None):
    """Ensemble of MLPs with matched prior functions."""
    mlp_priors = networks.make_mlp_ensemble_prior_fns(
        output_sizes, dummy_input, num_ensemble, seed)

    def net_fn(x: chex.Array) -> chex.Array:
      x = hk.Flatten()(x)
      return hk.nets.MLP(output_sizes, w_init, b_init)(x)
    transformed = hk.without_apply_rng(hk.transform_with_state(net_fn))

    ensemble = networks.EnsembleWithState(transformed, num_ensemble)
    enn = networks.priors.EnnWithAdditivePrior(
        enn=ensemble,
        prior_fn=networks.combine_functions_choice_via_index(mlp_priors),
        prior_scale=prior_scale,
    )
    super().__init__(enn.apply, enn.init, enn.indexer)


@dataclasses.dataclass
class Config:
  num_batch: int = 200
  index_dim: int = 10
  num_index_samples: int = 10
  seed: int = 0
  prior_scale: float = 5.
  learning_rate: float = 1e-3
  noise_std: float = 0.1

def get_dummy_dataset(input_dim, num_classes, num_batch, batch_size):
    seed = 0
    x = np.random.RandomState(seed).randn(input_dim, num_batch * batch_size).T
    y = np.random.RandomState(seed).randint(0,num_classes, num_batch * batch_size)
    dola_distribution = np.random.RandomState(seed).randn(num_classes, num_batch * batch_size).T
    print(x[0], y[0])
    return utils.make_batch_iterator(datasets.ArrayBatch(x=x, y=y, extra={"dola_distribution": dola_distribution}), batch_size)
            

FLAGS = Config()

output_dim = 32000
num_classes = 32000
input_dim = 4096
batch_size = 7


dataset = get_dummy_dataset(input_dim, num_classes, 10, batch_size)

dummy_input = next(dataset).x

cnt = 0

print(dummy_input, dummy_input.shape)

# dummy_input = np.asarray([[0]*4096 for _ in range(10)])

enn = networks.MLPEnsembleMatchedPrior(
# enn = MLPEnsembleMatchedPriorModified(
    # supposed to be 4096 for output_dim here
    output_sizes=[50,50,output_dim],
    dummy_input=dummy_input,
    num_ensemble=FLAGS.index_dim,
    prior_scale=FLAGS.prior_scale,
    seed=FLAGS.seed,
)

loss_fn = losses.average_single_index_loss(
    single_loss=losses.XentLoss(num_classes),
    num_index_samples=FLAGS.num_index_samples
)

optimizer = optax.adam(FLAGS.learning_rate)

logger = TerminalLogger('supervised_regression')

experiment = supervised.Experiment(
    enn, loss_fn, optimizer, dataset, FLAGS.seed, logger=logger)

experiment.train(FLAGS.num_batch)


test_data = next(dataset)
test_input = test_data.x
ground_truth = test_data.y

rng = hk.PRNGSequence(jax.random.PRNGKey(seed=0))

net_out = experiment.predict(test_input, next(rng))
logits = networks.parse_net_output(net_out=net_out)
preds_y = jax.nn.softmax(logits)
label = jax.numpy.argmax(preds_y, axis=1)

# print(type(ground_truth), type(label))

print("GT: ", ground_truth.reshape(ground_truth.shape[1], -1))
print("Pred: ", label)