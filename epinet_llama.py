from cgi import test
from cmath import exp, log
from multiprocessing import dummy
from operator import ne
from telnetlib import EXOPL
from tokenize import Single
import warnings

from enn.losses import single_index

warnings.filterwarnings('ignore')

#@title Development imports
from typing import Callable, NamedTuple, Sequence, Optional

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
import functools

@dataclasses.dataclass
class Config:
    feature_size: int = 4096
    num_classes: int = 32000
    num_batch: int = 200
    index_dim: int = 10
    num_index_samples: int = 100
    seed: int = 0
    prior_scale: float = 1.
    learning_rate: float = 1e-3
    noise_std: float = 0.1
    epinet_hiddens: Sequence[int] = [1024,256,1024,4096]

config = Config()


#### Create a single linear layer with loaded weights
class FrozenLinerLayer(hk.Module):
    def __init__(
        self, 
        input_size, 
        output_size, 
        weight, 
        bias=hk.initializers.Constant(0.0)):
        
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        w = hk.get_parameter(
            "pretrained_weights", shape=(self.output_size, self.input_size), init=self.weight)
        b = hk.get_parameter("bias", shape=(self.output_size,1), init=self.bias)
        # w = jax.lax.stop_gradient(w)
        # b = jax.lax.stop_gradient(b)
        y = jnp.dot(w, x) + b
        return jax.lax.stop_gradient(y)

class MatrixInitializer(hk.initializers.Initializer):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, shape, dtype):
        return self.weight

### TODO: 1. create base network for Llama-2 
###          (simplification: identity matrix, receiving dola features from DoLA-enhanced model)

def projection_layer(x, feature_size, logit_size, vocab_head_weight):
    vocab_head = FrozenLinerLayer(
                        input_size=feature_size, 
                        output_size=logit_size, 
                        weight=MatrixInitializer(vocab_head_weight))
    return vocab_head(x)



### TODO: 2. create epinet for the whole enn
class MLPEpinetWithTrainableAndPrior(networks.epinet.EpinetWithState):
  """MLP epinet with matching prior function."""
  def __init__(self,
               projection_layer,
               index_dim: int,
               num_classes: int,
               epinet_hiddens: Sequence[int],
               prior_epinet_hiddens: Optional[Sequence[int]] = None,
               prior_scale: float = 1):
    """Defines an MLP epinet with matching prior function."""
    if prior_epinet_hiddens is None:
      prior_epinet_hiddens = epinet_hiddens

    def epinet_fn(index: base.Index,
                  hidden: chex.Array) -> networks.base.OutputWithPrior:
      # Creating networks
      train_epinet = networks.ProjectedMLP(
          epinet_hiddens, num_classes, index_dim, name='train_epinet')
      prior_epinet = networks.ProjectedMLP(
          prior_epinet_hiddens, num_classes, index_dim, name='prior_epinet')

      epi_inputs = hidden

      # Wiring networks: add linear epinet (+ prior) from final output layer.
      epi_train_logits = projection_layer(train_epinet(epi_inputs, index))
      epi_prior_logits = projection_layer(prior_epinet(epi_inputs, index))
      return networks.OutputWithPrior(
          train=epi_train_logits,
          prior=prior_scale * epi_prior_logits,
      )

    # Form ENN from haiku transformed.
    transformed = hk.without_apply_rng(hk.transform_with_state(epinet_fn))
    indexer = networks.GaussianIndexer(index_dim)
    super().__init__(transformed.apply, transformed.init, indexer)

### TODO: 3. create loss function

class XentLoss(losses.SingleLossFnArray):
    """Cross-entropy single index loss with network state as auxiliary."""

    def __init__(self, num_classes: int):
        assert num_classes > 1
        super().__init__()
        self.num_classes = num_classes
        labeller = lambda x: jax.nn.one_hot(x, self.num_classes)
        self._loss = self.xent_loss_with_dola_distributions(labeller)

    def __call__(
        self,
        apply: networks.ApplyArray,
        params: hk.Params,
        state: hk.State,
        batch: datasets.ArrayBatch,
        index: base.Index,
    ) -> base.LossOutput:
        return self._loss(apply, params, state, batch, index)

    def xent_loss_with_dola_distributions(self,
        labeller: Callable[[chex.Array], chex.Array]
    ) -> losses.SingleLossFnArray:
        """Factory method to create a loss function with custom labelling."""

        def single_loss(
            apply: networks.ApplyArray,
            params: hk.Params,
            state: hk.State,
            batch: datasets.ArrayBatch,
            index: base.Index,
        ) -> base.LossOutput:
            """Xent loss with custom labelling."""
            chex.assert_shape(batch.y, (None, 1))
            net_out, state = apply(params, state, batch.x, index)
            logits = networks.parse_net_output(net_out)
            labels = labeller(batch.y[:, 0])

            # combine with dola distributions
            # logits.shape = [batch_size, num_classes]
            softmax_xent = -jnp.sum(
                labels * jax.nn.softmax(jax.nn.softmax(logits) + batch.extra['dola_distribution']), axis=1, keepdims=True)

            if batch.weights is None:
                batch_weights = jnp.ones_like(batch.y)
            else:
                batch_weights = batch.weights
            chex.assert_equal_shape([batch_weights, softmax_xent])

            loss = jnp.mean(batch_weights * softmax_xent)
            return loss, (state, {'loss': loss})
        return single_loss


# create dataset for training
dataset = None

# load vocab head here
vocab_head_pretrained_weight = None

vocab_head = functools.partial(projection_layer, 
                            feature_size=config.feature_size, 
                            logit_size=config.num_classes, 
                            vocab_head_weight=vocab_head_pretrained_weight)

epinet = MLPEpinetWithTrainableAndPrior(
               projection_laye=vocab_head,
               index_dim=config.index_dim,
               num_classes=config.num_classes,
               epinet_hiddens=config.epinet_hiddens)

loss_fn = losses.average_single_index_loss(
    single_loss=XentLoss(config.num_classes),
    num_index_samples=config.num_index_samples
)

optimizer = optax.adam(config.learning_rate)

logger = TerminalLogger('supervised_regression')

experiment = supervised.Experiment(
    enn, loss_fn, optimizer, dataset, config.seed, logger=logger)

experiment.train(config.num_batch)


### TODO: 4. create training and evaluation processes
