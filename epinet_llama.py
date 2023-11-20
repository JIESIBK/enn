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

import time
current_time = int(time.time())

@dataclasses.dataclass
class Config:
    feature_size: int = 128
    num_classes: int = 32000
    num_batch: int = 10
    batch_size: int = 10
    index_dim: int = 10
    num_index_samples: int = 100
    seed: int = current_time
    prior_scale: float = 1.
    learning_rate: float = 1e-3
    num_epoch: int = 50
    noise_std: float = 0.1

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
            "pretrained_weights", shape=(self.input_size, self.output_size), init=self.weight)
        b = hk.get_parameter("bias", shape=(1, self.output_size), init=self.bias)
        w = jax.lax.stop_gradient(w)
        b = jax.lax.stop_gradient(b)
        y = jnp.dot(x, w) + b
        return y

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

    def epinet_fn(hidden: chex.Array,
                  index: base.Index) -> networks.base.OutputWithPrior:
      # Creating networks
      train_epinet = networks.ProjectedMLP(
          epinet_hiddens, num_classes, index_dim, name='train_epinet')
      prior_epinet = networks.ProjectedMLP(
          prior_epinet_hiddens, num_classes, index_dim, name='prior_epinet')

      epi_inputs = hidden

      # Wiring networks: add linear epinet (+ prior) from final output layer.
    #   print("epi_inputs: ", epi_inputs.shape)
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
            # combined_logits = jax.nn.softmax(logits) + jax.lax.stop_gradient(batch.extra['dola_distribution'])
            combined_logits = logits + jax.lax.stop_gradient(batch.extra['dola_distribution'])
            # combined_logits = logits
            softmax_xent = -jnp.sum(
                labels * jax.nn.log_softmax(combined_logits), axis=1, keepdims=True)

            if batch.weights is None:
                batch_weights = jnp.ones_like(batch.y)
            else:
                batch_weights = batch.weights
            chex.assert_equal_shape([batch_weights, softmax_xent])

            loss = jnp.mean(batch_weights * softmax_xent)
            return loss, (state, {'loss': loss})
        return single_loss


# create dataset for training
def get_dummy_dataset(input_dim, num_classes, num_batch, batch_size):
    seed = 0
    # (num_batch * batch_size, input_dim)
    x = np.random.RandomState(seed).randn(input_dim, num_batch * batch_size).T
    y = np.random.RandomState(seed).randint(0,num_classes, num_batch * batch_size)
    # (num_batch * batch_size, num_classes)
    dola_distribution = np.random.RandomState(seed).randn(num_classes, num_batch * batch_size).T
    dola_distribution = jax.nn.softmax(dola_distribution)
    # print(x.shape, y.shape, dola_distribution.shape)
    return utils.make_batch_iterator(data=datasets.ArrayBatch(x=x, 
                                                         y=y, 
                                                         extra={"dola_distribution": dola_distribution}), 
                                     batch_size=batch_size)

def grid_search(model, loss_fn, dataset, seed, logger, num_batch, start_rate, end_rate, num_sector):
    lr_range = np.logspace(np.log10(start_rate), np.log10(end_rate), num_sector).tolist()
    best_losses = []
    final_epochs = []
    for lr in lr_range:
        optimizer = optax.adam(config.learning_rate)
        experiment = supervised.Experiment(epinet, loss_fn, 
                                           optimizer, 
                                           dataset, seed, logger)
        print("Training with lr: ", lr)
        best_loss, final_epoch = experiment.train(num_batch)
        best_losses.append(best_loss)
        final_epochs.append(final_epoch)
    
    print("Learning_rate: ", lr_range)
    print("Best_loss: ", best_losses)
    print("Epoch_elapsed: ", final_epochs)

    min_loss = min(best_losses)
    min_pos = lr_range[best_losses.index(min_loss)]

    if min_pos == 0:
        start_rate = lr_range[0]
        end_rate = lr_range[1]
    elif min_pos == num_batch-1:
        start_rate = lr_range[min_pos-1]
        end_rate = lr_range[min_pos]      
    else:
        if best_losses[min_loss-1] < best_losses[min_loss+1]:
            start_rate = lr_range[min_pos-1]
            end_rate = lr_range[min_pos]
        else:
            start_rate = lr_range[min_pos]
            end_rate = lr_range[min_pos+1]
    
    lr_range = np.linspace(start_rate, end_rate, num_sector).tolist()
    best_losses = []
    final_epochs = []

    for lr in lr_range:
        optimizer = optax.adam(config.learning_rate)
        experiment = supervised.Experiment(epinet, loss_fn, 
                                           optimizer, 
                                           dataset, seed, logger)
        print("Training with lr: ", lr)
        best_loss, final_epoch = experiment.train(num_batch)
        best_losses.append(best_loss)
        final_epochs.append(final_epoch)
    
    print("Learning_rate: ", lr_range)
    print("Best_loss: ", best_losses)
    print("Epoch_elapsed: ", final_epochs)

    return (min(best_losses), lr_range[best_losses.index(min(best_losses))])


dataset = get_dummy_dataset(config.feature_size, config.num_classes, config.num_batch, config.batch_size)

# print(next(dataset).x.shape, next(dataset).y.shape, next(dataset).extra['dola_distribution'].shape)
# print(next(dataset).extra['dola_distribution'].sum(axis=1))

# load vocab head here
vocab_head_pretrained_weight = jax.random.uniform(jax.random.PRNGKey(42), shape=(config.feature_size, config.num_classes))

vocab_head = functools.partial(projection_layer, 
                            feature_size=config.feature_size, 
                            logit_size=config.num_classes, 
                            vocab_head_weight=vocab_head_pretrained_weight)

epinet = MLPEpinetWithTrainableAndPrior(
               projection_layer=vocab_head,
               index_dim=config.index_dim,
               num_classes=config.feature_size,
            #    epinet_hiddens=[1024,256])
               epinet_hiddens=[50,50])  ## test for grid_search

loss_fn = losses.average_single_index_loss(
    single_loss=XentLoss(config.num_classes),
    num_index_samples=config.num_index_samples
)

logger = TerminalLogger('supervised_regression')

# optimizer = optax.adam(config.learning_rate)

# experiment = supervised.Experiment(
#     epinet, loss_fn, optimizer, dataset, config.seed, logger=logger)

# experiment.train(config.num_epoch*config.num_batch)

best_loss, best_lr = grid_search(epinet, loss_fn, dataset, config.seed, logger, config.num_batch, 1e-5, 1e-2, 4)


############### validation

optimizer = optax.adam(best_lr)

experiment = supervised.Experiment(
    epinet, loss_fn, optimizer, dataset, config.seed, logger)

experiment.train(config.num_batch)

test_data = next(dataset)
test_input = test_data.x
test_dola = test_data.extra['dola_distribution']
ground_truth = test_data.y


rng = hk.PRNGSequence(jax.random.PRNGKey(seed=42))
net_out = experiment.predict(test_input, next(rng))
logits = networks.parse_net_output(net_out=net_out)
preds_y = jax.nn.softmax(logits + test_dola)
label = jax.numpy.argmax(preds_y, axis=1)

# print(type(ground_truth), type(label))

print("GT: ", ground_truth.reshape(ground_truth.shape[1], -1))
print("Pred: ", label)


### TODO: 4. create training and evaluation processes
