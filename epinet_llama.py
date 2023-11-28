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
import torch

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

# Set environment variables for using GPU with JAX for training
import os 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".9"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
tf.config.experimental.set_visible_devices([], "GPU")

# Import util functions 
from epinet_llama_utils import *

import yaml

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)


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

### 1. create base network for Llama-2 
###          (simplification: identity matrix, receiving dola features from DoLA-enhanced model)

def projection_layer(x, feature_size, logit_size, vocab_head_weight):
    vocab_head = FrozenLinerLayer(
                        input_size=feature_size, 
                        output_size=logit_size, 
                        weight=MatrixInitializer(vocab_head_weight))
    return vocab_head(x)



### 2. create epinet for the whole enn
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

### 3. create loss function

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

    # Load the actual DoLa dataset for epinet training
    feats_actual = torch.load('/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/data/dola_data_test/CSE8803-DLT/C4_data_100samples/layer_features.pt')
    dola_actual = torch.load('/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/data/dola_data_test/CSE8803-DLT/C4_data_100samples/dola_output_logits.pt')
    labels_actual = torch.load('/srv/kira-lab/share4/yali30/fall_23/cse_8803/enn/data/dola_data_test/CSE8803-DLT/C4_data_100samples/labels.pt')

    # Remove the last row from each tensor using list comprehension as the last token does NOT have a next word prediction label
    feats_actual = [tensor[:,:-1,:] for tensor in feats_actual]
    dola_actual = [tensor[:-1,:] for tensor in dola_actual]

    # Reshape the dataset components appropriately for epinet training
    feats_actual = torch.cat(feats_actual, dim=1)
    feats_actual = feats_actual.reshape(-1, input_dim)      # (num_samples, input_dim)
    feats_actual = feats_actual.cpu().detach().numpy()

    dola_actual = torch.cat(dola_actual, dim=0)             # (num_samples, 32000)
    dola_actual = dola_actual.cpu().detach().numpy()
    dola_actual = jax.nn.softmax(dola_actual)               # Convert the DoLa logits into softmax distributions

    labels_actual = torch.cat(labels_actual, dim=1)         # (1, num_samples)
    labels_actual = labels_actual.squeeze(0).cpu().detach().numpy()

    feats_actual = feats_actual[:40960,:]
    dola_actual = dola_actual[:40960,:]
    labels_actual = labels_actual[:40960]

    print("\n Dummy dataset shapes: ")
    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape)
    print("dola dist shape: ", dola_distribution.shape)

    print("\n Actual dataset shapes: ")
    print("feats_actual.shape: ", feats_actual.shape)
    print("labels_actual.shape: ", labels_actual.shape)
    print("dola_actual shape: ", dola_actual.shape)

    return utils.make_batch_iterator(data=datasets.ArrayBatch(x=x, 
                                                         y=y, 
                                                         extra={"dola_distribution": dola_distribution}), 
                                     batch_size=batch_size)

    # return utils.make_batch_iterator(data=datasets.ArrayBatch(x=feats_actual, 
    #                                                      y=labels_actual, 
    #                                                      extra={"dola_distribution": dola_actual}), 
    #                                  batch_size=batch_size)

print("Loading DoLa dataset....")
dataset = get_dummy_dataset(config.feature_size, config.num_classes, config.num_batch, config.batch_size)
print("Loaded DoLa dataset !")

# print(next(dataset).x.shape, next(dataset).y.shape, next(dataset).extra['dola_distribution'].shape)
# print(next(dataset).extra['dola_distribution'].sum(axis=1))

# load vocab head here
vocab_head_pretrained_weight = jax.random.uniform(jax.random.PRNGKey(42), shape=(config['feature_size'], config['num_classes']))

vocab_head = functools.partial(projection_layer, 
                            feature_size=config['feature_size'], 
                            logit_size=config['num_classes'], 
                            vocab_head_weight=vocab_head_pretrained_weight)

epinet = MLPEpinetWithTrainableAndPrior(
               projection_layer=vocab_head,
               index_dim=config['index_dim'],
               num_classes=config['feature_size'],
               epinet_hiddens=config['epinet_hiddens'])

loss_fn = losses.average_single_index_loss(
    single_loss=XentLoss(config['num_classes']),
    num_index_samples=config['num_index_samples']
)

logger = TerminalLogger('supervised_regression')

# optimizer = optax.adam(config.learning_rate)

# experiment = supervised.Experiment(
#     epinet, loss_fn, optimizer, dataset, config.seed, logger=logger)

# experiment.train(config.num_epoch*config.num_batch)

# best_loss, best_lr = grid_search(epinet, loss_fn, dataset, config.seed, logger, config.num_batch, 1e-5, 1e-1, 5)

############### validation

# Create learning_rate scheduler
total_steps = 100
linear_decay_scheduler = optax.linear_schedule(init_value=0.001, end_value=0.00001,
                                               transition_steps=total_steps,
                                               transition_begin=int(total_steps*0.1))

# best_lr = 1e-5
optimizer = optax.adam(learning_rate=linear_decay_scheduler)


############### validation
with open("training.log", 'w') as f:
    f.write("Training with input: {} hidden layer size: {}".format(config['feature_size'], config['epinet_hiddens']))
print("Training with input:", config['feature_size'], "hidden layer size:", config['epinet_hiddens'])

experiment = supervised.Experiment(
    linear_decay_scheduler, epinet, loss_fn, optimizer, dataset, config['seed'], logger)

experiment.train(config['num_batch'])

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

print("Trained with input:", config['feature_size'], "hidden layer size:", config['epinet_hiddens'])

