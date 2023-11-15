from cgi import test
from cmath import exp, log
from multiprocessing import dummy
from telnetlib import EXOPL
from tokenize import Single
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
import functools

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
feature_size = 4096

def base_model(x, feature_size, logit_size, vocab_head_weight):
    vocab_head = FrozenLinerLayer(feature_size, logit_size, weight=MatrixInitializer(vocab_head_weight))
    return vocab_head(x)

# load vocab head here
vocab_head_pretrained_weight = None

base_model = functools.partial(base_model, feature_size=feature_size, logit_size = feature_size, vocab_head_weight=vocab_head_pretrained_weight)

### TODO: 2. create epinet for the whole enn



###### TODO: create trainable network

###### TODO: create prior network

###### TODO: combine networks together to form the epinet
######       (Note: stop gradient by: output = self.train + jax.lax.stop_gradient(self.prior))

### TODO: 3. combine epinet with base network
###          (Note: perform softmax() after getting base_out + softmax(epinet_logits))



### TODO: 4. create training and evaluation processes
