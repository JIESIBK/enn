import haiku as hk
import jax
from jax import numpy as jnp
import chex
import functools 

from enn.datasets.imagenet import load

class SingleLayerModule(hk.Module):
    def __init__(self, input_size, output_size, weight):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = weight

    def __call__(self, x):
        w = hk.get_parameter("pretrained_weights", shape=(self.output_size, self.input_size), init=self.weight)
        b = hk.get_parameter("bias", shape=(self.output_size,1), init=hk.initializers.Constant(0.0))
        y = jnp.dot(w, x) + b
        return jax.lax.stop_gradient(y)

class MatrixInitializer(hk.initializers.Initializer):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, shape, dtype):
        return self.weight

def create_model(x, input_size, output_size, load_pretrained_weights):
    return SingleLayerModule(input_size, output_size, MatrixInitializer(load_pretrained_weights))(x)

input_size = 3
output_size = 5

# weight = [[0.1, 0.2, 0.3], [1, 2, 3], [10, 20, 30], [7, 8, 9], [4, 5, 6]]
# 5x3
rng = jax.random.PRNGKey(42)
load_pretrained_weights = jax.random.uniform(rng, shape=(5,3))
create_model = functools.partial(create_model, input_size=input_size, output_size = output_size, load_pretrained_weights=load_pretrained_weights)

# 3x1
x = jnp.ones((input_size, 1))

# determine whether it should be float64
create_model_transformed = hk.transform(create_model)


params = create_model_transformed.init(rng, x)
y = create_model_transformed.apply(params, rng, x)

# print(load_pretrained_weights)

for i in range(load_pretrained_weights.shape[0]):
    value_ground_truth = (jnp.dot(load_pretrained_weights[i,:].reshape(1, 3), x)).sum()
    chex.assert_equal(y[i], value_ground_truth)
    

gradient = jax.grad(lambda params, rng, x: create_model_transformed.apply(params, rng, x).sum())(params, rng, x)

print(x.T, '\n', y.T, '\n', gradient)