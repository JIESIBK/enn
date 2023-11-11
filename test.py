import haiku as hk
import jax
from jax import numpy as jnp

class SingleLayerModule(hk.Module):
    def __init__(self, output_size, weight):
        super().__init__()
        self.output_size = output_size
        self.weight = weight

    def __call__(self, x):
        w = hk.get_parameter("pretrained_weights", shape=(x.shape[-1], self.output_size), init=self.weight)
        w = jax.lax.stop_gradient(w)
        b = hk.get_parameter("bias", shape=(self.output_size,), init=hk.initializers.Constant(0.0))
        b = jax.lax.stop_gradient(b)
        y = jnp.dot(x, w) + b
        return y

class MatrixInitializer(hk.initializers.Initializer):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, shape, dtype):
        return self.weight

def create_model(x, load_pretrained_weights):
    load_pretrained_weights = MatrixInitializer(load_pretrained_weights)
    model = SingleLayerModule(load_pretrained_weights.weight.shape[1], load_pretrained_weights)
    return model(x)


input_size = 3
output_size = 5
weight = [[0.1, 0.2, 0.3], [1, 2, 3], [10, 20, 30], [7, 8, 9], [4, 5, 6]]
# 3x5
load_pretrained_weights = jnp.array(weight, dtype=jnp.float32).T

# 1x3
x = jnp.ones((1, input_size))

# determine whether it should be float64
create_model_transformed = hk.transform(create_model)

rng = jax.random.PRNGKey(42)
params = create_model_transformed.init(rng, x, load_pretrained_weights)
y = create_model_transformed.apply(params, rng, x, load_pretrained_weights)

gradient = jax.grad(lambda params, rng, x, load_pretrained_weights: create_model_transformed.apply(params, rng, x, load_pretrained_weights).sum())(params, rng, x, load_pretrained_weights)

print(x, '\n', y, '\n', gradient)