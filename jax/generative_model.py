import jax
import jax.numpy as np
from jax.experimental import optimizers
from jax.experimental import stax
from functools import partial

import matplotlib.pyplot as plt
import numpy as onp

from sklearn.datasets import make_swiss_roll

from IPython.display import clear_output


def sample_batch(size, noise=1.0):
    x, _= make_swiss_roll(size, noise=noise)
    x = x[:, [0, 2]] / 10.0
    return onp.array(x)


def get_compute_loss(net_apply):
	# pass in the net_apply function and enable jit compilation
	@jax.jit
	def compute_loss(net_params, inputs):
	    jacobian = jax.jacfwd(net_apply, argnums=-1)
	    batch_jacobian = jax.vmap(partial(jacobian, net_params))(inputs)
	    trace_jacobian = np.trace(batch_jacobian, axis1=1, axis2=2)
	    output_norm_sq = np.square(net_apply(net_params, inputs)).sum(axis=1)
	    return np.mean(trace_jacobian + 1/2 * output_norm_sq)

	return compute_loss


def get_train_step(opt_update, get_params, net_apply):
	# pass in the net_apply function and enable jit compilation
	compute_loss = get_compute_loss(net_apply)

	@jax.jit
	def train_step(step_i, opt_state, batch):
	    net_params = get_params(opt_state)
	    loss = compute_loss(net_params, batch)
	    grads = jax.grad(compute_loss, argnums=0)(net_params, batch)
	    return loss, opt_update(step_i, grads, opt_state)

	return train_step


def plot_gradients(loss_history, opt_state, get_params, net_params, net_apply):
	# plot loss
	clear_output(True)
	plt.figure(figsize=[16, 8])
	plt.subplot(1, 2, 1)
	plt.title("mean loss = %.3f" % np.mean(np.array(loss_history[-32:])))
	plt.scatter(np.arange(len(loss_history)), loss_history)
	plt.grid()

	# plot gradient vectors
	plt.subplot(1, 2, 2)
	net_params = get_params(opt_state)
	xx = np.stack(np.meshgrid(np.linspace(-1.5, 2.0, 50), np.linspace(-1.5, 2.0, 50)), axis=-1).reshape(-1, 2)
	scores = net_apply(net_params, xx)
	scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
	scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)

	clear_output(True)

	plt.quiver(*xx.T, *scores_log1p.T, width=0.002, color='green')
	plt.xlim(-1.5, 2.0)
	plt.ylim(-1.5, 2.0)
	plt.show()

	print("displaying gradients...")
	plt.figure(figsize=[16, 16])

	net_params = get_params(opt_state)
	xx = np.stack(np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50)), axis=-1).reshape(-1, 2)
	scores = net_apply(net_params, xx)
	scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
	scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)

	plt.quiver(*xx.T, *scores_log1p.T, width=0.002, color='green')
	plt.scatter(*sample_batch(10_000).T, alpha=0.25)
	plt.show()


def main():

	net_init, net_apply = stax.serial(
	    stax.Dense(128), stax.Softplus,
	    stax.Dense(128), stax.Softplus,
	    stax.Dense(2),
	)

	opt_init, opt_update, get_params = optimizers.adam(1e-3)

	out_shape, net_params = net_init(jax.random.PRNGKey(seed=42), input_shape=(-1, 2))
	opt_state = opt_init(net_params)

	loss_history = []

	print("Training...")

	train_step = get_train_step(opt_update, get_params, net_apply)

	for i in range(2000):
	    x = sample_batch(size=128)
	    loss, opt_state = train_step(i, opt_state, x)
	    loss_history.append(loss.item())

	print("Training Finished...")

	plot_gradients(loss_history, opt_state, get_params, net_params, net_apply)


if __name__ == '__main__':
	main()
