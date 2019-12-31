import jax
import jax.numpy as np
import numpy as onp

from functools import partial
from jax import random, jit, grad, vmap, pmap, jacfwd, jacrev 

key = random.PRNGKey(0)
x = random.normal(key, (10,))


########## grad ##########
# def tanh(x):
# 	y = np.exp(-2.0 * x)
# 	return np.sum((1.0 - y) / (1.0 + y))

# grad_tanh = jit(grad(tanh))
# print(grad_tanh(x))
##########################


########## higher order ##########
# def hessian(fun):
#   return jit(jacfwd(jacrev(fun)))

# hess_tanh = hessian(tanh)
# print(hess_tanh(1.2))
##########################


########## vmap ########## 
# vv = lambda x, y: np.vdot(x, y)  #  ([a], [a]) -> []
# mv = vmap(vv, (0, None), 0)      #  ([b,a], [a]) -> [b]      
# mm = vmap(mv, (None, 1), 1)      #  ([b,a], [a,c]) -> [b,c] 

# mat = np.arange(9).reshape([3,3])
# print(mm(mat, mat))
# per_example_gradients = vmap(partial(grad(loss), params))(inputs, targets)
##########################


########## pmap ########## 
# @pmap
# def f(x):
#   y = np.sin(x)
#   @pmap
#   def g(z):
#     return np.cos(z) * np.tan(y.sum()) * np.tanh(x).sum()
#   return grad(lambda w: np.sum(g(w)))(x)

# print(f(np.ones((1, 1))))
##########################


########## Asynchronous dispatch ########## 
##########################

# .block_until_ready()