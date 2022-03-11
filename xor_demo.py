
import jax
import jax.numpy as np
import optax
# Current convention is to import original numpy as "onp"
import numpy as onp
from scipy.ndimage.filters import uniform_filter1d


KEY = jax.random.PRNGKey(5)


# Sigmoid nonlinearity
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Computes our network's output
def net(x, params):
    hidden = sigmoid(np.dot(x, params['hidden']) + params['hidden_bias'])
    return np.dot(hidden, params['output']) + params['output_bias']


# Cross-entropy loss
# def loss(params, x, y):
#     out = net(params, x)
#     cross_entropy = -y * np.log(out) - (1 - y)*np.log(1 - out)
#     return cross_entropy

def loss(params: optax.Params, batch: np.ndarray, labels: np.ndarray) -> np.ndarray:
    y_hat = net(batch, params)

    # optax also provides a number of common loss functions.
    loss_value = optax.sigmoid_binary_cross_entropy(y_hat, labels).sum(axis=-1)

    return loss_value.mean()


# Utility function for testing whether the net produces the correct
# output for all possible inputs
def test_all_inputs(inputs, params):
    predictions = [int(net(params, inp) > 0.5) for inp in inputs]
    for inp, out in zip(inputs, predictions):
        print(inp, '->', out)
    return (predictions == [onp.bitwise_xor(*inp) for inp in inputs])


def initial_params(n_hidden_nodes: int):
    return dict(
        hidden=jax.random.normal(key=KEY, shape=(2, n_hidden_nodes)),  #w1
        hidden_bias=jax.random.normal(key=KEY, shape=(n_hidden_nodes,)),  # b1
        output=jax.random.normal(key=KEY, shape=(n_hidden_nodes,)),  # w2
        output_bias=jax.random.normal(key=KEY, shape=()),  # b2
    )


def compose_measures(loss_fn, loss_history, grad_history, param_history, gradients, parameters, x_sample, y_sample):
    grad_vec = np.hstack([g.flatten() for g in gradients])
    temp = [loss_fn(parameters, xt, yt) for xt, yt in
            zip(x_sample, y_sample)]  # likely that a tensor-based solution exists
    loss_history.append(np.mean(np.asarray(temp)))
    grad_history.append(np.linalg.norm(grad_vec, 2))
    param_history.append(reshape_params(parameters))


def reshape_params(parameters):
    out = []
    for (key, param) in parameters.items():
        try:
            out.append(param.flatten())
        except AttributeError:
            out.append(param)
    return np.hstack(out)


class OptaxGD:

    def __init__(self, training_data, labels):
        self.training_data = training_data
        self.labels = labels

        self.loss_hist = []
        self.grad_hist = []
        self.param_hist = []

        self._opt_params = None

    def fit(self, parameters: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
        opt_state = optimizer.init(parameters)

        @jax.jit
        def step(parameters, opt_state, batch, labels):
            loss_value, gradients = jax.value_and_grad(loss)(parameters, batch, labels)
            updates, opt_state = optimizer.update(gradients, opt_state, parameters)
            parameters = optax.apply_updates(parameters, updates)

            return parameters, opt_state, loss_value, gradients

        # for now assume full batch

        for i, (batch, labels) in enumerate(zip(self.training_data, self.labels)):
            parameters, opt_state, loss_value, gradients = step(parameters, opt_state, batch, labels)

            self.grad_hist.append(reshape_params(gradients))
            self.param_hist.append(reshape_params(parameters))
            self.loss_hist.append(loss_value)

            if i % 100 == 0:
                print(f'step {i}, loss: {loss_value}')

            self._opt_params = parameters

        return parameters

    def predict(self, x_input):
        x_temp = net(x_input, self._opt_params)
        return jax.nn.sigmoid(x_temp)

    def get_metadata(self):
        grad_norm = [onp.linalg.norm(p) for p in self.grad_hist]
        return dict(loss=np.array(self.loss_hist),
                    grad=np.array(self.grad_hist),
                    grad_norm=grad_norm,
                    parameters=np.array(self.param_hist)
                    )


def homecooked_gd(train_x, train_y, param_init, learning_rate, show_plot=False):

    n = 0
    param_hist = []
    params = param_init
    while n < 5000:
        n += 1
        # The call to loss_grad remains the same!
        grads = loss_grad(params, train_x, train_y)
        compose_measures(loss, loss_hist, grad_hist, param_hist, grads, params, train_x, train_y)
        # Note that we now need to average gradients over the batch (SA: didnt write this, not sure this is right/most efficent??)
        params = [param - learning_rate * np.mean(grad, axis=0)
                  for param, grad in zip(params, grads)]
        if not n % 500:
            print('Iteration {}'.format(n))
            # if test_all_inputs(inputs, params):
            #     break
            learning_rate /= 2

    if show_plot:
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(np.asarray(loss_hist))
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('Epoch')
        ax[1].plot(np.asarray(grad_hist))
        ax[1].set_ylabel('Grad norm')
        ax[1].set_xlabel('Epoch')
        ax[2].plot(np.asarray(param_hist))
        ax[2].set_ylabel('Param traj')
        ax[1].set_xlabel('Epoch')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Stochastic gradient descent learning rate
    learning_rate = .1
    # All possible inputs
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    N_HIDDEN = 20
    PARAMS = initial_params(N_HIDDEN)

    loss_grad = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0), out_axes=0))

    batch_size = 500
    num_train_steps = 5000
    loss_hist = []
    grad_hist = []
    n = 0
    # Generate a batch of inputs
    x = inputs[jax.random.choice(key=KEY, a=inputs.shape[0], shape=(num_train_steps, batch_size))]
    y = onp.bitwise_xor(x[:, :, 0], x[:, :, 1])
    # learning rate schedule
    schedule = optax.piecewise_constant_schedule(init_value=.2,
                                                 boundaries_and_scales={400: 1e-2, 800: 1e-3, 1200: 1e-5})
    # if you want to use the schedule, set learning_rate=schedule below
    opti = optax.adam(learning_rate=learning_rate)
    ogd = OptaxGD(x, y)
    ogd.fit(parameters=PARAMS, optimizer=opti)
    META = ogd.get_metadata()
    preds = ogd.predict(inputs)
    print(inputs, preds)

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(META['parameters'])
    ax[0].set_ylabel('Param. traj')
    ax[1].plot(META['grad'], linewidth=.1)
    ax[1].set_ylabel('gradient')
    ax[2].plot(META['grad_norm'])
    ax[2].plot(uniform_filter1d(META['grad_norm'], size=100))
    ax[2].set_ylabel('grad norm')
    ax[3].plot(META['loss'])
    ax[3].plot(uniform_filter1d(META['loss'], size=100))
    ax[3].set_ylabel('loss')

    ax[3].set_xlabel('Epoch')
    plt.tight_layout()
    plt.show()
