import itertools
import timeit

from jax import grad, jit
import jax.numpy as jnp


class GradientDescent:

    def __init__(self, loss_fn: callable, learning_rate: float, track_params=False):
        """
        Gradient descent with a fixed learning rate
        :param loss_fn: callable, the function to be optimized
        :param learning_rate: float, the step size
        :param track_params: bool, whether to keep track of the parameter history or not
        """
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.grad = jit(grad(self.loss_fn))

        self.track_params = track_params
        if self.track_params:
            self.param_history = []

    def step(self, params: jnp.ndarray):
        """
        Perform vanilla gradient descent
        :param params:
        :return:
        """
        return params - self.learning_rate * self.grad(params)

    def n_step(self, params_0: jnp.ndarray, n_step: int):
        """
        Convenience method for doing n-step gradient descent
        :param params_0:
        :param n_step:
        :return:
        """
        count = 0
        start = timeit.default_timer()
        while count <= n_step:
            params_0 = self.step(params_0)
            count += 1

            if self.track_params:
                self.param_history.append(params_0)
        print('Elapsed time: {} seconds'.format(timeit.default_timer() - start))

        return params_0

    def get_history(self):
        """
        Get the history of the parameters across the training
        :return:
        """
        if self.track_params:
            return jnp.asarray(self.param_history)
        else:
            raise AttributeError('Need to set track_params to True in init')


def loss_example(theta: jnp.ndarray):
    """
    Loss function from
    :param theta: (2,) array that parameterizes the loss function
    :return: the value of the loss function
    """
    return 100*jnp.sin(theta[0])*jnp.sin(theta[1])


def generate_level_sets(x_lim: list, y_lim: list, loss_function: callable, n_grid: int):
    """
    Generate level sets for visualization purposes
    :param x_lim: list, [x_min, x_max]
    :param y_lim: list, [y_min, y_max]
    :param loss_function: callable, the loss function of interest
    :param n_grid: int, the number of grid points per dimension
    :return: tuple, resulting X,Y,Z
    """

    x = jnp.linspace(x_lim[0], x_lim[1], n_grid)
    y = jnp.linspace(y_lim[0], y_lim[1], n_grid)
    x_mat, y_mat = jnp.meshgrid(x, y)
    z = []
    # this is stupid slow, go to a vectorized format
    for x_t, y_t in itertools.product(x, y):
        p_temp = jnp.array([x_t, y_t])
        z.append(loss_function(p_temp))

    z_mat = jnp.asarray(z).reshape((n_grid, n_grid))

    return x_mat, y_mat, z_mat


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    LR = .015
    PARAMS_0 = jnp.array([1, 1.5])
    k_viz = 500  # only show last k data points in scatter plot on level set graph
    gd = GradientDescent(loss_fn=loss_example, learning_rate=LR, track_params=True)

    gd.n_step(PARAMS_0, n_step=1000)
    PARAM_HIST = gd.get_history()
    PARAM_HIST_SEMI_CONV = PARAM_HIST[k_viz:, :]

    # Visualize the evolution of the parameters
    X, Y, Z = generate_level_sets([-4, .5], [-2, 4.5], loss_example, n_grid=50)
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].plot(PARAM_HIST[:, 0], label=r'$\theta_1$')
    ax[0, 0].plot(PARAM_HIST[:, 1], label=r'$\theta_2$')
    ax[0, 0].set_title('Parameter Trajectories')
    ax[0, 0].set_xlabel('Iteration')
    ax[0, 0].set_ylabel('Value')
    ax[0, 0].set_xlim([0, 50])
    ax[0, 0].legend()

    ax[1, 0].scatter(PARAM_HIST[:k_viz, 0], PARAM_HIST[:k_viz, 1], c='g', alpha=0.1, label=r'$\theta_{k \leq 500}$')
    ax[1, 0].scatter(PARAM_HIST_SEMI_CONV[:, 0], PARAM_HIST_SEMI_CONV[:, 1], c='r', alpha=0.3, label=r'$\theta_{k>500}}$')
    ax[1, 0].scatter(PARAM_HIST[-1, 0], PARAM_HIST[-1, 1], c='b', label=r'$\theta_{final}$')
    ax[1, 0].contour(X, Y, Z)
    ax[1, 0].set_title('Parameter space with level sets')
    ax[1, 0].set_xlabel(r'$\theta_1$')
    ax[1, 0].set_ylabel(r'$\theta_2$')
    ax[1, 0].legend()

    # fig2, ax2 = plt.subplots(2, 1)
    bins = np.histogram(PARAM_HIST[:, 0], bins=100)[1]
    bins2 = np.histogram(PARAM_HIST[:, 1], bins=100)[1]
    ax[0, 1].hist(PARAM_HIST[:, 0], bins=bins, label='all k')
    ax[1, 1].hist(PARAM_HIST[:, 1], bins=bins2, label='all k')
    ax[0, 1].hist(PARAM_HIST_SEMI_CONV[:, 0], bins=bins, label='k > 500')
    ax[1, 1].hist(PARAM_HIST_SEMI_CONV[:, 1], bins=bins2, label='k > 500')

    ax[0, 1].set_title('Parameter histograms')
    # fig2.suptitle('Parameter histograms')
    for i in range(2):
        ax[i, 1].legend()
        ax[i, 1].set_ylabel('Frequency (count)')
    ax[0, 1].set_xlabel(r'$\theta_1$')
    ax[1, 1].set_xlabel(r'$\theta_2$')

    fig.suptitle('Parameters for learning rate={}'.format(LR), y=0.93, fontsize=14)
    plt.tight_layout()
    plt.show()
