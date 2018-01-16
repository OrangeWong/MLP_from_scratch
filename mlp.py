import numpy as np


class MLP:
    """The class is the implementation of multi layer perception (or neural network)."""
    def __init__(self, hidden_layer_sizes=(5,), activation=np.tanh, learning_rate=0.1, max_iter=2000, tol=0.05,
                 momentum=0.9, random_seed=1985):
        """
        Args:
            hidden_layer_sizes(tuple): A tuple represents the number of neurons for each hidden layers. Default: (5,).
            activation (callable): This is the activation function. Default: np.tanh.
            learning_rate (float): The learning used in back-propagation. Default: 0.1.
            max_iter (int): The maximum number of iterations allowed. Default: 2000.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.momentum = momentum
        self.random_seed = random_seed
        self.weights = None
        self._product_sums = None
        self._outputs = None
        self._derivatives = None
        self._errors = None
        self._inputs = None
        self._velocity = None

    @staticmethod
    def d_tanh(x):
        """Derivative function of tanh."""
        return 1 - (np.tanh(x)) ** 2

    def _initiate_weights(self, input_dimension):
        """This function initiates the weights in the neural network based on the parameters passed to the class."""
        weights = []
        for layer in range(len(self.hidden_layer_sizes) + 1):
            # When layer = 0, this is the weight connecting the input to the first hidden layer
            if layer == 0:
                temp = np.random.rand(input_dimension + 1, self.hidden_layer_sizes[0]) - 0.5
                weights.append(temp)
            # When layer = self.n_hidden_layers, then this weight connects the last hidden layer and the output layer.
            elif layer == len(self.hidden_layer_sizes):
                temp = np.random.rand(self.hidden_layer_sizes[len(self.hidden_layer_sizes)-1] + 1, 1) - 0.5
                weights.append(temp)
            else:
                temp = np.random.rand(self.hidden_layer_sizes[layer-1]+1, self.hidden_layer_sizes[layer]) - 0.5
                weights.append(temp)
        self.weights = weights

    def _feed_forward(self, x):
        """Feed-forward the data from the input layer to the output layer via the hidden layers

        Args:
            x (array): The array represents the data of a sample
        """
        self._outputs = []
        self._product_sums = []
        self._inputs = np.append(x, 1).reshape(1, -1)
        # feed the signal forward
        for layer in range(len(self.hidden_layer_sizes) + 1):
            if layer == 0:
                product_sum = self._inputs @ self.weights[layer]
            else:
                product_sum = self._outputs[layer - 1] @ self.weights[layer]
            self._product_sums.append(product_sum)

            if layer == len(self.hidden_layer_sizes):
                output = product_sum
            else:
                output = np.append(self.activation(product_sum), 1).reshape(1, -1)
            self._outputs.append(output)

    def predict(self, X):
        """This function calculates the prediction based on the calculated weights"""
        y_predict = []
        for observation in range(X.shape[0]):
            self._feed_forward(X[observation, :])
            y_predict.append(np.asscalar(self._outputs[-1]))
        return y_predict

    @staticmethod
    def _calculate_mse(y_predict, y_true):
        """This function calculates the mean square error after feed-forwarding the network

        Args:
            y (array): The values of the true output.

        """
        mse = 0.5 * np.dot(y_predict - y_true, y_predict - y_true) / len(y_true)
        return mse

    def _back_propagation(self, y):
        """This function runs the back propagation to transmit the error from the output layer to the input layer via
        the hidden layers.

        Args:
            y (array): The data contains the true values.
        """
        y_predict = self._outputs[-1]
        self._errors = []
        self._derivatives = []
        for layer in range(len(self.hidden_layer_sizes), -1, -1):
            if layer == len(self.hidden_layer_sizes):
                error = (y_predict-y)
            else:
                error = self.d_tanh(self._product_sums[layer]) * \
                        (self._errors[len(self.hidden_layer_sizes) - layer - 1] @ self.weights[layer + 1][:-1, :].T)
            if layer == 0:
                derivative = self._inputs.T @ error
            else:
                derivative = self._outputs[layer - 1].T @ error
            self._derivatives.append(derivative)
            self._errors.append(error)

    def _update_weights(self):
        """This function uses the calculated results in back propagation to update the weights"""
        if self.momentum is not None:
            # initiate the velocity
            self._velocity = [np.zeros_like(v) for v in self.weights]
            for layer in range(len(self.hidden_layer_sizes) + 1):
                self._velocity[layer] = self.momentum * self._velocity[layer] \
                                        - self.learning_rate * self._derivatives[-1 - layer]
                self.weights[layer] += self._velocity[layer]
        else:
            for layer in range(len(self.hidden_layer_sizes) + 1):
                self.weights[layer] -= self.learning_rate * self._derivatives[-1 - layer]

    def _should_stop(self, mse, iters):
        if mse is None:
            return False
        else:
            if mse < self.tol:
                return True
            if iters > self.max_iter:
                return True
            return False

    def fit(self, X, y=None):
        """To fit the neural network using X and y.

        Args:
            X (array): The data stores the attributes.
            y (array): The data stores the labels.
        """
        np.random.seed(self.random_seed)
        self._initiate_weights(X.shape[1])
        iters = 0
        mse = None
        while not self._should_stop(mse, iters):
            iters += 1
            for observation in range(X.shape[0]):
                x_s, y_s = X[observation, :], y[observation]
                self._feed_forward(x_s)
                self._back_propagation(y_s)
                self._update_weights()
            indexes = np.random.permutation(X.shape[0])
            X, y = X[indexes, :], y[indexes]
            y_predict = self.predict(X)
            mse = self._calculate_mse(y_predict, y)
            print(iters, '-iters, mse: ', mse, 'predict: ', y_predict)


if __name__ == "__main__":
    import numpy as np

    testobject = MLP(hidden_layer_sizes=(2, ), tol=0.05, max_iter=5000,
                     random_seed=315, momentum=0.9)

    X = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    y = np.array([-0.5, 0.5, 0.5, -0.5])

    testobject.fit(X, y)

