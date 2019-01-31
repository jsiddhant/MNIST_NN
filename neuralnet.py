import numpy as np
import pickle
import matplotlib.pyplot as plt

config = {}
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid'  # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 10  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0.0001  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001  # Learning rate of gradient descent algorithm


def softmax(x):
    """
    Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
    """
    a_x = x - x.max(axis=1).reshape(-1, 1)
    output = np.exp(a_x) / np.sum(np.exp(a_x), axis=1).reshape(-1, 1)
    return output


def load_data(fname):
    """
    Write code to read the data and return it as 2 numpy arrays.
    Make sure to convert labels to one hot encoded format.
    """
    f = open(fname, 'rb')
    data = pickle.load(f)
    images = data[:, :784]
    labels = np.zeros((data.shape[0], 10))
    labels[np.arange(0, labels.shape[0]), data[:, 784:785].astype(int).flatten()] = 1

    return images, labels


class Activation:
    def __init__(self, activation_type="sigmoid"):
        self.activation_type = activation_type
        self.x = None  # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.

    def forward_pass(self, a):
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.relu(a)

    def backward_pass(self, delta):
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        output = 1 / (1+np.exp(-x))
        return output

    def tanh(self, x):
        """
        Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
        """
        # (ex - e - x) / (ex + e - x)
        self.x = x
        output = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return output

    def relu(self, x):
        """
        Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        output = x * (x > 0)
        return output

    def grad_sigmoid(self):
        """
        Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        # Calculate g'(x)
        a = self.x
        g_a = 1 / (1 + np.exp(-a))
        grad = g_a * (1 - g_a)

        return grad

    def grad_tanh(self):
        """
        Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
        """
        a = self.x
        tanh_a = (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
        grad = 1 - (tanh_a * tanh_a)

        return grad

    def grad_ReLU(self):
        """
        Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        a = self.x
        grad = 1 * (a > 0)
        return grad


class Layer():
    def __init__(self, in_units, out_units):
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)  # Weight matrix
        self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
        self.d_w_old = None
        self.d_b_old = None
        self.x = None  # Save the input to forward_pass in this
        self.a = None  # Save the output of forward pass in this (without activation)
        self.d_x = None  # Save the gradient w.r.t x in this #dj/da #delta -> grad_act * d_x
        self.d_w = None  # Save the gradient w.r.t w in this #delta*input.T * delta
        self.d_b = None  # Save the gradient w.r.t b in this #delta

    def forward_pass(self, x):
        """
        Write the code for forward pass through a layer. Do not apply activation function here.
        """
        self.x = x
        output = x.dot(self.w) + self.b
        # output = self.w.T.dot(x.T)+ self.b.T
        self.a = output
        return self.a

    def backward_pass(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        """

        self.d_x = delta.dot(self.w.T)
        self.d_w_old = calc_momentum(self.d_w_old, self.d_w, self.w.shape)
        self.d_b_old = calc_momentum(self.d_b_old, self.d_b, self.b.shape)
        self.d_w = self.x.T.dot(delta)
        self.d_b = np.sum(delta, axis=0)

        return self.d_x


class Neuralnetwork():
    def __init__(self, config):
        self.layers = []
        self.x = None  # Save the input to forward_pass in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward_pass in this variable
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def forward_pass(self, x, targets=None):
        """
        Write the code for forward pass through all layers of the model and return loss and predictions.
        If targets == None, loss should be None. If not, then return the loss computed.
        """

        self.x = x
        self.targets = targets
        regular_loss = 0
        for layer in self.layers:
            x = layer.forward_pass(x)
            # regular_loss = regular_loss + config['L2_penalty']*np.linalg.norm(layer.w)
            # Momentum Here
            if isinstance(layer, Layer):
                regular_loss = regular_loss + np.linalg.norm(layer.w)

        self.y = softmax(x)
        loss = None if targets is None else self.loss_func(self.y, targets) + config['L2_penalty'] * regular_loss
        return loss, self.y

    def loss_func(self, logits, targets):
        '''
        find cross entropy loss between logits and targets
        '''
        eps = 10e-12
        prediction = np.clip(targets, eps, 1. - eps)
        output = -np.mean(logits * np.log(prediction + 1e-9))

        return output

    def backward_pass(self):
        '''
        implement the backward pass for the whole network.
        hint - use previously built functions.
        '''
        delta = self.targets - self.y
        for layer in reversed(self.layers):
            delta = layer.backward_pass(delta)


def calc_momentum(old_grad, new_grad, shape, gamma = config['momentum_gamma']):

    if new_grad is None:
        if old_grad is None:
            return np.zeros(shape)
        else:
            return gamma * old_grad
    else:
        return gamma * (old_grad + new_grad)


def trainer(model, X_train, y_train, X_valid, y_valid, config):
    """
    Write the code to train the network. Use values from config to set parameters
    such as L2 penalty, number of epochs, momentum, etc.
    """

    train_loss = []
    validation_loss = []
    train_acc = []
    valid_acc = []
    epoch = config['epochs']
    B = config['batch_size']
    learning_rate = config['learning_rate'] # 0.01 times slower learning rate required for ReLU
    regularization = config['L2_penalty']
    # momentum = config['momentum_gamma']
    N = X_train.shape[0]
    id = np.arange(0, N)
    early_stop_idx = 0
    prev_loss_valid = 0

    for i in range(0, epoch):
        np.random.shuffle(id)
        for l in range(0, int((N + B -1) / B)):
            idx = id[B * l : min(B * (l + 1), N)]

            loss, y = model.forward_pass(X_train[idx, :], targets=y_train[idx, :])
            model.backward_pass()
            for layer in model.layers:
                if isinstance(layer, Layer):
                    layer.w = layer.w + learning_rate * layer.d_w - 2*learning_rate*regularization*layer.w \
                              + (learning_rate * layer.d_w_old if config['momentum'] else 0)
                    layer.b = layer.b + learning_rate * layer.d_b \
                              + (learning_rate * layer.d_b_old if config['momentum'] else 0)

        print('Epoch: ' + str(i) + ' Loss: ' + str(loss))  # Random Pass

        loss_valid, valid_pred = model.forward_pass(X_valid, y_valid)
        train_loss.append(loss)
        validation_loss.append(loss_valid)
        train_acc.append(test(model, X_train, y_train, config))
        valid_acc.append(test(model, X_valid, y_valid, config))
        if i > 0 and config['early_stop'] and loss_valid > prev_loss_valid:
            early_stop_idx += 1
            if early_stop_idx == 5:
                print("Early Stop due to overfitting in epoch: " + str(i))
                break
        else:
            early_stop_idx = 0
            prev_loss_valid = loss_valid

    print('Training Accuracy: ' + str(train_acc[-1]))
    print('Validation Accuracy: ' + str(valid_acc[-1]))

    fig = plt.figure()
    create_train_plot(epoch, train_loss, validation_loss, 'Loss and Accuracy: ' + config['activation'], label1='Training Loss', label2='Validation Loss')
    create_train_plot(epoch, train_acc, valid_acc, 'Loss and Accuracy: ' + config['activation'], label1='Training Accuracy', label2='Validation Accuracy')
    print('-------------------------Training Finished---------------------------')
    print('Testing Accuracy: ' + str(test(model, X_test, y_test, config)))
    plt.show()


def test(model, X_test, y_test, config):
    """
    Write code to run the model on the data passed as input and return accuracy.
    """
    _, y_model = model.forward_pass(X_test)
    accuracy = np.mean(np.argmax(y_test, axis=1) == np.argmax(y_model, axis=1))
    return accuracy


def create_train_plot(epochs, train_err, hold_err, title, label1 = 'Training', label2 = 'Validation'):


    plt.suptitle(title)
    ep = [i+1 for i in range(0, epochs)]
    train_err = np.asarray(train_err)
    hold_err = np.asarray(hold_err)

    plt.plot(ep, train_err, '-', label=label1)
    plt.plot(ep, hold_err, '--', label=label2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')



if __name__ == "__main__":

    train_data_fname = 'MNIST_train.pkl'
    valid_data_fname = 'MNIST_valid.pkl'
    test_data_fname = 'MNIST_test.pkl'

    model = Neuralnetwork(config)
    X_train, y_train = load_data(train_data_fname)
    X_valid, y_valid = load_data(valid_data_fname)
    X_test, y_test = load_data(test_data_fname)
    trainer(model, X_train, y_train, X_valid, y_valid, config)
    test_acc = test(model, X_test, y_test, config)
