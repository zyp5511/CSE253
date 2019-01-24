import numpy as np
import pickle
from os import listdir
from PIL import Image


config = {}
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = False# Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm

def softmax(x):
    """
    Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
    """
    return np.exp(x) / np.array([np.sum(np.exp(x), axis=1)]).T

def oneHot(Y_oh, max_val):
    """ Computes onehot.
    Input: Y_oh: list of number
          max_val: The max one-hot size
    Returns: 2D list correspind to the each label's one hot representation
    """
    result_oh = []
    for i in range(len(Y_oh)):
        onehot = [0] * (int(max_val)+1)
        onehot[int(Y_oh[i])] = 1
        result_oh.append(onehot)
    return np.array(result_oh)

def load_data(fname):
    """
    Write code to read the data and return it as 2 numpy arrays.
    Make sure to convert labels to one hot encoded format.
    fname : folder name
    """
    fname = 'data/' + fname
    training_data = pickle.load(open(fname, 'rb'), encoding='latin1')
    images = training_data[:,:784]
    labels = oneHot(training_data[:,784],9)
    print("Total number of images:", len(images), "and labels:", len(labels))

    return images, labels


class Activation:
    def __init__(self, activation_type = "sigmoid"):
        self.activation_type = activation_type
        self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.

    def forward_pass(self, a):
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

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
        return 1. / (1. + np.exp(-x))

    def tanh(self, x):
        """
        Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        return 2*self.sigmoid(2*x)-1.0

    def ReLU(self, x):
        """
        Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        return np.multiply(x,x > 0)

    def grad_sigmoid(self):
        """
        Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        grad = np.multiply(self.sigmoid(self.x), (1-self.sigmoid(self.x)))
        return grad

    def grad_tanh(self):
        """
        Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
        """
        grad = 1/(1+(self.tanh(self.x)**2))
        return grad

    def grad_ReLU(self):
        """
        Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        grad = np.ones(self.x.shape)
        grad[(self.x < 0)] = 0
        return grad


class Layer():
    def __init__(self, in_units, out_units):
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)  # Weight matrix
        self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
        self.x = None  # Save the input to forward_pass in this
        self.a = None  # Save the output of forward pass in this (without activation)
        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def forward_pass(self, x):
        """
        Write the code for forward pass through a layer. Do not apply activation function here.
        """
        self.x = x
        self.a = self.x.dot(self.w)+self.b
        return self.a

    def backward_pass(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        """
        lr = config['learning_rate']
        self.d_x = delta.dot(self.w.T)
        self.d_w = self.x.T.dot(delta)
        self.d_b = delta.sum(axis=0)
        return self.d_x

class Neuralnetwork():
    def __init__(self, config):
        self.layers = []
        self.x = None  # Save the input to forward_pass in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward_pass in this variable
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]) )
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def forward_pass(self, x, targets=None):
        """
        Write the code for forward pass through all layers of the model and return loss and predictions.
        If targets == None, loss should be None. If not, then return the loss computed.
        """
        self.x = x
        self.targets = targets
        layer_in = x;
        for layer in self.layers:
            layer_in = layer.forward_pass(layer_in)
        self.y = softmax(layer_in)
        if targets is not None:
            loss = self.loss_func(self.y,targets)
        else:
            loss = None
        return loss, self.y

    def loss_func(self, logits, targets):
        '''
        find cross entropy loss between logits and targets
        '''
        m = np.array(logits).shape[0]
        n = np.array(logits).shape[1]
        logits = np.array(logits, dtype=np.float)

        output = -(1.0 /(m * n)) * np.sum(np.multiply(targets, np.log(logits)))
        return output

    def backward_pass(self):
        '''
        implement the backward pass for the whole network.
        hint - use previously built functions.
        '''
        delta = (self.targets - self.y)
        #print(delta[3])
        #print("first y is")
        #print(self.y[3])
        #print("first target is")
        #print(self.targets[3])
        for l in reversed(self.layers):
            delta = l.backward_pass(delta)
        self.update_weight()

    def update_weight(self):
        '''
        implement the weight update for each layer
        :return: none
        '''
        lr = config['learning_rate']
        for l in self.layers:
            if isinstance(l, Layer):
                #print("pre-layer")
                #print((l.x.dot(l.w))[3])
                l.w = l.w + l.d_w*lr
                l.b = l.b + l.d_b*lr
                #print("layer")
                #print((l.x.dot(l.w))[3])


def trainer(model, X_train, y_train, X_valid, y_valid, config):
    """
    Write the code to train the network. Use values from config to set parameters
    such as L2 penalty, number of epochs, momentum, etc.
    """
    loss_train = []
    loss_valid = float('inf')
    for i in range(config['epochs']):
        for j in range(len(X_train)):
            start = j*config['batch_size']
            end = (j+1)*config['batch_size']
            if start >= len(X_train):
                break
            batch_x = X_train[start:end,:]
            batch_y = y_train[start:end,:]
            [loss_train_temp, prediction] = model.forward_pass(batch_x,batch_y)
            #print("train loss is ",loss_train_temp)
            loss_train.append(loss_train_temp)
            model.backward_pass()
            #print(accuracy(prediction,batch_y))
            #print(batch_y)
            #break
        [loss_valid_temp, prediction] = model.forward_pass(X_valid,y_valid)
        print(y_valid[1:5])
        print(prediction[1:5])
        print(accuracy(prediction, y_valid))
        line = "Interation" + str(i) + "training loss is " + str(loss_train_temp) + "validation loss " + str(loss_valid_temp)
        print(line)

        if config['early_stop'] == True:
            if loss_valid_temp < loss_valid:
                loss_valid = loss_valid_temp
            else:
                break

def accuracy(pred, t):
    return sum(pred.argmax(axis=1) == t.argmax(axis=1)) / pred.shape[0]

def test(model, X_test, y_test, config):
    """
    Write code to run the model on the data passed as input and return accuracy.
    """
    [_,prediction] = model.forward_pass(X_test)

    return accuracy(prediction,y_test)


if __name__ == "__main__":
    train_data_fname = 'MNIST_train.pkl'
    valid_data_fname = 'MNIST_valid.pkl'
    test_data_fname = 'MNIST_test.pkl'

    ### Train the network ###
    model = Neuralnetwork(config)
    X_train, y_train = load_data(train_data_fname)
    X_valid, y_valid = load_data(valid_data_fname)
    X_test, y_test = load_data(test_data_fname)
    trainer(model, X_train, y_train, X_valid, y_valid, config)
    test_acc = test(model, X_test, y_test, config)
    print("test acc is " + str(test_acc))

