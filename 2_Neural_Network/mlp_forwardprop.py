import numpy as np


class MLP:

    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):

        # 3개의 속성 정의
        self.num_inputs = num_inputs  # number of input neurons
        self.num_hidden = num_hidden  # list of number of neurons in each hidden layer
        self.num_outputs = num_outputs  # number of output neurons

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        # returns list of number of neurons in each layer
        # thus len(layers) is the number of layers

        # initiate random weights
        self.weights = []  # 행렬들의 list. 항상 갯수가 layers-1 일 것
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            # w: a 2D matrix (현재 layer의 뉴런 x 다음 layer의 뉴런 수) with random values
            self.weights.append(w)

    def forward_propagate(self, inputs):  # 1.net input => 2.activation

        activations = inputs

        for w in self.weights:
            # calculate net inputs
            net_inputs = np.dot(activations, w)  # np를 활용하면 행렬곱이 매우 간단!

            # calculate activations
            activations = self._sigmoid(net_inputs)

        print("actvations: ", activations)
        return activations


    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))


if __name__ == "__main__":

    # create MLP
    mlp = MLP()

    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)
    print("The network inputs: ", inputs)

    # perform forward propagation
    outputs = mlp.forward_propagate(inputs)

    # print the results
    print("The netwok outputs: {}".format(outputs))