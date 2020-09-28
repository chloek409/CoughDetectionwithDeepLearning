import math


def sigmoid(x):
    # y = 1/(1+e^(-x))
    y = 1.0/(1+math.exp(-x))
    return y


def activate(inputs, weights):
    # perform net input
    h = 0
    for x, weight in zip(inputs, weights):
        # h=h1w1 + h2w2+ h323 + ...
        h += x*weight

    # perform activation
    return sigmoid(h)

if __name__ == "__main__":
    inputs = [.5, .3, .2]
    weights = [.4, .7, .2]
    output = activate(inputs, weights)
    print(output)