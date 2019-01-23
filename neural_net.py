#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse

from utils.csv_utils import split_matrix_into_training_set, import_plain_data_from_csv


def sigmoid(x: np.ndarray):
    """
    A function that works out sigmoid for x
    :param x: a numpy array or number to work out sigmoid
    :return:
    """
    return 1 / (1 + np.exp(-x))


def derivatives_sigmoid(x: np.ndarray):
    """
    A function that works out the sigmoid derivation for x
    :param x: a numpy array or number to work out sigmoid prime
    :return: the value of the sigmoid derivative
    """
    return x * (1 - x)


class NeuralNet(object):
    """
    A class that represents a neural net
    """
    def __init__(self, inputs: np.ndarray, hidden: int, outputs: np.ndarray, error_threshold: float = 0.2,
                 learning_rate: float = 0.2, wh=None, wo=None, activation=sigmoid,
                 activation_derivation=derivatives_sigmoid) -> None:
        """
        A constructor that initializes the Neural net with the required parameters.
        :param inputs: a numpy array representing the inputs
        :param hidden: an int representing the size of the hidden layer
        :param outputs: a numpy array representing the outputs
        :param error_threshold: a float representing the error threshold
        :param learning_rate: a float representing the learning rate
        :param wh: a numpy array representing the hidden layer weights
        :param wo: a numpy array representing the output layer weights
        :param activation: the activation function
        :param activation_derivation: the derivation of the activation function
        """
        self.activation = activation
        self.activation_derivation = activation_derivation

        self.inputs = inputs
        self.hidden = hidden
        self.targets = outputs

        self.error_threshold = error_threshold
        self.learning_rate = learning_rate

        if wh is None:
            self.wh = np.random.random_sample((inputs[0].size, hidden))
        else:
            self.wh = wh
        if wo is None:
            self.wo = np.random.random_sample((hidden, outputs[0].size))
        else:
            self.wo = wo

    def __round_nn_output__(self, value: int):
        """
        A helper function that rounds a value using the error threshold as a midpoint
        :param value: the value to round
        :return: 0 IFF value < error threshold ELSE 1
        """
        return 0 if value <= self.error_threshold else 1

    def __calc_bad_facts__(self, errors: np.ndarray) -> int:
        """
        A helper function that calculates the amount of bad facts in an epoch.
        :param errors: a numpy array representing the errors
        :return: an int representing the bad facts
        """
        bad_facts = 0
        for error in errors:
            bad_facts += len(list(filter(lambda x: (abs(x) >= self.error_threshold), error)))
        return bad_facts

    def feedforward(self, inp: np.ndarray = None) -> (np.ndarray, np.ndarray):
        """
        A feed forward method that allows the neural net to 'think'.
        :param inp: a numpy array representing the inputs
        :return: a tuple representing the output of the hidden and final output
        """
        if inp is None:
            inp = self.inputs
        net_h = np.dot(inp, self.wh)
        out_h = self.activation(net_h)
        net_o = np.dot(out_h, self.wo)
        out_o = self.activation(net_o)
        return out_h, out_o

    def train(self, epoch: int = 1000) -> list:
        """
        A method responsible for training the neural network.
        :param epoch: an int representing the number of iterations over the training data set
        :return: a list of epochs vs bad facts
        """
        bad_facts = []
        for i in range(epoch):
            out_h, out_o = self.feedforward(self.inputs)
            error = self.targets - out_o
            d_output = error * self.activation_derivation(out_o)

            error_hidden_layer = d_output.dot(self.wo.T)
            d_hidden = self.activation_derivation(out_h) * error_hidden_layer

            layer1_adjustment = self.inputs.T.dot(d_hidden)
            layer2_adjustment = out_h.T.dot(d_output)

            self.wh += self.learning_rate * layer1_adjustment
            self.wo += self.learning_rate * layer2_adjustment

            bad_facts.append(self.__calc_bad_facts__(error))
        return bad_facts

    def get_confusion_matrix(self, x_verify: np.ndarray, y_verify: np.ndarray) -> (np.ndarray, tuple):
        """
        A helper function that feeds forward the inputs, verifies their output and returns the neural net's accuracy.
        :param x_verify: A numpy 2d array of the inputs
        :param y_verify: A numpy 2d array of the expected outputs
        :return: a numpy array representing the confusion matrix and a tuple (recall, specificity, type_one_error,
        type_two_error)
        """
        test_neural_net = NeuralNet(inputs=x_verify, hidden=self.hidden, outputs=y_verify, wh=self.wh, wo=self.wo)
        round_output = np.vectorize(self.__round_nn_output__)
        out_o = round_output(test_neural_net.feedforward(x_verify)[1])
        errors = int(np.sum(out_o == y_verify))
        true_pos = np.sum(np.logical_and(out_o == 1, y_verify == 1))
        true_negative = np.sum(np.logical_and(out_o == 0, y_verify == 0))
        false_positive = np.sum(np.logical_and(out_o == 1, y_verify == 0))
        false_negative = np.sum(np.logical_and(out_o == 0, y_verify == 1))
        recall = true_pos + true_pos / true_pos + false_negative
        specificity = true_negative / (false_positive + true_negative)
        precision = true_pos / (true_pos + false_positive)
        type_one_error = false_positive / (false_positive + true_negative)
        type_two_error = false_positive / (false_positive + true_negative) if false_positive != 0 else 0
        rates = (recall, specificity, precision, type_one_error, type_two_error)
        return np.array([[true_pos, false_negative], [false_positive, true_negative]], np.int32), rates


if __name__ == '__main__':
    # Load boolean function dataset (simple_problem, normal_problem, hard_problem)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='normal_problem',
                        help='The data set to use: simple_problem, normal_problem, hard_problem. '
                             + 'Default: normal_problem')
    parser.add_argument('--epochs', default='1000',
                        help='The amount of epochs the training algorithm will iterate through. Default: 1000')
    args = parser.parse_args()

    training_data, verification_data = split_matrix_into_training_set(import_plain_data_from_csv(args.dataset))
    X_train = np.asarray([row[:5] for row in training_data], dtype=int)
    y_train = np.asarray([row[5:] for row in training_data], dtype=int)

    neural_net = NeuralNet(X_train, 4, y_train)
    # Neural net initialization
    total_bad_facts = neural_net.train(epoch=int(args.epochs))
    plt.plot(total_bad_facts)
    plt.ylabel('bad facts')
    plt.show()
    print(f"Bad Facts on Last Epoc: {total_bad_facts[len(total_bad_facts) - 1]}")

    # Verification
    X_verif = np.asarray([row[:5] for row in verification_data], dtype=int)
    Y_verif = np.asarray([row[5:] for row in verification_data], dtype=int)
    matrix, rates = neural_net.get_confusion_matrix(X_verif, Y_verif)
    print(f"Confusion matrix: \n {matrix}")
    print(f"Recall {rates[0]}, Specificity {rates[1]}, Precision {rates[2]},"
          + f" Type 1 error {rates[3]}, Type 2 error {rates[4]}")
