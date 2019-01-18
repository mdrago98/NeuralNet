import numpy as np
import matplotlib.pyplot as plt

from CsvUtils import split_matrix_into_training_set, import_plain_data_from_csv


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivatives_sigmoid(x):
    return x * (1 - x)


def calc_bad_facts(errors, error_threshold):
    bad_facts = 0
    for error in errors:
        bad_facts += len(list(filter(lambda x: (abs(x) >= error_threshold), error)))
    return bad_facts


class NeuralNet(object):
    def __init__(self, inputs, hidden, outputs, error_threshold=0.2, learning_rate=0.2):
        self.wh = np.random.random_sample((inputs[0].size, hidden))
        self.wo = np.random.random_sample((hidden, outputs[0].size))

        self.error_threshold = error_threshold
        self.learning_rate = learning_rate

        self.targets = outputs
        self.inputs = inputs

    def feedforward(self, inp):
        """:return outO"""
        net_h = np.dot(inp, self.wh)
        out_h = sigmoid(net_h)
        net_o = np.dot(out_h, self.wo)
        out_o = sigmoid(net_o)
        return out_h, out_o

    def train(self, epoch=1000):
        bad_facts = []
        for i in range(epoch):
            out_h, out_o = self.feedforward(self.inputs)
            error = self.targets - out_o
            d_output = error * derivatives_sigmoid(out_o)

            error_hidden_layer = d_output.dot(self.wo.T)
            d_hidden = derivatives_sigmoid(out_h) * error_hidden_layer

            layer1_adjustment = self.inputs.T.dot(d_hidden)
            layer2_adjustment = out_h.T.dot(d_output)

            self.wh += layer1_adjustment
            self.wo += layer2_adjustment

            bad_facts.append(calc_bad_facts(error, self.error_threshold))
        return bad_facts

    def predict(self, inp):

        # Allocate memory for the outputs
        y = np.zeros([inp.shape[0], self.wo.shape[1]])

        # Loop the inputs
        for i in range(0, inp.shape[0]):
            y[i] = self.feedforward(inp[i])

        # Return the results
        return y


if __name__ == '__main__':
    # Digits dataset loading
    training_data, verification_data = split_matrix_into_training_set(import_plain_data_from_csv())
    X_train = np.asarray([row[:5] for row in training_data], dtype=int)
    y_train = np.asarray([row[5:] for row in training_data], dtype=int)

    # Neural Network initialization
    NN = NeuralNet(X_train, 4, y_train)
    bad_facts = NN.train()
    print(bad_facts)
    plt.plot(bad_facts)
    plt.ylabel('bad facts')
    plt.show()
    print(bad_facts[len(bad_facts) - 1])

    X_verif = np.asarray([row[:5] for row in verification_data], dtype=int)
    Y_verif = np.asarray([row[5:] for row in verification_data], dtype=int)
    NN.inputs = verification_data
    # NN predictions
    # y_predicted = NN.predict(X_test)
    #
    # # Metrics
    # y_predicted = np.argmax(y_predicted, axis=1).astype(int)
    # y_test = np.argmax(y_test, axis=1).astype(int)
    #
    # print("\nClassification report for classifier:\n\n%s\n"
    #       % (metrics.classification_report(y_test, y_predicted)))
    # print("Confusion matrix:\n\n%s" % metrics.confusion_matrix(y_test, y_predicted))
