from CsvUtils import import_plain_data_from_csv, split_matrix_into_training_set
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivatives_sigmoid(x):
    return x * (1 - x)


input_layer_size = 5
hidden_layer_size = 4
output_layer_size = 3

epoch = 2500
learning_rate = 0.2
error_threshold = 0.2

training_data, verification_data = split_matrix_into_training_set(import_plain_data_from_csv())
inputs = np.asarray([row[:5] for row in training_data], dtype=int)
outputs = np.asarray([row[5:] for row in training_data], dtype=int)

wh = np.random.random_sample((input_layer_size, hidden_layer_size))
bh = np.random.random_sample((1, hidden_layer_size))
wout = np.random.random_sample((hidden_layer_size, output_layer_size))
bout = np.random.random_sample((1, output_layer_size))

output = sigmoid(outputs)
bad_facts = []


def calc_bad_facts(errors):
    bad_facts = 0
    for error in errors:
        bad_facts += len(list(filter(lambda x: (abs(x) >= error_threshold), error)))
    return bad_facts


def forward_propogation(inputs, wh):
    hidden_layer_input_without_bias = np.dot(inputs, wh)
    hidden_layer_input_added_with_bias = hidden_layer_input_without_bias + bh
    hidden_layer_activ = sigmoid(hidden_layer_input_added_with_bias)
    output_layer_input_without_bias = np.dot(hidden_layer_activ, wout)
    output_layer_with_bias = output_layer_input_without_bias + bout
    output = sigmoid(output_layer_with_bias)
    return output, hidden_layer_activ


def back_propagation(wout, wh):
    error = outputs - output
    bad_facts.append(calc_bad_facts(error))
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hidden_layer_activ)
    output_delta = error * slope_output_layer
    hidden_layer_error = output_delta.dot(wout.T)
    hidden_layer_delta = hidden_layer_error * slope_hidden_layer
    wout += hidden_layer_activ.T.dot(output_delta) * learning_rate
    wh += inputs.T.dot(hidden_layer_delta) * learning_rate
    return wout, wh,


def feed_forward(input, wh, wo):
    net_h = np.dot(input, wh)
    out_h = sigmoid(net_h)
    net_o = np.dot(out_h, wo)
    return sigmoid(net_o)


def delta(out, target):
    return out(1 - out) * (target - out)


for i in range(epoch):
    # forward
    output, hidden_layer_activ = forward_propogation(inputs, wh)
    # back prop
    wout, wh = back_propagation(wout, wh)

plt.plot(bad_facts)
plt.ylabel('bad facts')
plt.show()
print(bad_facts[len(bad_facts) - 1])
