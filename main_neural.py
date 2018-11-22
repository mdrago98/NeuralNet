from CsvUtils import import_plain_data_from_csv, split_matrix_into_training_set
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivatives_sigmoid(x):
    return x * (1 - x)


epoch = 2500
learning_rate = 0.2
error_threshold = 0.2

training_data, verification_data = split_matrix_into_training_set(import_plain_data_from_csv())
inputs = np.asarray([row[:5] for row in training_data], dtype=int)
outputs = np.asarray([row[5:] for row in training_data], dtype=int)

wh = np.random.random_sample((5, 4))
bh = np.random.random_sample((1, 4))
wout = np.random.random_sample((4, 3))
bout = np.random.random_sample((1, 3))

output = sigmoid(outputs)
bad_facts = []


def calc_bad_facts(errors):
    bad_facts = 0
    for error in errors:
        bad_facts += len(list(filter(lambda x: (abs(x) >= learning_rate), error)))
    return bad_facts


def forward_propogation(inputs, wh):
    hidden_layer_input_without_bias = np.dot(inputs, wh)
    hidden_layer_input_added_with_bias = hidden_layer_input_without_bias + bh
    hidden_layer_activ = sigmoid(hidden_layer_input_added_with_bias)
    output_layer_input_without_bias = np.dot(hidden_layer_activ, wout)
    output_layer_with_bias = output_layer_input_without_bias + bout
    output = sigmoid(output_layer_with_bias)
    return output, hidden_layer_activ


def back_propagation(wout, bout, wh, bh):
    error = outputs - output
    bad_facts.append(calc_bad_facts(error))
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hidden_layer_activ)
    output_delta = error * slope_output_layer
    hidden_layer_error = output_delta.dot(wout.T)
    hidden_layer_delta = hidden_layer_error * slope_hidden_layer
    wout += hidden_layer_activ.T.dot(output_delta) * learning_rate
    bout += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    wh += inputs.T.dot(hidden_layer_delta) * learning_rate
    bh += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate
    return wout, bout, wh, bh


for i in range(epoch):
    # forward
    output, hidden_layer_activ = forward_propogation(inputs, wh)
    # back prop
    wout, bout, wh, bh = back_propagation(wout, bout, wh, bh)

plt.plot(bad_facts)
plt.ylabel('bad facts')
plt.show()
print(bad_facts[len(bad_facts) - 1])
