import numpy as np
import pandas as pd
import csv
import random
import os

resource_loc = 'resources'


def to_matrix(n):
    def gen(n):
        for i in range(1, 2 ** n - 1):
            yield '{:0{n}b}'.format(i, n=n)

    matrix = [[0 for i in range(n)]]
    for perm in list(gen(n)):
        matrix.append([int(s) for s in perm])
    matrix.append([1 for i in range(n)])
    return matrix


def transformation(x):
    return [x[0] | x[4], x[1] ^ x[2], x[3] & x[4]]


def generate_data_to_csv(matrix_size: int, file_name: str = 'foo', transformation_function=transformation):
    """
    A helper function to aid in generating csv data
    :param transformation_function: The transformation function for generating the output bits
    :param matrix_size: The size of the matrix
    :param file_name: The file name to produce
    :return: Input matrix and output matrix
    """
    input_array = np.asarray(to_matrix(matrix_size))
    output = np.apply_along_axis(transformation_function, 1, input_array)
    data_frame = pd.DataFrame(np.concatenate((input_array, output), axis=1))
    data_frame.to_csv(f'{file_name}.csv', header=None, index=None)
    return input_array, output


def import_data_from_csv_and_split(filename: str = 'foo', input_range=5):
    with open(f'{filename}.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        input_matrix = []
        output_matrix = []
        for row in csv_reader:
            input_matrix.append(row[:input_range])
            output_matrix.append(row[input_range:])
        return input_matrix, output_matrix


def import_plain_data_from_csv(filename: str = 'foo'):
    with open(f'{filename}.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        return [row for row in csv_reader]


def split_matrix_into_training_set(data: list, train_size=26):
    data_copy = data
    random.shuffle(data_copy)
    return data_copy[:train_size], data_copy[train_size:]
