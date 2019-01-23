import numpy as np
import pandas as pd
import csv
import random
import os

resource_loc = 'resources'


def to_matrix(n: int):
    """
    A helper function that generates a matrix of bit combinations
    :param n: size of the matrix
    :return: A matrix with the input combinations
    """
    def gen(n: int):
        for i in range(1, 2 ** n - 1):
            yield '{:0{n}b}'.format(i, n=n)

    matrix = [[0 for i in range(n)]]
    for perm in list(gen(n)):
        matrix.append([int(s) for s in perm])
    matrix.append([1 for i in range(n)])
    return matrix


def transformation(x):
    """
    A function that transforms 5 inputs into 3 outputs. (generates hard data set)
    :param x: A list to transform
    :return: The transformed list of size 3
    """
    return [x[0] | x[4], x[1] ^ x[2], x[3] & x[4]]


def generate_data_to_csv(matrix_size: int, file_name: str = 'hard_problem', transformation_function=transformation):
    """
    A helper function to aid in the generation of csv data
    :param transformation_function: The transformation function for generating the output bits
    :param matrix_size: The size of the input matrix
    :param file_name: The file name to produce
    :return: Input matrix and output matrix
    """
    input_array = np.asarray(to_matrix(matrix_size))
    output = np.apply_along_axis(transformation_function, 1, input_array)
    data_frame = pd.DataFrame(np.concatenate((input_array, output), axis=1))
    data_frame.to_csv(os.path.join('resources', f'{file_name}.csv'), header=None, index=None)
    return input_array, output


def import_data_from_csv_and_split(filename: str = 'hard_problem', input_range=5):
    """
    A function that parses a csv file with training data and converts it into a matrix
    :param filename: a string representation of the file name
    :param input_range: the size of input data
    :return: a tuple (input matrix, output matrix)
    """
    with open(os.path.join('resources', f'{filename}.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        input_matrix = []
        output_matrix = []
        for row in csv_reader:
            input_matrix.append(row[:input_range])
            output_matrix.append(row[input_range:])
        return input_matrix, output_matrix


def import_plain_data_from_csv(filename: str = 'hard_problem'):
    """
    A function that is responsible for importing data from csv
    :param filename: a string representing the file name
    :return: a list representing the csv
    """
    with open(os.path.join('resources', f'{filename}.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        return [row for row in csv_reader]


def split_matrix_into_training_set(data: list, train_size=26) -> (list, list):
    """
    A method that splits a matrix into a training matrix a verification matrix
    :param data: the matrix
    :param train_size: the size of the training data
    :return: a tuple (training data, verification data)
    """
    data_copy = data
    random.shuffle(data_copy)
    return data_copy[:train_size], data_copy[train_size:]
