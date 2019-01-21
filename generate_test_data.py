#!/usr/bin/env python3
from utils.csv_utils import generate_data_to_csv, transformation

generate_data_to_csv(5, "simple_problem", lambda x: [x[0], x[1], x[4]])
generate_data_to_csv(5, 'normal_problem', lambda x: [x[0] | x[4], x[1] & x[2], x[4]])
generate_data_to_csv(5, 'hard_problem', transformation)
