# coding=utf-8
import csv
import os
import random

import math


def random_list(length):
	return [(random.randrange(1, 100, 1) - 50) / 100.0 for _ in range(length)]


def random_weight(input_count, node_count):
	return [random_list(input_count) for _ in range(node_count)]


def dot_product(x, y):
	return [v * w for v, w in zip(x, y)]


def weighted_sum(values, weights):
	return sum(v * w for v, w in zip(values, weights))


def activation(x):
	# print x
	# if x < -20:
	# 	return 0
	# if x > 20:
	# 	return 1
	# return (math.e ** x - math.e ** -x) / (math.e ** x + math.e ** -x)
	return 1 / (1 + math.e ** -x)


# return x / (1 + abs(x))

# if x == 0.5:
# 	return 0.5
# return (((x - 0.5) / abs(x - 0.5)) + 1) / 2.0


def to_der(input):
	return input * (1 - input)


# return (input + 0.00005) * (1.00005 - input)


def to_float(row):
	return [float(i) for i in row]


def calc_errors(predicted, desired):
	return [pre - des for pre, des in zip(predicted, desired)]


def sum_of_error(errors):
	return sum(e ** 2 for e in errors) / 2.0


def error_mul(errors, results):
	return [error * to_der(result) for error, result in zip(errors, results)]


def result_writer():
	file_name = 'result/learn.csv'
	if os.path.exists(file_name):
		return get_writer(file_name)

	writer = get_writer(file_name)
	writer.writerow(['iteration', 'learning_rate', 'success', 'test', 'db', 'tag'])
	return writer


def get_writer(file_name):
	return csv.writer(open(file_name, 'a'))
