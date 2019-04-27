# coding=utf-8
import csv
import json
import os
import random

import configparser

import helper


class Network:
	weights = []
	training_set = []
	test_set = []

	def __init__(self, config_path):
		self.config = configparser.ConfigParser()
		self.config.read(config_path)

		self.input_node = self.config.getint('nodes', 'input')
		self.output_node = self.config.getint('nodes', 'output')
		self.layers = [(0, self.config.getint('nodes', 'hidden')), (1, self.output_node)]
		self.layer_count = len(self.layers)
		self.learning_rate = self.config.getfloat('learning', 'rate')
		self.error_muls = [0] * self.layer_count
		self.iteration = 0
		self.result_writer = helper.result_writer()

		self.reset()

	def reset(self):
		self.iteration = 0
		self.weights = self.init_weights()
		self.training_set = self.get_training_set()

		self.training_set = self.training_set.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

		self.test_set = self.get_test_set()
		if self.config.getboolean('learning', 'shuffle'):
			random.shuffle(self.training_set)
			random.shuffle(self.test_set)

	def prediction(self, input_vector):
		inputs = [input_vector]
		for layer, node_count in self.layers:
			outputs = []
			for node in range(0, node_count):
				out = helper.weighted_sum(inputs[layer], self.weights[layer][node])
				# if node != node_count - 1:
				out = helper.activation(out)
				outputs.append(out)
			inputs.append(outputs)

		return inputs.pop(), inputs

	def init_weights(self):
		if os.path.exists('weight.json'):
			print 'weight file exists'
			with open('weight.json') as data_file:
				return json.load(data_file)

		return [
			helper.random_weight(self.input_node, self.layers[0][1]),
			helper.random_weight(self.layers[0][1], self.layers[1][1]),
		]

	def get_training_set(self):
		return self.get_data_set(self.config.get('data', 'trainingSet'), self.config.getboolean('data', 'header'))

	def get_test_set(self):
		# limit = int(len(training_set) * 0.2)
		#
		# random.shuffle(training_set)
		# test_set = training_set[:limit]
		# training_set = training_set[:-limit]
		return self.get_data_set(self.config.get('data', 'testSet'), self.config.getboolean('data', 'header'))

	def to_output(self, cluster):
		out = [0] * self.output_node
		out[cluster - 1] = 1
		return out

	def get_data_set(self, data_file, header=False):
		class_column = self.config.getint('data', 'classColumn')

		data = []
		with open(data_file, 'rb') as fp:
			reader = csv.reader(fp)
			if header:
				next(reader, None)
			for row in reader:
				row = helper.to_float(row)
				out = self.to_output(int(row.pop(class_column)))
				data.append((row, out))

		return data

	def calc_error_muls(self, inputs, result, errors):
		reversed_layer = self.layers[:]
		reversed_layer.reverse()

		for layer, node_count in reversed_layer:
			if layer == self.layer_count - 1:
				self.error_muls[layer] = helper.error_mul(errors, result)
				continue
			self.error_muls[layer] = [
				helper.weighted_sum(t, self.error_muls[layer + 1]) * helper.to_der(inputs[layer + 1][key])
				for key, t in enumerate(zip(*self.weights[self.layer_count - 1]))]

	def weights_update(self, inputs):
		reversed_layer = self.layers[:]
		reversed_layer.reverse()

		for layer, node_count in reversed_layer:
			for node in range(0, node_count):
				ws = self.weights[layer][node]
				for key, w in enumerate(ws):
					ws[key] += self.learning_rate * self.error_muls[layer][node] * inputs[layer][key]

	def state_log(self, extra):
		self.result_writer.writerow([self.iteration, self.learning_rate] + extra)

	def export_weights(self):
		print 'exporting...'
		with open('weight.json', 'w') as outfile:
			json.dump(self.weights, outfile)
