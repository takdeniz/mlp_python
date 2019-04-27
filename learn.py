# coding=utf-8
import csv
import configparser
import helper
from network import Network

tag = 'shuffle'

config = configparser.ConfigParser()
config.read('app.cfg')

db = config.get('data', 'config')
network = Network('data/' + db + '/' + db + '.cfg')

while True:
	network.iteration += 1
	# if network.iteration % 500 == 0 and network.iteration < 4000:
	# 	network.learning_rate *= 0.67

	# print '-' * 60, learning_rate
	error_sum = 0

	for input_vector, desired_output in network.training_set:
		result, inputs = network.prediction(input_vector)
		errors = helper.calc_errors(desired_output, result)
		error = helper.sum_of_error(errors)

		if error != 0:
			error_sum += error
			network.calc_error_muls(inputs, result, errors)
			network.weights_update(inputs)

	if network.iteration % 1 == 0:
		learning = (1 - (error_sum / len(network.training_set))) * 100
		print network.iteration, 'train: ', learning
		network.state_log([learning, 'train', db, tag])

		error_sum = 0
		for inp, des in network.test_set:
			output, t = network.prediction(inp)
			errors = helper.calc_errors(des, output)
			error_sum += helper.sum_of_error(errors)

		test_success = (1 - (error_sum / len(network.test_set))) * 100
		network.state_log([test_success, 'test', db, tag])

		print network.iteration, 'test: ', test_success

		if network.iteration > 10000:
			break
		#
		# if abs(error_sum) < 0.04:
		# 	break
