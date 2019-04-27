import csv

import helper


def to_output(self, cluster):
	out = [0] * 10
	out[cluster - 1] = 1
	return out


with open('data/brain/brain.csv', 'rb') as fp:
	reader = csv.reader(fp)
	if True:
		next(reader, None)
	for row in reader:
		row = helper.to_float(row)
		out = to_output(int(row.pop(15)))
		data.append((row, out))

return data
