import argparse
import codecs
import os, sys
import numpy as np
import random
import difflib
from difflib import SequenceMatcher
import re
import numpy as np
from collections import Counter
import pickle
import pandas as pd

def read_data(filename):
	with codecs.open(filename, 'r', 'utf-8') as inp:
		lines = inp.readlines()
	inputs = []
	outputs = []
	tags = []
	triplets = []
	for l in lines:
		l = l.strip().split('\t')
		# if len(l)<6:
		# 	print(len(l), l)
		if len(l) > 1:
			inputs.append(l[0])
			outputs.append(l[1])
			# tags.append(re.split('\W+', l[2].strip()))
			tags.append(l[2])
			triplets.append([l[3], l[4], l[5]])
	return inputs, outputs, tags, triplets

def count_sum(input, output, pos, triplets):
	np_triplets = np.array(triplets)
	prefs, sc, suffs = np_triplets[:, 0], np_triplets[:, 1], np_triplets[:, 2]
	char3, char2, char1 = [],[],[]
	for i,x in enumerate(input):
		char3.append(x[-3:] if len(x) >= 3 else (3 - len(x)) * "<s>" + x[-3:])
		char2.append(x[-2:] if len(x) >= 2 else (2 - len(x)) * "<s>" + x[-2:])
		char1.append(x[-1:] if len(x) >= 1 else (1 - len(x)) * "<s>" + x[-1:])

	char3_p_s_t = Counter(zip(char3, prefs, suffs, pos))
	sc_char3_p_s_t = Counter(zip(sc, char3, prefs, suffs, pos))
	char2_p_s_t = Counter(zip(char2, prefs, suffs, pos))
	sc_char2_p_s_t = Counter(zip(sc, char2, prefs, suffs, pos))
	char1_p_s_t = Counter(zip(char1, prefs, suffs, pos))
	sc_char1_p_s_t = Counter(zip(sc, char1, prefs, suffs, pos))
	p_s_t = Counter(zip(prefs, suffs, pos))
	sc_p_s_t = Counter(zip(sc, prefs, suffs, pos))
	sc_all = Counter(sc)
	counting_dict = {
		'char3_p_s_t': char3_p_s_t,
		'sc_char3_p_s_t': sc_char3_p_s_t,
		'char2_p_s_t': char2_p_s_t,
		'sc_char2_p_s_t': sc_char2_p_s_t,
		'char1_p_s_t': char1_p_s_t,
		'sc_char1_p_s_t': sc_char1_p_s_t,
		'p_s_t': p_s_t,
		'sc_p_s_t': sc_p_s_t,
		'sc_all': sc_all,
		'all': len(input)
	}

	return counting_dict, list(zip(input, char3, char2, char1))


def calc_lambdas(tri_input, output, pos, triplets, sums):
	termination_condition = 0.0001
	lambdas = []
	for i in range(5):
		lambdas.append(random.randint(1,10))
	norm = sum(lambdas)
	for i in range(5):
		lambdas[i] /= norm
	next_lambdas = [0.0] * 5
	expected_counts = [0.0] * 5
	while True:
		for i, trip in enumerate(triplets):
			inp, char3, char2, char1 = tri_input[i]
			t = pos[i]
			p,sc,s = trip[0],trip[1],trip[2]
			p0 = sums['sc_char3_p_s_t'][(sc,char3, p, s, t)]/sums['char3_p_s_t'][(char3, p, s, t)]
			p1 = sums['sc_char2_p_s_t'][(sc, char2, p, s, t)] / sums['char2_p_s_t'][(char2, p, s, t)]
			p2 = sums['sc_char1_p_s_t'][(sc, char1, p, s, t)] / sums['char1_p_s_t'][(char1, p, s, t)]
			p3 = sums['sc_p_s_t'][(sc, p, s, t)] / sums['p_s_t'][(p, s, t)]
			p4 = sums['sc_all'][sc] / sums['all']
			prob = lambdas[0]*p0 + lambdas[1]*p1 + lambdas[2]*p2 + lambdas[3]*p3 + lambdas[4]*p4

			expected_counts[0] += lambdas[0] * p0 / prob
			expected_counts[1] += lambdas[1] * p1 / prob
			expected_counts[2] += lambdas[2] * p2 / prob
			expected_counts[3] += lambdas[3] * p3 / prob
			expected_counts[4] += lambdas[4] * p4 / prob
		arr = []
		for i in range(5):
			next_lambdas[i] = expected_counts[i] / sum(expected_counts)
			arr.append(abs((lambdas[i] - next_lambdas[i])) < termination_condition)

		lambdas = next_lambdas.copy()
		print(lambdas)
		expected_counts = [0.0] * 5
		if all(arr):
			break
	print("Smoothed lambdas:")
	print(lambdas)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("datapath", help="path to data", type=str)
	parser.add_argument("language", help="language", type=str)
	# parser.add_argument("pair", help="lemma:inflection", type=str)
	# parser.add_argument("--all", help="all pairs? (def: false)", default=False, action="store_true")
	args = parser.parse_args()

	DATA_PATH = args.datapath
	L2 = args.language
	LOW_PATH = os.path.join(DATA_PATH, L2 + "-hall-o")
	# lemma, inf =  args.pair.split(":")
	input, output, pos, triplets = read_data(LOW_PATH)
	# lowi, lowo, lowt = lowi[:100], lowo[:100], lowt[:100]

	# counting_dict, tri_input =  count_sum(input, output, pos, triplets)
	# with open(os.path.join(DATA_PATH, L2 + "count_sum.pickle"), 'wb') as outp:
	# 	pickle.dump(counting_dict, outp)
	# with open(os.path.join(DATA_PATH, L2 + "tri_input.pickle"), 'wb') as outp:
	# 	pickle.dump(tri_input, outp)

	with open(os.path.join(DATA_PATH, L2 + "count_sum.pickle"), 'rb') as outp:
		counting_dict = pickle.load(outp)
	with open(os.path.join(DATA_PATH, L2 + "tri_input.pickle"), 'rb') as outp:
		tri_input = pickle.load(outp)

	calc_lambdas(tri_input, output, pos, triplets, counting_dict)
