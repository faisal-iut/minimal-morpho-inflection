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
from utils import df_to_dict, calculate_avg_ln

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


def calc_lambdas(proba, tot_it):
	termination_condition = 0.0001
	lambdas = []
	for i in range(5):
		lambdas.append(random.randint(1,10))
	norm = sum(lambdas)
	for i in range(5):
		lambdas[i] /= norm
	next_lambdas = [0.0] * 5
	expected_counts = [0.0] * 5
	triplet =  list(zip(proba['prefix'], proba['stem_change'], proba['suffix'], proba['n3'], proba['n2'], proba['n1']))
	n1_pscs = df_to_dict(proba, ['n1', 'prefix', 'stem_change', 'suffix'], 'n1_pscs')
	n2_pscs = df_to_dict(proba, ['n2', 'prefix', 'stem_change', 'suffix'], 'n2_pscs')
	n3_pscs = df_to_dict(proba, ['n3', 'prefix', 'stem_change', 'suffix'], 'n3_pscs')
	sc_ps = df_to_dict(proba, ['prefix', 'stem_change', 'suffix'], 'sc_ps')
	p_sc = df_to_dict(proba, ['stem_change'], 'p_sc')
	iteration =0
	prob_matrix = np.zeros((len(triplet), len(lambdas)))
	tracked_info = {}
	while True:
		print("iteration no {}".format(iteration))
		for i, trip in enumerate(triplet):
			p,sc,s = trip[0],trip[1],trip[2]
			n3, n2, n1 = trip[3],trip[4],trip[5]
			p0 = n3_pscs[(n3, p, sc, s)]
			p1 = n2_pscs[(n2, p, sc, s)]
			p2 = n1_pscs[(n1, p, sc, s)]
			p3 = sc_ps[(p, sc, s)]
			p4 = p_sc[sc]
			prob = lambdas[0]*p0 + lambdas[1]*p1 + lambdas[2]*p2 + lambdas[3]*p3 + lambdas[4]*p4
			expected_counts[0] += lambdas[0] * p0 / prob
			expected_counts[1] += lambdas[1] * p1 / prob
			expected_counts[2] += lambdas[2] * p2 / prob
			expected_counts[3] += lambdas[3] * p3 / prob
			expected_counts[4] += lambdas[4] * p4 / prob
			if iteration==0:
				prob_matrix[i][0] = p0
				prob_matrix[i][1] = p1
				prob_matrix[i][2] = p2
				prob_matrix[i][3] = p3
				prob_matrix[i][4] = p4
		tracked_info[iteration] = {'lambdas': lambdas,
						'avg_ll': calculate_avg_ln(prob_matrix, lambdas)}
		arr = []
		for i in range(5):
			next_lambdas[i] = expected_counts[i] / sum(expected_counts)
			arr.append(abs((lambdas[i] - next_lambdas[i])) < termination_condition)

		lambdas = next_lambdas.copy()
		expected_counts = [0.0] * 5
		iteration = iteration+1
		# if all(arr):
		if iteration>tot_it:
			tracked_info[iteration] = {'lambdas': lambdas,
									   'avg_ll': calculate_avg_ln(prob_matrix, lambdas)}
			break

	return tracked_info

def em_iteration(df, iterations):
	print(iterations)
	it = 0
	df['p(l|w)']=0
	df['ec(l)']=0
	df['ec(l,w)']=0

	while True:
		df['p(l,w)'] = df['p(l)'] * df['p(w|l)']
		for i, row in df.iterrows():
			l,w, p_lw, ec_l, ec_lw = row['l'],row['w'], row['p(l,w)'], row['ec(l)'], row['ec(l,w)']
			p_w =  sum(df.loc[df['w'] == w, 'p(l,w)'].values)
			df.loc[(df['l']==l) & (df['w']==w),'p(l|w)'] = p_lw/p_w
			df.loc[(df['l'] == l) , 'ec(l)'] = df.loc[(df['l'] == l) , 'ec(l)'] + p_lw/p_w
			df.loc[(df['l'] == l) & (df['w'] == w), 'ec(l,w)'] = df.loc[(df['l'] == l) & (df['w'] == w), 'ec(l,w)']\
																 + p_lw/p_w
		df['p(l)'] = df['ec(l)']/len(set(df['w']))
		df['p(w|l)'] = df['ec(l,w)'] / df['ec(l)']
		if it==iterations:
			break
		it=it+1
		print('iteration', it)
		print(df.iloc[:, np.r_[0:6,7]])
		df['ec(l)'] = 0
		df['ec(l,w)'] = 0


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("datapath", help="path to data", type=str)
	parser.add_argument("language", help="language", type=str)
	parser.add_argument("-em", "--em", type=int,
						help="calculate lambdas [train/dev]")
	parser.add_argument("-t", "--toy", type=int,
						help="calculate lambdas [train/dev]")
	args = parser.parse_args()

	DATA_PATH = args.datapath
	L2 = args.language
	LOW_PATH = os.path.join(DATA_PATH, L2 + "-hall-o")
	# lemma, inf =  args.pair.split(":")
	input, output, pos, triplets = read_data(LOW_PATH)

	if args.em:
		tot_it = args.em
		with open(os.path.join(DATA_PATH, L2+ "-dev-proba.pickle"), 'rb') as outp:
			proba = pickle.load(outp)
		print("EM ITERATION FOR LANGUAGE: {}".format(L2))
		tracked_info = calc_lambdas(proba, tot_it)
		with open(os.path.join(DATA_PATH, L2 + "-dev_avg_ll.pickle"), 'wb') as outp:
			pickle.dump(tracked_info, outp)
	if args.toy>=0:
		filename = os.path.join(DATA_PATH, L2 + "-toy")
		with codecs.open(filename, 'r', 'utf-8') as inp:
			lines = inp.readlines()
		inputs = []
		outputs = []
		for l in lines:
			l = l.strip().split('\t')
			if len(l) > 1:
				inputs.append(l[0])
				outputs.append(l[1])
		ds = list(zip(inputs, outputs))

		lst = []
		cols = ["l","w","p(l)","p(w|l)"]
		for i,lw in enumerate(ds):
			l,w = lw[0],lw[1]
			p_l = 1/len(set(inputs))
			# p_l = inputs.count(l) / len(inputs)
			p_wbl =1/inputs.count(l)
			lst.append([l,w,p_l, p_wbl])
		df = pd.DataFrame(lst, columns=cols)
		print(df)
		em_iteration(df, args.toy)

	# lowi, lowo, lowt = lowi[:100], lowo[:100], lowt[:100]

	# counting_dict, tri_input =  count_sum(input, output, pos, triplets)
	# with open(os.path.join(DATA_PATH, L2 + "count_sum.pickle"), 'wb') as outp:
	# 	pickle.dump(counting_dict, outp)
	# with open(os.path.join(DATA_PATH, L2 + "tri_input.pickle"), 'wb') as outp:
	# 	pickle.dump(tri_input, outp)

	# with open(os.path.join(DATA_PATH, L2 + "count_sum.pickle"), 'rb') as outp:
	# 	counting_dict = pickle.load(outp)
	# with open(os.path.join(DATA_PATH, L2 + "tri_input.pickle"), 'rb') as outp:
	# 	tri_input = pickle.load(outp)
