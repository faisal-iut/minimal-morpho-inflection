import argparse
import codecs
import os, sys
import random
import difflib
from difflib import SequenceMatcher
import re
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from typing import List, Tuple, Callable, Union

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


def reconstruct_inflection(lemma, inf, trip):
    eps = '\u0395'
    sc = trip[1].split("->")
    if sc[0]!= eps and (sc[0] != lemma[-len(sc[0]):]):
        return 0
    add_p = trip[0] + lemma if trip[0] != eps else lemma
    add_sc1 = add_p[:-len(sc[0])] if sc[0] == lemma[-len(sc[0]):] else add_p
    add_sc2 = add_sc1 + sc[1] if sc[1] != eps else add_sc1
    add_s = add_sc2 + trip[2] if trip[2] != eps else add_sc2
    if add_s == inf:
        return 1
    else:
        return 0

def argmax_triplet(choosen_triplets, pos_v , counting_dict):
	max_prob = -10000
	max_i = 0
	for i, tr in enumerate(choosen_triplets):
		p, sc, s= tr[0],tr[1],tr[2]
		prob_n =  counting_dict['p_sc'][(p,sc)]* counting_dict['s_sc'][(s,sc)]*counting_dict['pos_sc'][(pos_v,sc)]
		prob_d = counting_dict['all_sc'][sc]*counting_dict['all_sc'][sc]*counting_dict['all']
		prob  =prob_n/prob_d
		if prob>max_prob:
			max_i = i
			max_prob = prob
	return choosen_triplets[max_i]

def get_triplet(lemma, inf, triplets):
	# t = ''.join(zip(triplets))
	choosen_triplets = [(x[0],x[1],x[2]) for i,x in enumerate(triplets) if reconstruct_inflection(lemma,inf,x)]

	return list(set(choosen_triplets))


def count_all(triplets, pos):
	np_triplets =  np.array(triplets)
	prefs, sc, suffs = np_triplets[:,0], np_triplets[:,1], np_triplets[:,2]
	pref_sc = Counter(zip(prefs, sc))
	s_sc = Counter(zip(suffs, sc))
	pos_sc =  Counter(zip(pos, sc))
	all_sc =  Counter(sc)
	all_eg = len(pos)
	counting_dict =  {
		'p_sc': pref_sc,
		's_sc': s_sc,
		'pos_sc': pos_sc,
		'all_sc': all_sc,
		'all': all_eg
	}
	return counting_dict

def split_dataset(filepath, L, r1, r2):
	with codecs.open(os.path.join(filepath, L), 'r', 'utf-8') as inp:
		lines = inp.readlines()

	data = {}
	for l in lines:
		l = l.strip().split('\t')
		if l[0] not in data.keys() and l[0] != '':
			data[l[0]] = []
		if len(l) > 1:
			data[l[0]].append(l[1:])
	l = list(data.items())
	random.shuffle(l)
	data = dict(l)
	train = dict(list(data.items())[:r1*len(data) // (r1+r2)])
	dev = dict(list(data.items())[r1*len(data) // (r1+r2):])

	with codecs.open(os.path.join(filepath, L + "-train"), 'w', 'utf-8') as outp:
		for key, val in train.items():
			lemma = key
			for form in val:
				outp.write(lemma + '\t' + form[0] + '\t' + form[1] + '\n')
			outp.write('\n')

	with codecs.open(os.path.join(filepath, L + "-dev"), 'w', 'utf-8') as outp:
		for key, val in dev.items():
			lemma = key
			for form in val:
				outp.write(lemma + '\t' + form[0] + '\t' + form[1] + '\n')
			outp.write('\n')
	print("{} language data splited into train:dev = {}:{} ratio".format(L, len(train), len(dev)))

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


def calc_train_probalility(L, input, output, pos, triplets):
	np_triplets = np.array(triplets)
	prefs, sc, suffs = np_triplets[:, 0], np_triplets[:, 1], np_triplets[:, 2]
	n3, n2, n1 = [],[],[]
	for i,x in enumerate(input):
		n3.append(x[-3:] if len(x) >= 3 else "<UNK>")
		n2.append(x[-2:] if len(x) >= 2 else "<UNK>")
		n1.append(x[-1:] if len(x) >= 1 else "<UNK>")

	p_s_t = Counter(zip(prefs, suffs, pos))
	s_sc = Counter(zip(suffs, sc))
	t_sc = Counter(zip(pos, sc))
	count_sc = Counter(sc)
	p_sc_s =  Counter(zip(prefs, sc, suffs))
	p_s = Counter(zip(prefs, suffs))
	n3_p_sc_s = Counter(zip(n3, prefs, sc, suffs))
	n2_p_sc_s = Counter(zip(n2, prefs, sc, suffs))
	n1_p_sc_s = Counter(zip(n1, prefs, sc, suffs))
	lst = []
	cols = ['input', 'output', 'n3', 'n2', 'n1', 'prefix',
			'stem_change', 'suffix', 'n3_pscs', 'n2_pscs', 'n1_pscs', 'sc_ps', 'p_sc']
	for i,trip in enumerate(triplets):
		cond_n3pscs = n3_p_sc_s[(n3[i], trip[0],trip[1],trip[2])]/p_sc_s[(trip[0],trip[1],trip[2])]
		cond_n2pscs = n2_p_sc_s[(n2[i], trip[0], trip[1], trip[2])] / p_sc_s[(trip[0], trip[1], trip[2])]
		cond_n1pscs = n1_p_sc_s[(n1[i], trip[0], trip[1], trip[2])] / p_sc_s[(trip[0], trip[1], trip[2])]
		cond_scps = p_sc_s[(trip[0], trip[1], trip[2])] / p_s[(trip[0], trip[2])]
		cond_sc = count_sc[trip[1]]/len(sc)
		lst.append([input[i], output[i], n3[i], n2[i], n1[i], trip[0], trip[1], trip[2],
					cond_n3pscs, cond_n2pscs, cond_n1pscs, cond_scps, cond_sc])
	df = pd.DataFrame(lst, columns=cols)
	print("{} Train probability dataframe shape {}".format(L, df.shape))
	return df

def reconstruct_lemma(output, triplet):
	eps = '\u0395'
	new_trip = []
	for trip in triplet:
		p, sc, s = trip[0], trip[1], trip[2]
		scb, sca = trip[1].strip().split('->')
		p = "" if p == eps or p == "*" else p
		s = "" if s == eps else s
		scb = "" if scb == eps else scb
		sca = "" if sca == eps else sca
		stm = p + '.*' + sca + s
		stm = stm.replace(")", "")
		stm = stm.replace("(", "")
		pat = re.compile(stm)
		m = pat.match(output)
		if m is not None:
			trail = sca + s
			if trail == output[-len(trail):]:
				f1 = output[:-len(trail)]
				f2 = f1 + scb
				f3 = p + f2
				new_trip.append((trip[0], trip[1], trip[2], f3))
	new_trip = list(set(new_trip))
	if len(new_trip) == 0:
		new_trip.append((eps, "{}->{}".format(eps, eps), eps, output))

	return new_trip

def df_to_dict(df, column_l, val_col):
    temp_dict = {}
    temp = df.set_index(column_l)
    temp = dict(zip(temp.index,temp[val_col]))
    for k, v in temp.items():
        if k not in temp_dict.keys():
            temp_dict[k] = v
    return temp_dict

def proba_from_dict(key, d):
    if key in d.keys():
        proba = d[key]
    else:
        proba = 1/len(d)
    return proba

def calc_dev_probalility(DATA_PATH,L):
	eps = '\u0395'
	train_proba_file = os.path.join(DATA_PATH, L + "-train-proba.pickle")
	with open(train_proba_file, 'rb') as outp:
		tr_prob = pickle.load(outp)
		proba = tr_prob

	devfile = os.path.join(DATA_PATH, L + "-dev")
	with codecs.open(devfile, 'r', 'utf-8') as inp:
		lines = inp.readlines()

	train_triplet = list(set(zip(tr_prob['prefix'],
							 tr_prob['stem_change'], tr_prob['suffix'])))

	cols = ['input', 'output', 'n3', 'n2', 'n1', 'prefix',
			'stem_change', 'suffix', 'n3_pscs', 'n2_pscs', 'n1_pscs', 'sc_ps','p_sc']
	lst = []

	n1_pscs = df_to_dict(proba, ['n1', 'prefix', 'stem_change', 'suffix'], 'n1_pscs')
	n2_pscs = df_to_dict(proba, ['n2', 'prefix', 'stem_change', 'suffix'], 'n2_pscs')
	n3_pscs = df_to_dict(proba, ['n3', 'prefix', 'stem_change', 'suffix'], 'n3_pscs')
	sc_ps = df_to_dict(proba, ['prefix', 'stem_change', 'suffix'], 'sc_ps')
	p_sc = df_to_dict(proba, ['stem_change'], 'p_sc')

	for i,l in enumerate(lines):
		# if i>1000:
		# 	break
		l = l.strip().split('\t')
		if len(l)>1:
			input, output =  l[0], l[1]
			new_trip = reconstruct_lemma(output, train_triplet)
			for new_t in new_trip:
				x = new_t[3]
				p, sc, s = new_t[0], new_t[1], new_t[2]

				n1 = x[-1:] if len(x) >= 1 else "<UNK>"
				n2 = x[-2:] if len(x) >= 2 else "<UNK>"
				n3 = x[-3:] if len(x) >= 3 else "<UNK>"

				un1_pscs = proba_from_dict((n1, p, sc, s), n1_pscs)
				un2_pscs = proba_from_dict((n2, p, sc, s), n2_pscs)
				un3_pscs = proba_from_dict((n3, p, sc, s), n3_pscs)
				usc_ps = proba_from_dict((p, sc, s), sc_ps)
				up_sc = proba_from_dict(sc, p_sc)

				lst.append(
					[new_t[3], output, n3, n2, n1, new_t[0], new_t[1],
					 new_t[2], un3_pscs, un2_pscs, un1_pscs, usc_ps,
					 up_sc])
		# print('{} out of {} done {}'.format(i,len(lines), len(train_triplet)))

	df = pd.DataFrame(lst, columns=cols)

	print("{} Dev probability dataframe shape {}".format(L, df.shape))
	return df

def calculate_avg_ll(prob_matrix: np.ndarray, weights: List[float] = None, log_function: Callable = np.log2) -> float:
    """
    Calculate average log likelihood from weighted combination of columns in probability matrix of evaluation text
    :param prob_matrix: probability matrix of evaluation text
    :param weights: corresponding weight of each column
    :param log_function: log function to use (often np.log2 for base 2 log, or np.log for natural log)
    :return: average log likelihood from weighted combination of columns
    """
    n_models = prob_matrix.shape[1]
    if weights is None:
        weights = np.ones(n_models) / n_models
    interpolated_probs = np.sum(prob_matrix * weights, axis=1)
    average_log_likelihood = log_function(interpolated_probs).mean()
    return average_log_likelihood


def calculate_avg_ln(prob_matrix: np.array, weights: Union[List[float], np.array] = None) -> float:
    """
    Calculate average natural log likelihood of evaluation text with given interpolation weights
    :param prob_matrix: probability matrix of n_words x n_models
    :param weights: given weights for each model
    :return: average natural log of evaluation text with given weights
    """
    return calculate_avg_ll(prob_matrix, weights, log_function=np.log)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("datapath", help="path to data", type=str)
	parser.add_argument("language", help="language", type=str)
	parser.add_argument("-s", "--split_ds", nargs='+', type = int,
						help="split dataset to train:dev by given ratio[eg. 1 1, 2 3]")
	parser.add_argument("-p", "--calc_prob", type=str,
						help="calculate probabilities [train/dev]")
	# parser.add_argument("pair", help="lemma:inflection", type=str)
	# parser.add_argument("--all", help="all pairs? (def: false)", default=False, action="store_true")
	args = parser.parse_args()

	DATA_PATH = args.datapath
	L2 = args.language
	# lowi, lowo, lowt = lowi[:100], lowo[:100], lowt[:100]

	if args.split_ds:
		r1, r2 =  args.split_ds
		split_dataset(DATA_PATH, L2, r1, r2)

	if args.calc_prob=="train":
		LOW_PATH = os.path.join(DATA_PATH, L2 + "-hall")
		# lemma, inf =  args.pair.split(":")
		input, output, pos, triplets = read_data(LOW_PATH)
		proba = calc_train_probalility(L2, input, output, pos, triplets)
		with open(os.path.join(DATA_PATH, L2 + "-train-proba.pickle"), 'wb') as outp:
			pickle.dump(proba, outp)

	elif args.calc_prob == "dev":
		proba = calc_dev_probalility(DATA_PATH, L2)
		with open(os.path.join(DATA_PATH, L2 + "-dev-proba.pickle"), 'wb') as outp:
			pickle.dump(proba, outp)
	# counting_dict =  count_all(triplets, pos)
	# data_dict = {}
	# for k,trip in enumerate(triplets):
	# 	if (input[k], output[k], pos[k]) not in data_dict.keys():
	# 		data_dict[(input[k], output[k], pos[k])] = []
	# 	data_dict[(input[k], output[k], pos[k])].append(trip)
	#
	# with codecs.open(os.path.join(DATA_PATH, L2 + "-hall-o"), 'w', 'utf-8') as outp:
	# 	for key, val in data_dict.items():
	# 		lemma, inf, pos_v = key[0], key[1], key[2]
	# 		best_triplet = argmax_triplet(val, pos_v, counting_dict)
	# 		outp.write(lemma + '\t' + inf + '\t' + pos_v + '\t' +
	# 				   best_triplet[0] + '\t' + best_triplet[1] + '\t' + best_triplet[2] + '\n')