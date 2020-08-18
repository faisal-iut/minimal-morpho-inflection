import argparse
import codecs
import os, sys
from random import random, choice
import difflib
from difflib import SequenceMatcher
import re
import numpy as np
from collections import Counter

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("datapath", help="path to data", type=str)
	parser.add_argument("language", help="language", type=str)
	# parser.add_argument("pair", help="lemma:inflection", type=str)
	# parser.add_argument("--all", help="all pairs? (def: false)", default=False, action="store_true")
	args = parser.parse_args()

	DATA_PATH = args.datapath
	L2 = args.language
	LOW_PATH = os.path.join(DATA_PATH, L2 + "-hall")
	# lemma, inf =  args.pair.split(":")
	input, output, pos, triplets = read_data(LOW_PATH)
	# lowi, lowo, lowt = lowi[:100], lowo[:100], lowt[:100]
	counting_dict =  count_all(triplets, pos)

	data_dict = {}
	for k,trip in enumerate(triplets):
		if (input[k], output[k], pos[k]) not in data_dict.keys():
			data_dict[(input[k], output[k], pos[k])] = []
		data_dict[(input[k], output[k], pos[k])].append(trip)

	with codecs.open(os.path.join(DATA_PATH, L2 + "-hall-o"), 'w', 'utf-8') as outp:
		for key, val in data_dict.items():
			lemma, inf, pos_v = key[0], key[1], key[2]
			best_triplet = argmax_triplet(val, pos_v, counting_dict)
			outp.write(lemma + '\t' + inf + '\t' + pos_v + '\t' +
					   best_triplet[0] + '\t' + best_triplet[1] + '\t' + best_triplet[2] + '\n')