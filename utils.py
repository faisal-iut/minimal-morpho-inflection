import argparse
import codecs
import os, sys
from random import random, choice
import difflib
from difflib import SequenceMatcher
import re

def read_data(filename):
	with codecs.open(filename, 'r', 'utf-8') as inp:
		lines = inp.readlines()
	inputs = []
	outputs = []
	tags = []
	triplets = []
	for l in lines:
		l = l.strip().split('\t')
		if len(l) > 1:
			inputs.append(list(l[0].strip()))
			outputs.append(list(l[1].strip()))
			tags.append(re.split('\W+', l[2].strip()))
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


def get_triplet(lemma, inf, triplets):
	# t = ''.join(zip(triplets))
	choosen_triplets = [(x[0],x[1],x[2]) for x in triplets if reconstruct_inflection(lemma,inf,x)]
	print(set(choosen_triplets))
	return 0

def get_chars(l):
	flat_list = [char for word in l for char in word]
	return list(set(flat_list))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("datapath", help="path to data", type=str)
	parser.add_argument("language", help="language", type=str)
	parser.add_argument("pair", help="lemma:inflection", type=str)
	parser.add_argument("--all", help="all pairs? (def: false)", default=False, action="store_true")
	args = parser.parse_args()

	DATA_PATH = args.datapath
	L2 = args.language
	LOW_PATH = os.path.join(DATA_PATH, L2 + "-hall")
	lemma, inf =  args.pair.split(":")
	input, output, pos, triplets = read_data(LOW_PATH)
	# lowi, lowo, lowt = lowi[:100], lowo[:100], lowt[:100]

	vocab = get_chars(input+output)
	get_triplet(lemma, inf, triplets)