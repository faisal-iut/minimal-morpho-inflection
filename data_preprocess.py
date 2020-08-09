import align
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
	for l in lines:
		l = l.strip().split('\t')
		if len(l) > 1:
			inputs.append(list(l[0].strip()))
			outputs.append(list(l[1].strip()))
			tags.append(re.split('\W+', l[2].strip()))
	return inputs, outputs, tags


def seqMatch(str1, str2):
	is_junk = lambda x: x in " _"
	seqMatch = SequenceMatcher(is_junk, str1, str2)
	match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))
	# longest substring
	if (match.size != 0):
		lcs = (str1[match.a: match.a + match.size], (match.a, match.a + match.size))
	else:
		lcs = -1
	matches = seqMatch.get_matching_blocks()
	blocks = [(str1[match.a:match.a + match.size],
			   (match.a, match.a + match.size, match.b, match.b + match.size))
			  for match in matches if len(str1[match.a:match.a + match.size]) > 1]

	stc = []
	for tag, i1, i2, j1, j2 in seqMatch.get_opcodes():
		# print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(
		# 	tag, i1, i2, j1, j2, str1[i1:i2], str2[j1:j2]))
		stc.append((tag[0], i1, i2, j1, j2, str1[i1:i2], str2[j1:j2]))
	return lcs, blocks, stc


# e: equal, i: insert, r: replace, d: delete
rules = {'ei': '+s',
		 'er': '-sc+s',
		 'ed': '-sc-',
		 'ere': '-sc+s',
		 'ered': '-sc+s',
		 'e': 'eee',
		 'eiei': '-sc+s[1:]',
		 'rer': '~sc+s',
		 'reiei': '~sc+s',
		 'erer': '~sc+s-3',
		 'ereiei': '~sc+s-3',
		 'eied': 'spe',
		 'eier': 'spe',
		 'eieier': 'spe',
		 'eierei': 'spe',
		 'eieiei': 'spe',
		 'eder': '-sc+s-1',
		 'erei': '-sc+s-1',
		 'eded': '-sc+s-1',
		 'edeiei': '-sc+s-1',
		 'rerer': '-sc+s-2',
		 'ererei': 'sc->rer:s->i',
		 'rereie': '~sc+s:-2'}


def make_triplet(tx_set, p, sc, s):
	triplet = {}
	triplet = {'inp': tx_set[0],
			   'oup': tx_set[1],
			   'pos': tx_set[2],
			   'p': p,
			   'sc': sc,
			   's': s}
	return triplet


def augment(inputs, outputs, poss, characters):
	temp = [(''.join(inputs[i]), ''.join(outputs[i])) for i in range(len(outputs))]
	# aligned = align.Aligner(temp).alignedpairs
	vocab = list(characters)
	try:
		vocab.remove(u" ")
	except:
		pass

	triplets = []
	for k, item in enumerate(inputs):
		inp, oup, pos = ''.join(inputs[k]), ''.join(outputs[k]), '-'.join(poss[k])
		tx_set = [inp, oup, pos]
		# i, o = item[0], item[1]
		# inp_len, oup_len, al_len = len(inp), len(oup), len(i)
		lss, mb, stc = seqMatch(inp, oup)
		eps = '\u0395'
		prefix_flag = False
		tag = ''.join(list(zip(*stc))[0])
		# print(tag)
		if tag[0] == 'i':
			tag = tag[1:]
			i_stc = stc[0]
			stc = stc[1:]
			prefix_flag = True
		if tag in rules:
			if rules[tag] == '+s':
				triplet = make_triplet(tx_set, eps, "{}->{}".format(eps, eps), stc[1][6])
				triplets.append(triplet)
				# rule for dance:danced [p:E, e->E, s:eed]
				triplet = make_triplet(tx_set, eps, "{}->{}".format(stc[0][6][-1], eps), stc[0][6][-1] + stc[1][6])
				# rule for abet:abetted [p:E, sc:t->tt, s:ed]
				if (stc[0][5][-1] == stc[1][6][0]) and len(stc[1][6]) > 1:
					triplets.append(triplet)
					triplet = make_triplet(tx_set, eps, "{}->{}".
										   format(stc[0][5][-1],
												  stc[0][5][-1] + stc[1][6][0]), stc[1][6][1:])
			elif rules[tag] == '-sc+s':
				triplet = make_triplet(tx_set, eps, "{}->{}".format(inp[stc[1][1]:], eps), oup[stc[1][3]:])
			elif rules[tag] == 'eee':
				triplet = make_triplet(tx_set, eps, "{}->{}".format(eps, eps), eps)
			elif rules[tag] == '-sc+s[1:]':
				triplet = make_triplet(tx_set, eps, "{}->{}".format(stc[2][5], eps), oup[stc[1][3]:])
			elif rules[tag] == '~sc+s':
				triplet = make_triplet(tx_set, eps, "{}->{}".format(inp, oup[:stc[2][3]]), oup[stc[2][3]:])
			elif rules[tag] == '~sc+s-3':
				triplet = make_triplet(tx_set, eps, "{}->{}".
									   format(inp[stc[1][1]:], oup[stc[1][1]:stc[3][3]]), oup[stc[3][3]:])
			elif rules[tag] == 'sc->rer:s->i':
				triplet = make_triplet(tx_set, eps, "{}->{}".
									   format(inp[stc[1][1]:stc[3][2]],
											  oup[stc[1][3]:stc[3][4]]),
									   oup[stc[5][3]:])
			elif rules[tag] == '-sc+s-1':
				if len(mb) != 0:
					triplet = make_triplet(tx_set, eps, "{}->{}".format(inp[stc[1][1]:], eps), oup[mb[0][1][3]:])
				else:
					triplet = make_triplet(tx_set, eps, "{}->{}".
										   format(inp[stc[1][1]:], oup[stc[1][3]:stc[-2][4]]), oup[stc[-1][3]:])
			elif rules[tag] == '-sc+s-2':
				triplet = make_triplet(tx_set, eps, "{}->{}".format(inp, oup[:stc[-1][3]]), oup[stc[-1][3]:])
				triplets.append(triplet)
				pref = stc[0][6][:-1]
				prefix_flag = False
				triplet = make_triplet(tx_set, pref,
									   "{}->{}".format(inp, oup[:stc[-1][3]].replace(pref, "")), oup[stc[-1][3]:])
			elif rules[tag] == '-sc-':
				triplet = make_triplet(tx_set, eps, "{}->{}".format(inp[stc[1][1]:], eps), eps)
			elif rules[tag] == '~sc+s:-2':
				triplet = make_triplet(tx_set, eps, "{}->{}".format(inp, oup[:stc[-2][3]]), oup[stc[-2][3]:])
			elif rules[tag] == 'spe':
				if len(mb) == 1:
					triplet = make_triplet(tx_set, eps, "{}->{}".
										   format(inp[stc[1][1]:], eps), oup[stc[1][3]:])
				else:
					triplet = make_triplet(tx_set, eps, "{}->{}".format(inp, oup[:len(inp) - 2]), oup[len(inp) - 2:])
					triplets.append(triplet)
					triplet = make_triplet(tx_set, eps, "{}->{}".format(inp, oup[:len(inp) - 1]), oup[len(inp) - 1:])

		elif len(inp.split(" ")) < len(oup.split(" ")):
			oups = oup.split(" ")
			for ind_o, oup_s in enumerate(oups):
				pref = ' '.join(oups[0:ind_o])
				pref = pref if len(pref) > 0 else eps
				prefix_flag = False if pref == eps else prefix_flag
				oup_r = oup.replace(pref, "")
				triplet = make_triplet(tx_set, pref, "{}->{}".format(inp, oup_r), eps)
				triplets.append(triplet)
				triplet = make_triplet(tx_set, pref, "{}->{}".format(inp, oup_r[:-1]), oup[-1:])
				triplets.append(triplet)
				triplet = make_triplet(tx_set, pref, "{}->{}".format(inp, oup_r[:-2]), oup[-2:])
		else:
			triplet = make_triplet(tx_set, eps, "{}->{}".format(inp, oup), eps)
			triplets.append(triplet)
			if len(oup) > 2:
				triplet = make_triplet(tx_set, eps, "{}->{}".format(inp, oup[:-1]), oup[-1:])
				triplets.append(triplet)
				triplet = make_triplet(tx_set, eps, "{}->{}".format(inp, oup[:-2]), oup[-2:])
				triplets.append(triplet)
				triplet = make_triplet(tx_set, eps, "{}->{}".format(inp, oup[:-3]), oup[-3:])
		if prefix_flag:
			triplet['p'] = i_stc[6]
			triplet['sc'] = triplet['sc'].replace(triplet['p'], "")
		triplets.append(triplet)
	return triplets


def get_chars(l):
	flat_list = [char for word in l for char in word]
	return list(set(flat_list))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("datapath", help="path to data", type=str)
	parser.add_argument("language", help="language", type=str)
	args = parser.parse_args()

	DATA_PATH = args.datapath
	L2 = args.language
	LOW_PATH = os.path.join(DATA_PATH, L2 + "-train")

	lowi, lowo, lowt = read_data(LOW_PATH)
	# lowi, lowo, lowt = lowi[:100], lowo[:100], lowt[:100]
	# devi, devo, devt = read_data(DEV_PATH)

	vocab = get_chars(lowi + lowo)
	triplets = augment(lowi, lowo, lowt, vocab)
	with codecs.open(os.path.join(DATA_PATH, L2 + "-hall"), 'w', 'utf-8') as outp:
		for val in triplets:
			outp.write(val['inp'] + '\t' + val['oup'] + '\t' + val['pos'] + '\t' +
					   val['p'] + '\t' + val['sc'] + '\t' + val['s'] + '\n')
