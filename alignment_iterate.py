from log import logger
from ast import literal_eval 
import pandas as pd
import numpy as np
import warnings
import pickle
import sys
import os
import itertools
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
from collections import Counter
from nltk import FreqDist
from boltons.iterutils import remap
import copy
if not sys.warnoptions:
    warnings.simplefilter("ignore")
eps = '\u0395'

import logging
from log import logger
# logger = logging.getLogger('MyLogger')
logger.setLevel(logging.INFO)

def get_affix(strings, counts = 10):
    suffixDist=FreqDist()
    prefixDist=FreqDist()
    strings = [s for s in strings if type(s)!=int and len(s)>1]
    for word in strings:
        for i in range(1,4):
            prefixDist[word[:i]] +=1
            suffixDist[word[-i:]] +=1
    s3=[]
    s2=[]
    s1=[]
    cs = suffixDist.most_common(100)
    cp = prefixDist.most_common(100)
    dist_dict ={}
    for i in range(2,4):
        dist_dict['s{}'.format(i)] = [(s,c) for s,c in cs if len(s)==i][:counts]
        dist_dict['p{}'.format(i)] = [(p,c) for p,c in cp if len(p)==i][:counts]
    return dist_dict

def feats_transform_phase2(df, gpdf, wfs):
    s_2 = defaultdict(list)
    s_3 = defaultdict(list)
    for i in range(6, len(gpdf.columns)):
        affixes = get_affix(gpdf.iloc[:,i], 100)
        for v1,v2 in affixes['s2']:
            s_2[v1].append((i,int(v2)))
        for k, v in s_2.items():
            l =  sorted(v, key = lambda x: x[1], reverse =  True)
            s_2[k] = l
        for v1,v2 in affixes['s3']:
            s_3[v1].append((i,int(v2)))
        for k, v in s_3.items():
            l =  sorted(v, key = lambda x: x[1], reverse =  True)
            s_3[k] = l
    gpdf['other']= 0
    column = gpdf.columns[-1]
    for wf in wfs:
        if len(wf)>=3:
            s2 = wf[-2:]
            s3 = wf[-3:]
            if s3 in s_3.keys():
                column  = s_3[s3][0][0]
            elif s2 in s_2.keys():
                column  = s_2[s2][0][0]
            ###work needs to be done for other exception like: words with len<3
        dst_lemma  = df.loc[df['dst_word']==wf]['dst_lemma'].values[0]
        src_lemma  = df.loc[df['dst_word']==wf]['src_lemma'].values[0]
        row = gpdf.loc[(gpdf['dst_lemma']==dst_lemma) & (gpdf['src_lemma']==src_lemma)].index[0]
        position =  gpdf.iloc[row, column]
        if position==0:
            gpdf.iloc[row, column] = wf
        else:
            gpdf.loc[row, 'other'] = wf 
    return gpdf

def feats_transform_phase1(df):
    feats =  df.feats.unique()
    n_ds = np.zeros(shape=(df.shape[0],df.shape[1]+len(feats)-1))
    gdf = pd.DataFrame(df.groupby(['src_lemma','dst_lemma']).aggregate('first')).reset_index()
    pdf = pd.pivot_table(df, values='dst_word', index=['src_lemma','dst_lemma'],
                    columns=['feats'], aggfunc='first', fill_value=0).reset_index()
    gpdf = pd.merge(gdf, pdf[pdf.columns[2:]], left_index=True, right_index=True,how='inner')
    logger.info("feature transform phase 1 done")
    
    feat_vals  = gpdf.iloc[:,6:].values.flatten()
    feat_vals = set(feat_vals[feat_vals!=0])
    orig_vals  = set(df['dst_word'])
    left_wfs =  orig_vals.difference(feat_vals)
    logger.info("after phase1 transformation: left_wfs: {}, feat_vals: {}, orig_vals: {}".format(len(left_wfs), 
                                                                    len(feat_vals), len(orig_vals)))
    gpdf_2 = feats_transform_phase2(df, gpdf, left_wfs)
    feat_vals  = gpdf_2.iloc[:,6:].values.flatten()
    feat_vals = set(feat_vals[feat_vals!=0])
    orig_vals  = set(df['dst_word'])
    left_wfs = orig_vals.difference(feat_vals)
    logger.info("after phase2 transformation: left_wfs: {}, feat_vals: {}, orig_vals: {}".format(len(left_wfs), 
                                                                        len(feat_vals), len(orig_vals)))
    return gpdf_2

# def feats_transform(df):
#     feats =  df.feats.unique()
#     n_ds = np.zeros(shape=(df.shape[0],df.shape[1]+len(feats)-1))
#     gdf = pd.DataFrame(df.groupby(['src_lemma','dst_lemma']).aggregate('first')).reset_index()
#     pdf = pd.pivot_table(df, values='dst_word', index=['src_lemma','dst_lemma'],
#                     columns=['feats'], aggfunc='first', fill_value=0).reset_index()
#     gpdf = pd.merge(gdf, pdf[pdf.columns[2:]], left_index=True, right_index=True,how='inner')
#     logger.info("feature transform done")
#     return gpdf

def convert(lst): 
    return literal_eval(lst) 


def do_alignment(src_all, dst_all, alignment):
    src_form_lemma = {}
    src_lemma_count = defaultdict(int)
    src_dst_align = {}
    dst_src_align = {}
    dst_upos = {}
    dst_xpos = {}
    dst_feats = {}
    # pos_allowed = ['NOUN','VERB']
    for i, align in enumerate(alignment):
        try:
            if len(align)>1:
                for al in align.split(" "):
                    src,dst = al.split('-')
                    src,dst =  int(src), int(dst)
                    try:
                        src_morph_len  = len(src_all[i][src].split('||'))
                        if src_morph_len==5:
                            src_form, src_lemma, src_upos, src_xpos, src_feats = src_all[i][src].split('||')
                            dst_word = dst_all[i][dst]
                            # if src_upos in pos_allowed:
                            c=src_dst_align.setdefault(src_lemma,{})
                            c[dst_word]=c.setdefault(dst_word,0)+1
                            c=dst_src_align.setdefault(dst_word,{})
                            c[src_lemma]=c.setdefault(src_lemma,0)+1
                            
                            c=dst_upos.setdefault(dst_word,{})
                            c[src_upos]=c.setdefault(src_upos,0)+1
                            
                            c=dst_xpos.setdefault(dst_word,{})
                            c[src_xpos]=c.setdefault(src_xpos,0)+1
                            
                            c=dst_feats.setdefault(dst_word,{})
                            c[src_feats]=c.setdefault(src_feats,0)+1
                            
                            src_form_lemma[src_form] = src_lemma
                            src_lemma_count[src_lemma] = src_lemma_count[src_lemma]+1
                    except IndexError:
                        logger.info("Missalignment: length of en_all[{}]:{}, en:{}".format(i,len(src_all[i]),src))
#                         logger.info("en_all[{}]:{}".format(i,en_all[i][en]))
        except ValueError:
            logger.info("Value error: en_all[{}]:{}".format(i,src_all[i]))
    logger.info("do alignment done")
    return src_dst_align, dst_src_align, dst_upos, dst_xpos, dst_feats, src_form_lemma, src_lemma_count

def construct_table(dst_src_align, dst_lemma, dst_upos, dst_xpos, dst_feats):
    dst_property = [['dst_word','src_lemma','dst_lemma','upos','xpos','feats']]
    for dst,srcs in dst_src_align.items():
        #find en
        opt_src = max(srcs.items(), key=lambda k: k[1])
        #find lemma
        opt_lemma = max(dst_lemma[dst].items(), key=lambda k: k[1])
        #find upos
        opt_upos =  max(dst_upos[dst].items(), key=lambda k: k[1])
        #find xpos
        opt_xpos =  max(dst_xpos[dst].items(), key=lambda k: k[1])
        #find feats
        opt_feats =  max(dst_feats[dst].items(), key=lambda k: k[1])
        dst_property.append([dst, opt_src[0], opt_lemma[0], opt_upos[0], opt_xpos[0], opt_feats[0]])
    dst_property_df = pd.DataFrame.from_records(dst_property[1:],columns=dst_property[0])
    dst_property_df = dst_property_df.sort_values(by=['dst_lemma'])
    logger.info("construct table done")
    return dst_property_df

def common(l,m):
    l_o = l
    if not l: return []
    if m is not None: l.sort()
    lm = list(filter(lambda x: len(x)>=m, l))
    left = set(l).difference(set(lm))
    lemma_words = [(os.path.commonprefix(g),list(set(g))) if len(os.path.commonprefix(g))>4 
          else (min(filter(None, g), key=len),list(set(g)))
            for g in [list(g) for k, g in groupby(lm, itemgetter(0,1,2))]]
    lw_tuples = []
    for lemma, words in lemma_words:
        lw_tuples.extend(list(zip(itertools.repeat(lemma),words)))
    others = list(zip(left,left))
    if len(others)!=0:
        lw_tuples.extend(others)
#     print(l_o)
    lw_tuples.sort(key=lambda x: l_o.index(x[1]))
    return lw_tuples

def lemma_cluster_2nd(dst_property):
    grouped = dst_property.groupby(['src_lemma'])
    for name,group in grouped:
        lemma_l = list(group['dst_lemma'].values)
        nl,ol = zip(*common(lemma_l,4))
        c = [nl[list(ol).index(a)] for a in lemma_l]
        dst_property['dst_lemma'] = dst_property['dst_lemma'].replace(lemma_l, c)
    dst_property = dst_property.sort_values(by=['dst_lemma'])
    logger.info("lemma clustering 2nd done")
    return dst_property



def dict_key_filter(obj, obj_filter):
    def inner_dict_key_filter(obj): return dict_key_filter(obj, obj_filter)
    def to_keep(subtree): return not isinstance(subtree, (dict, list)) or subtree

    def build_subtree(key, value):
        if key in obj_filter:
            return copy.deepcopy(value) # keep the branch
        elif isinstance(value, (dict, list)):
            return inner_dict_key_filter(value) # continue to search
        return [] # just an orphan value here

    if isinstance(obj, dict):
        key_subtree_pairs = ((key, build_subtree(key, value)) for key, value in obj.items())
        return {key:subtree for key, subtree in key_subtree_pairs if to_keep(subtree)}
    elif isinstance(obj, list):
        return list(filter(to_keep, map(inner_dict_key_filter, obj)))
    return []

def wfs_filter(wfs, src_lemma_count, src_dst_align, dst_src_align):
    src_lemma_counter = Counter(src_lemma_count)
    #50 most frequent data
    src_lemma_50 = [a for a,b in src_lemma_counter.most_common(len(src_lemma_count)//2) if len(a)>2 and a.isalpha()]
    #cluster in es lemma
    lemma_cluster=[]
    for key, value in src_dst_align.items():
        # if 'alaron' in list(value.keys()):
        #     print(key, value)
        lemma_cluster.extend(common(list(value.keys()),4))
    logger.info("lemma clustering done")
    #es lemma-word dict
    dst_lemma = {}
    for k, v in lemma_cluster:
        c=dst_lemma.setdefault(v,{})
        c[k]=c.setdefault(k,0)+1

    dst_src_align = dict_key_filter(dst_src_align, wfs)
    src_dst_align = dict_key_filter(src_dst_align, wfs)

    logger.info("filter data done")
    with open('./data/temp.pickle','wb') as op:
        pickle.dump(dst_lemma, op)
    return src_lemma_50, dst_lemma, src_dst_align, dst_src_align

def lemma_filter_cluster(en_lemma_count, en_es_align, es_en_align, freq_to_discard):
    en_lemma_counter = Counter(en_lemma_count)
    #50 most frequent data
    en_lemma_50 = [a for a,b in en_lemma_counter.most_common(len(en_lemma_count)//2) if len(a)>2 and a.isalpha()]
    #cluster in es lemma
    lemma_cluster=[]
    for key, value in en_es_align.items():
        lemma_cluster.extend(common(list(value.keys()),4))
    logger.info("lemma clustering done")
    #es lemma-word dict
    src_lemma = {}
    for k, v in lemma_cluster:
        c=src_lemma.setdefault(v,{})
        c[k]=c.setdefault(k,0)+1
    #discard words based on freq_to_discard
    en_es_align = remap(en_es_align, lambda p, k, v: v not in freq_to_discard and v!={})
    es_en_align = remap(es_en_align, lambda p, k, v: v not in freq_to_discard and v!={})
    logger.info("filter data done")
    return en_lemma_50, src_lemma, en_es_align, es_en_align


def do_iteration(dst_all, src_all, alignment, freq_order_reduce):
###################
# ##  load alignment data
#     alignment = file_read(filepath, alignfile, 'codecs')
#     logger.info("alignment length:{}".format(len(alignment)))
#     #load english and spanish parallel data
#     en_data, es_data = split_translation(filepath, parallelfile)
#     logger.info("es data length:{} en data length: {}".format(len(en_data),len(es_data)))
#    #load english morphological data
#     with codecs.open(os.path.join(filepath, en_morph_file), 'r', 'utf-8') as inp:
#             en_morph_lemma = inp.read()
#     en_all = convert(en_morph_lemma)
#     logger.info("en morphological data length:{}".format(len(en_all)))
####################
    
#    #create data dictionaries
#     es_all = [line.split(" ") for line in es_data]
#     logger.info("es_all:{} en_all:{} alignment:{}".format(len(es_all),len(en_all),len(alignment)))
    en_es_align, es_en_align, es_upos, es_xpos, es_feats, en_form_lemma, en_lemma_count = do_alignment(src_all, 
                                                                                                   dst_all, 
                                                                                                   alignment)
    #lemma filter
#     freq_order_reduce = [0,1]

    lexica_path = './data/Lexica/lex.eng_V'
    from data_manager import Lexicon
    gold = Lexicon(lexica_path)
    wfs = set(gold.wf_2_lem)
    my_wfs = set(es_en_align)
    common_wfs =  wfs.intersection(my_wfs)
    en_lemma_50, src_lemma, en_es_align, es_en_align = wfs_filter(wfs, en_lemma_count, en_es_align, es_en_align)

    # en_lemma_50, src_lemma, en_es_align, es_en_align = lemma_filter_cluster(en_lemma_count, 
    #                                                                     en_es_align, 
    #                                                                     es_en_align,freq_order_reduce)

######################
    #construct table
    es_property = construct_table(es_en_align, src_lemma, es_upos, es_xpos, es_feats)
    logger.info("es prperty dataframe shape:{}".format(es_property.shape))
    es_property.to_csv('./data/dst_property.csv')
    ### 2nd filter based on wfs value
    x = set(es_property['es_word']).difference(wfs)
    for val in x:
        index_name =  es_property.loc[es_property['es_word']==val].index
        es_property.drop(index_name, inplace= True)
    #2nd order lemma cluster
    es_property = lemma_cluster_2nd(es_property)
    es_property.to_csv('./data/dst_property_cluster2.csv')
    es_property = es_property[es_property['dst_lemma'].isin(en_lemma_50)]
    es_nouns = es_property.loc[es_property['upos']=='NOUN'].copy()
    logger.info("es_nouns dataframe shape:{}".format(es_nouns.shape))
    es_nouns_t = feats_transform(es_nouns)
    logger.info("es_nouns_t dataframe shape:{}".format(es_nouns_t.shape))
    es_verbs = es_property.loc[es_property['upos']=='VERB'].copy()
    logger.info("es_verbs dataframe shape:{}".format(es_verbs.shape))
    es_verbs_t = feats_transform(es_verbs)
    logger.info("es_verbs_t dataframe shape:{}".format(es_verbs_t.shape))
    es_form_lemma = dict(zip(es_property['es_word'].values,es_property['src_lemma'].values))
    en_form_lemma_f = dict((k, v) for k,v in en_form_lemma.items() if v in en_lemma_50)
    return es_property, es_nouns_t, es_verbs_t, es_form_lemma, en_form_lemma_f, en_es_align, es_en_align