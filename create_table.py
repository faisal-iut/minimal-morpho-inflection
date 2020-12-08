from log import logger
import codecs
import os
from create_lemmatized_data import file_read, split_translation
from alignment_iterate import convert, do_iteration, \
    do_alignment, wfs_filter, construct_table, lemma_cluster_2nd, feats_transform_phase1
import pickle
import pandas as pd
import numpy as np

if __name__ == "__main__":
    mname = 'marian'
    it='0'
    src_lang = 'es'
    dst_lang = 'en'
    filepath = './data/eng_V/'
    freq_order_reduce = [0,1]
    logger.info("Iteration: {}".format(it))

    ##  load alignment data
    alignfile =  '{}_{}.align'.format(src_lang, dst_lang)#'es-en.align'
    alignment = file_read(filepath, alignfile, 'codecs')
    if alignment[len(alignment)-1]=='\n' or alignment[len(alignment)-1]=='':
        alignment = alignment[:-1]
    logger.info("alignment length:{}".format(len(alignment)))

    ## load src and dst parallel data
    file_form = 'tokenized'
    parallelfile = '{}-{}-{}.{}-{}'.format(src_lang, dst_lang, file_form, src_lang, dst_lang)  ##es-en-tokenized.es-en 
    src_data, dst_data = split_translation(filepath, parallelfile)
    if src_data[-1]=='\n' or src_data[-1]=='' or src_data[-1]==False:
        src_data = src_data[:-1] 
    if dst_data[-1]=='\n' or dst_data[-1]=='' or dst_data[-1]==False:
        dst_data = dst_data[:-1]   
    logger.info("dst data length:{}, src data length: {}".format(len(dst_data),len(src_data)))
    dst_all = [line.split(" ") for line in dst_data]
    logger.info("dst_all:{}, alignment:{}".format(len(dst_all),len(alignment)))

    ## load english morphological data
    with open(os.path.join(filepath, '{}_morph_lemma_{}.pickle'.format(src_lang, mname)),'rb') as op:
        src_all = pickle.load(op)
        logger.info("src morphological data length:{}".format(len(src_all)))

    ## load given lexicon
    lexica_path = './data/Lexica/lex.eng_V'
    from data_manager import Lexicon
    gold = Lexicon(lexica_path)
    wfs = set(gold.wf_2_lem)

    ## create all data dictionary
    src_dst_align, dst_src_align, dst_upos, dst_xpos, dst_feats, src_form_lemma, src_lemma_count = do_alignment(src_all, 
                                                                                                dst_all, 
                                                                                                alignment)

    ## cluster and then filter based on given lexicon
    src_lemma_50, dst_lemma, src_dst_align, dst_src_align = wfs_filter(wfs, src_lemma_count, src_dst_align, dst_src_align)

    ##construct dest_table properties
    dst_property = construct_table(dst_src_align, dst_lemma, dst_upos, dst_xpos, dst_feats)
    logger.info("dst prperty dataframe shape:{}".format(dst_property.shape))
    dst_property.to_csv(os.path.join(filepath,'dst_property.csv'))
    logger.info(dst_property.values[:10])

    ## 2nd order lemma clustering and wfs filtering
    x = set(dst_property['dst_word']).difference(wfs)
    for val in x:
        index_name =  dst_property.loc[dst_property['dst_word']==val].index
        dst_property.drop(index_name, inplace= True)
    dst_property = lemma_cluster_2nd(dst_property)
    dst_property.to_csv(os.path.join(filepath,'dst_property_cluster2.csv'))
    logger.info("dst prperty dataframe shape after cluster_2:{}".format(dst_property.shape))

    ## transform dst property table: feats goes to column headers
    dst_property_t = feats_transform_phase1(dst_property)
    dst_property_t.to_csv(os.path.join(filepath,'dst_property_transformed.csv'))
    with open(os.path.join(filepath, 'dst_property_t.pickle'),'wb') as oup:
        pickle.dump(dst_property_t, oup)
    logger.info("dst prperty transformed dataframe shape:{}".format(dst_property_t.shape))

    ## run paradigm table construction operation    
    # dst_property, dst_nouns_t, dst_verbs_t, dst_form_lemma, \
    #     src_form_lemma, src_dst_align, dst_src_align = do_iteration(dst_all, src_all, alignment,freq_order_reduce)
    # with open(os.path.join(filepath, 'dst_property_wfs.pickle'),'wb') as oup:
    #     pickle.dump(dst_property, oup)
    # # print()
    # # temp  = feats_transform(dst_property)
    print()





