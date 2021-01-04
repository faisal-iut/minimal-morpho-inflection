import codecs
import os
import stanza
import logging
import multiprocessing
from functools import partial
import pickle
import json
from log import logger
# logger = logging.getLogger('MyLogger')
logger.setLevel(logging.DEBUG)

# !cat data/en.txt | sacremoses -l en -j 4 \
#     normalize -c tokenize -a truecase -a -m data/en.truemodel \
#     > data/en.txt.norm.tok.true

# !cat data/es.txt | sacremoses -l es -j 4 \
#     normalize -c tokenize -a truecase -a -m data/es.truemodel \
#     > data/es.txt.norm.tok.true


def file_read(path, filename, formats):
    if formats=='codecs':
        with codecs.open(os.path.join(path, filename), 'r', 'utf-8') as inp:
            data = inp.read().split("\n")
        return data

def file_write(path, filename, formats, data):
    if formats=='codecs':
        with codecs.open(os.path.join(path, filename), 'w') as writer:
            writer.write(data)

def split_translation(path, filename):
    with codecs.open(os.path.join(path, filename), 'r', 'utf-8') as inp:
        data = inp.read().split("\n")
    src_data = [line.split(' |||')[0] for line in data]
    dst_data = [line.split('||| ')[1] if len(line.split('|||'))>1 else False for line in data]
    return src_data, dst_data

# dst_data, src_data = split_translation('./data/Europarl3/spa_eng/', 'Europarl3.orig.spa-eng')

def morph_info(start_end,dic,task,nlp):
    sp_doc = nlp(dic[start_end[0]:start_end[1]])
    if task=='token':
        eng_doc1 = [" ".join([token.text for i,token in enumerate(sent.tokens)]) for sent in sp_doc.sentences]
    if task=='lemma':
        eng_doc1 = [" ".join([word.lemma for i,word in enumerate(sent.words)]) for sent in sp_doc.sentences]
    if task=='morph':
        eng_doc1 = [["{}||{}||{}||{}||{}".format(word.text, word.lemma, word.upos, word.xpos, word.feats)
                                for i,word in enumerate(sent.words)] for sent in sp_doc.sentences]
    logger.info("length: {} done".format(start_end))
    return eng_doc1

def lemma_morph_data(en_text,lang,task):
    logger.debug("debug message")
    doc_len = len(en_text)
    if task =='token':
        nlp = stanza.Pipeline(lang=lang, processors='tokenize', tokenize_no_ssplit = True)
    else:    
        nlp = stanza.Pipeline(lang=lang, tokenize_pretokenized=True, tokenize_no_ssplit = True, 
                            processors='tokenize,mwt,pos,lemma')
    curr_len = 0
    n_l = 100000
#     n_l = 500
    eng_tokens = []
    start_end = []
    i=0
    while curr_len<doc_len:
        prev_len = curr_len
        curr_len = curr_len+n_l
        if curr_len<=doc_len-2 and en_text[curr_len]!='\n' and en_text[curr_len+1]!='\n':
            while curr_len<=doc_len-2 and en_text[curr_len]!='\n' and en_text[curr_len+1]!='\n':
                curr_len=curr_len+1
            curr_len=curr_len+1
        if curr_len>doc_len:
            start_end.append((prev_len,doc_len))
#             eng_doc = nlp(en_text[prev_len:doc_len])
        else:
            start_end.append((prev_len,curr_len))
#             eng_doc = nlp(en_text[prev_len:curr_len])
#         if task=='lemma':
#             eng_doc1 = [" ".join([word.lemma for i,word in enumerate(sent.words)]) for sent in eng_doc.sentences]
#         elif task == 'morph':
#             eng_doc1 = [["{}||{}||{}||{}||{}".format(word.text, word.lemma, word.upos, word.xpos, word.feats)
#                         for i,word in enumerate(sent.words)] for sent in eng_doc.sentences]
        logger.debug("currently processing: {}".format(curr_len))
    logger.info("length list: {}".format(start_end))
    p = multiprocessing.Pool(processes = multiprocessing.cpu_count()-1)
    sent_x=partial(morph_info, dic=en_text, task=task, nlp=nlp)
    eng_tokens = p.map(sent_x, start_end)
    p.close()
    p.join()
#         logger.debug("currently processing: {}".format(curr_len))
    flattens = [tok2 for tok1 in eng_tokens for tok2 in tok1]
    return flattens

def join_splitted(src_file, dst_file, filepath = './data/'):      
    src_tok_true = file_read(filepath, src_file, 'codecs')
    dst_tok_true = file_read(filepath, dst_file, 'codecs')  
    if (len(src_tok_true)!=len(dst_tok_true)):
        logger.debug('src:{} dst:{} lenghts are not equal'.format(len(src_tok_true),len(dst_tok_true)))
        return None
    logger.debug('src:{} dst:{} lenghts are equal'.format(len(src_tok_true),len(dst_tok_true)))
    src_dst_tok_true = "\n".join("{0} ||| {1}".format(x,y) for x,y in zip(src_tok_true,dst_tok_true))
    return src_dst_tok_true

# load_save_tokenize('es.txt.norm.tok.true', 'en.txt.norm.tok.true', 'tokenized')

def load_lemmatize(lang): 
    filepath = './data/'    
    L=lang+'.txt.norm.tok.true'
    tok_true = file_read(filepath, L, 'codecs')
    data = '\n\n'.join(tok_true)
    data_lemma = lemma_morph_data(data, lang, 'lemma')
    data_lemma = "\n".join(data_lemma)
    return data_lemma


if __name__ == "__main__":
    filepath = './data/eng_V'
    # operation = 'src_morph'
    operation = 'join_splitted'
#     src_file = load_lemmatize('es')
#     file_write(filepath, 'es_tok_lemma.txt', 'codecs', src_file)
#     dst_file = load_lemmatize('en')
#     file_write(filepath, 'en_tok_lemma.txt', 'codecs', dst_file)
##### save tokenized parallel data
#     es_en_joined_tokenized = join_splitted('es.txt.norm.tok.true', 'en.txt.norm.tok.true') 
#     file_write(filepath,'spa-eng-{}.spa-eng'.format('tokenized'),'codecs', es_en_joined_tokenized)
##### save lemmatized parallel data
#     es_en_joined_lemmatized = join_splitted('es_tok_lemma.txt', 'en_tok_lemma.txt') 
#     file_write(filepath,'spa-eng-{}.spa-eng'.format('lemmatized'),'codecs', es_en_joined_lemmatized)


##### Construct tokenized data
    if operation=='token':
        lang =  'es'
        src_file = 'corp_sent_detok_marian.{}_V'.format(lang)
        dst_file =  '{}.txt.norm.tok.true'.format(lang)
        src_data =  file_read(filepath, src_file, 'codecs')
        src_data_joined = '\n\n'.join(src_data)
        src_tok_data =  lemma_morph_data(src_data_joined, lang, 'token')
        src_tok_data_joined = '\n'.join(src_tok_data)
        file_write(filepath, dst_file,'codecs', str(src_tok_data_joined))
        logger.info("{} data tokenization done and file saved".format(lang))

        lang =  'en'
        src_file = 'corp_sent_detok.{}_V'.format(lang)
        dst_file =  '{}.txt.norm.tok.true'.format(lang)
        src_data =  file_read(filepath, src_file, 'codecs')
        src_data_joined = '\n\n'.join(src_data)
        src_tok_data =  lemma_morph_data(src_data_joined, lang, 'token')
        src_tok_data_joined = '\n'.join(src_tok_data)
        file_write(filepath, dst_file,'codecs', str(src_tok_data_joined))
        logger.info("{} data tokenization done and file saved".format(lang))
        print()


##### constract and save source side morphological data
    if operation=='src_morph':
        lang = 'es'
        src_tok_true = file_read(filepath, '{}.txt.norm.tok.true'.format(lang), 'codecs')
        src_tok_true = '\n\n'.join(src_tok_true)
        src_morph_lemma = lemma_morph_data(src_tok_true, lang, 'morph')
    #     print(en_morph_lemma)
        with open (os.path.join(filepath,'{}_morph_lemma_marian.pickle'.format(lang)),'wb') as inp:
            pickle.dump(src_morph_lemma, inp)
        # file_write(filepath,'{}_morph_lemma_marian.txt','codecs', str(en_morph_lemma))
    
    if operation =='join_splitted':
        src_lang = 'es'
        dest_lang = 'en'
        src_file = 'es.txt.norm.tok.true'
        dst_file = 'en.txt.norm.tok.true'
        file_form = 'tokenized'
        src_dst_joined = join_splitted(src_file, dst_file, filepath)
        file_write(filepath,'{}-{}-{}.{}-{}'.format(src_lang, dest_lang,file_form, src_lang, dest_lang),'codecs', src_dst_joined)



