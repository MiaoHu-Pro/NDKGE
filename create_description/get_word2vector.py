
import numpy as np
import pandas as pd
from torch.nn import Embedding

from text_analytics.text_analytics.text_analytics import text_analytics
import ast
import torch.nn as nn
import torch

import torchtext

EMBEDDING_DIM = 300
glove = torchtext.vocab.GloVe(name="6B", dim=EMBEDDING_DIM)

import time

def read_word_bag(in_path):
    """
    read training data (complex_triple2vector)
    :param in_path:
    :param all_data:  return data
    :return:
    """
    print("read word_bag \n")
    all_data = []
    all_word = {}

    try:
        fopen_in = open(in_path,  'r',encoding='utf-8')
    except IOError as err:
        print('file open error: {0}'.format(err))
    else:
        i = 0
        for eachLine in fopen_in:
            if eachLine:
                each = eachLine.strip()
                all_data.append(each)
                all_word[each] = i
                i += 1

        fopen_in.close()

    print("read train data over! ")
    # all_word_bag = all_data
    return all_data,all_word

def load_w2v_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    print("load bin vec \n")

    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        c = 0
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)

            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                c +=1
            else:

                f.read(binary_len)
    return word_vecs

def load_golve_vec(word_list):

    print("load bin vec \n")

    word_vectors = {}
    c = 0
    for w in glove.itos:
        if w in word_list:
            word_vectors[w] = glove[w]
            c += 1
    # print(len(word_list), c, len(word_list) - c)

    return word_vectors

def word2vector(word_dict):
    """

    """
    print("Word to vector ... \n")
    # word2vector
    word_to_idx = word_dict

    pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(word_dict), EMBEDDING_DIM))

    # word2vec = load_w2v_vec('./data/GoogleNews-vectors-negative300.bin', word_to_idx)
    word2vec = load_golve_vec(word_to_idx)

    for word, vector in word2vec.items():

        pretrained_embeddings[word_to_idx[word]] = vector

    #
    pretrained_embeddings = torch.as_tensor(pretrained_embeddings)
    return pretrained_embeddings.to(torch.float32)

def get_word2vec(word_bag_path):

    wb_list,all_word_dic = read_word_bag(word_bag_path)

    pre_train_embeddings = word2vector(all_word_dic)

    return all_word_dic,pre_train_embeddings

    # print(wb_list)
    # print(len(wb_list))

def get_sentence_init_embedding(pre_embeddings,word_bag,sentence_set):

    # print("get_sentence_init_embedding ... ")

    # print("\n",len(sentence_set))

    # print("sentence_set",sentence_set)
    length_list = [len(x) for x in sentence_set]
    # print(length_list)
    max_num_words = max(length_list) # get max number of words for each batch

    # print(max_num_words)#
    # time.sleep(3)

    sentence_word_index_set = []
    other_words = []
    for i in range(len(sentence_set)):

        if len(sentence_set[i]) <= max_num_words:
            sentence_set[i] = sentence_set[i] + ["NULL"] * (max_num_words - len(sentence_set[i]))
        else:
            sentence_set[i] = sentence_set[i][:max_num_words]

        # print("sentence_set[i]",sentence_set[i])

        for x in sentence_set[i]:
            if word_bag.get(x):
                sentence_word_index_set.append(word_bag[x])
            else:
                other_words.append(x)
                sentence_word_index_set.append(word_bag["NULL"])

    sentence_word_index_set = np.array(sentence_word_index_set).T

    tensor_sentence_word_index_set = torch.as_tensor(sentence_word_index_set)

    init_embedding = pre_embeddings[tensor_sentence_word_index_set].view(max_num_words, len(sentence_set), -1)

    return init_embedding

if __name__=="__main__":

    """word2vector"""
    pass
    # word_bag_path = "FB15K/word_bag.txt"
    #
    # word_bag, pre_embeddings = get_word2vec(word_bag_path)
    #
    # # pre_embeddings = torch.as_tensor(pre_word2vec)
    #
    # # pre_embeddings = nn.Embedding(len(word_bag), 300)
    #
    # sentence_set = [['/m/06cx9', 'republic', 'form', 'of', 'state'],
    #                  ['/m/06cx9', 'has', 'a', 'relationship', 'of', 'government','form', 'of', 'government'],
    #                  ['countries', 'with', '/m/02lx0', '/m/06cx9', 'has', 'a','relationship', 'of', 'government', 'form', 'of'],
    #                  ['government', 'countries','with', '/m/06f32', '/m/06cx9', 'has', 'a', 'relationship', 'of', 'government', 'form', 'of', 'government', 'countries','with', '/m/036b_'],
    #                  ['/m/06cx9', 'has', 'a', 'relationship', 'of', 'government', 'form', 'of', 'government', 'countries', 'with', '/m/04hqz', '/m/06cx9', 'has', 'a', 'relationship', 'of', 'government', 'form', 'of', 'government', 'countries', 'with', '/m/016wzw']]
    #
    # get_sentence_init_embedding(pre_embeddings, word_bag, sentence_set)
    #
    # """
    # Next, put it into LSTM
    # lstm_out, self.hidden = self.lstm(x, self.hidden)
    # """




