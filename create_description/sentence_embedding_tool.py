import os
import numpy as np
import pandas as pd
import ast
import sys

sys.path.append("../")

import torchtext
def load_golve_vec(word_list, _dim):
    glove = torchtext.vocab.GloVe(name="6B", dim=_dim)

    print("load bin vec \n")
    word_vectors = {}

    for w in glove.itos:
        if w in word_list:
            word_vectors[w] = glove[w]

    return word_vectors

def word2vector(word_dict, dim):
    """

    """
    print("Word to vector ... \n")
    # word2vector
    word_to_idx = word_dict
    # print(len(word_to_idx))

    pretrained_embeddings = np.random.uniform(-0.5, 0.5, (len(word_dict), dim))

    # print("pretrained_embeddings.shape",pretrained_embeddings.shape)

    # word2vec = load_w2v_vec('./data/GoogleNews-vectors-negative300.bin', word_to_idx)
    word2vec = load_golve_vec(word_to_idx, dim)
    # print(len(word2vec))
    for word, vector in word2vec.items():
        # print(word)
        # print(word_to_idx[word])
        pretrained_embeddings[word_to_idx[word]] = vector

    import torch
    pretrained_embeddings = torch.as_tensor(pretrained_embeddings)
    return pretrained_embeddings

def get_des_embedding(pre_embeddings, word_bag, sentence_set):
    print("Begin get_des_embedding ... ")

    init_embedding = []
    sentence_word_index_set = []
    other_words = []
    for i in range(len(sentence_set)):
        # word_index = [word_bag[x] for x in sentence_set[i]]
        for x in sentence_set[i]:
            if word_bag.get(x):
                sentence_word_index_set.append(word_bag[x])
            else:
                other_words.append(x)
                sentence_word_index_set.append(word_bag["NULL"])

        tmp_embedding = pre_embeddings[sentence_word_index_set]
        # print(tmp_embedding)
        init_mean_embedding = np.mean(np.array(tmp_embedding), axis=0).tolist()
        # print("init_embedding mean :",init_mean_embedding)
        init_embedding.append(init_mean_embedding)

    print("Finish get_des_embedding ...")
    return init_embedding


