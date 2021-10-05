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
    word_to_idx = word_dict  # 数据集中所有的单词
    # print(len(word_to_idx))

    pretrained_embeddings = np.random.uniform(-0.5, 0.5, (len(word_dict), dim))

    # print("pretrained_embeddings.shape",pretrained_embeddings.shape)

    # word2vec = load_w2v_vec('./data/GoogleNews-vectors-negative300.bin', word_to_idx)
    word2vec = load_golve_vec(word_to_idx, dim)
    # print(len(word2vec))
    for word, vector in word2vec.items():  # 初始化每个词
        # print(word)
        # print(word_to_idx[word])
        pretrained_embeddings[word_to_idx[word]] = vector

    # 打印测试
    # print("NULL -> ",word_to_idx['NULL'],pretrained_embeddings[word_to_idx['NULL']])
    # print("contemporary -> ",pretrained_embeddings[30080])
    # print("the - > ",pretrained_embeddings[12104])
    # print("the - > ",pretrained_embeddings[word_to_idx['the']])

    # singer_index = word_to_idx['singer']
    # print(singer_index)
    # print("singer - > ",pretrained_embeddings[singer_index])
    # print("bryan - > ",pretrained_embeddings[25286])
    #
    import torch
    pretrained_embeddings = torch.as_tensor(pretrained_embeddings)
    return pretrained_embeddings

def get_des_embedding(pre_embeddings, word_bag, sentence_set):
    print("Begin get_des_embedding ... ")

    init_embedding = []
    for i in range(len(sentence_set)):
        word_index = [word_bag[x] for x in sentence_set[i]]

        tmp_embedding = pre_embeddings[word_index]
        # print(tmp_embedding)
        init_mean_embedding = np.mean(np.array(tmp_embedding), axis=0).tolist()
        # print("init_embedding mean :",init_mean_embedding)
        init_embedding.append(init_mean_embedding)

    print("Finish get_des_embedding ...")
    return init_embedding


