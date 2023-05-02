# define a class,ERDes,to finish the task of obtaining entity and relation description

# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from tqdm import tqdm
from create_description.sentence_embedding_tool import word2vector, get_des_embedding
from create_description.utilities import clean

use_gpu = False


def to_var(x):
    return Variable(torch.from_numpy(x).to(device))


sys.path.append("../")
from create_description.utilities_get_entity_description import read_all_triples, read_entity2obj, read_entity2id, \
    obtain_entity_res, read_ent_rel_2id
from create_description.utilities_get_triples_description import get_hrt_description_embedding
from create_description.utilities import read_data2id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    use_gpu = True


def to_var(x):
    return Variable(torch.from_numpy(x).to(device))


def write_to_file_the_number_of_num_neighbours(out_path, data):
    ls = os.linesep
    num = len(data)
    try:
        fobj = open(out_path, 'w')
        fobj.writelines('%s\n' % num)
    except IOError as err:

        print('file open error: {0}'.format(err))

    else:

        fobj.writelines('%s\n' % x for x in data)

        fobj.close()

    print('WRITE FILE DONE!')


def write_to_file_entity_description_des(out_path, all_data):
    ls = os.linesep
    char = " "

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for x in all_data:
            # _str = str(x.id) + '\t' + str(x.symbol) + '\t' + str(x.label) + '\t' + str(x.mention) + '\t' + char.join(
            #     x.neighbours) + '\n'
            #
            _str = str(x.id) + '\t' + str(x.symbol) + '\t' + char.join(x.entity_description_des) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')


def write_to_file_entity_mention_des(out_path, all_data):
    ls = os.linesep
    char = " "

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for x in all_data:
            #
            _str = str(x.id) + '\t' + str(x.symbol) + '\t' + char.join(x.entity_mention_des) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')


def write_word_bag(out_path, all_data):
    ls = os.linesep

    try:
        fobj = open(out_path, 'w')
    except IOError as err:

        print('file open error: {0}'.format(err))

    else:

        fobj.writelines('%s\n' % x for x in all_data)

        fobj.close()

    print('WRITE FILE DONE!')


def write_triple_descriptions(out_path, head_des, rel_des, tail_des):
    num_triples = len(head_des)
    ls = os.linesep
    head_len = []
    rel_len = []
    tail_len = []
    char = " "

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for i in range(num_triples):
            head = head_des[i]
            rel = rel_des[i]
            tail = tail_des[i]
            head_len.append(len(head))
            rel_len.append(len(rel))
            tail_len.append(len(tail))
            _str = str(i) + '\t' + char.join(head) + '\t' + char.join(rel) + '\t' + char.join(tail) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')
    print("head len ", np.mean(head_len))
    print("rel len ", np.mean(rel_len))
    print("tail len ", np.mean(tail_len))


class ERDes(object):

    def __init__(self, _Paras):
        self.num_neighbours = _Paras['num_neighbours']
        self.num_step = _Paras['num_step']
        self.word_dim = _Paras['word_dim']
        self.all_triples_path = _Paras['all_triples_path']
        self.entity2Obj_path = _Paras['entity2Obj_path']
        self.entity2id_path = _Paras['entity2id_path']
        self.relation2id_path = _Paras['relation2id_path']
        self.file_path = _Paras['file_path']
        self.entity_description_des_path = _Paras['entity_description_des_path']
        self.entity_mention_des_path = _Paras['entity_mention_des_path']
        self.word_bag_path = _Paras['word_bag_path']
        self.the_number_of_neighbours_path = _Paras['the_number_of_neighbours']
        self.entity_res = None
        self.entity_symbol_set = []
        self.entity_word_bag = []
        self.all_word_bag = []
        self.all_word_dic = []

        self.all_entity_name_des_list = []
        self.all_entity_description_word_list = []
        self.all_entity_mention_des_list = []

        self.all_relation_description_list = []
        self.pre_word_embedding = None
        self.relation2id = []  # ['/location/country/form_of_government' '0']

    def pre_process(self):

        self.all_entity_description_word_list, self.all_entity_mention_des_list, self.all_entity_name_des_list, self.entity_word_bag = self._get_entity_des()
        self.all_relation_description_list, self.all_word_bag = self._get_relation_des()
        self.pre_word_embedding, self.all_word_dic = self._get_pre_trained_word_embedding()

    def _get_entity_des(self):
        print("get_entity_des ...")

        X, relation_set, entity_set, entityPair_set = read_all_triples(self.all_triples_path)
        sub_x_obj = read_entity2obj(
            self.entity2Obj_path)

        relation_set = list(set(relation_set))  # all relation
        entity_set = list(set(entity_set))  # all entity

        # obtain entity id
        entity_id_read_file = read_entity2id(self.entity2id_path)
        self.entity_symbol_set = entity_id_read_file[:, 0].tolist()

        # entity2name
        entity2name = {}  # '/m/03_48k': 'Fred Ward'
        for i in range(len(sub_x_obj)):
            entity2name[sub_x_obj[i][1]] = sub_x_obj[i][2]

        """
        get the description of the entity
        set_entity_obj     14951 entities and its description
        all_entity2vec_set 14951 randomly generate vector as entity id
        """
        all_entity_obj_list, all_entity_description_word_list, all_entity_mention_des_list, all_entity_mame_des_list, entity_word_bag, num_neighbours_list = \
            obtain_entity_res(
                X, sub_x_obj, entity2name, self.entity_symbol_set, self.num_neighbours, self.num_step)

        print("entity_word_bag : ", len(entity_word_bag))

        self.entity_res = {'all_entity_obj_list': all_entity_obj_list,
                           'all_entity_description_word_list': all_entity_description_word_list,
                           'all_word_bag': entity_word_bag,
                           'neighbours_num': num_neighbours_list
                           }

        self.relation2id = read_ent_rel_2id(self.relation2id_path)  # ['/location/country/form_of_government' '0']
        write_to_file_entity_description_des(self.entity_description_des_path, all_entity_obj_list)
        write_to_file_entity_mention_des(self.entity_mention_des_path, all_entity_obj_list)
        write_to_file_the_number_of_num_neighbours(self.the_number_of_neighbours_path, num_neighbours_list)

        print("=========OVER==========")

        return all_entity_description_word_list, all_entity_mention_des_list, all_entity_mame_des_list, entity_word_bag

    def _get_relation_des(self):

        # ['/location/country/form_of_government' '0']
        self.relation2id = read_ent_rel_2id(self.relation2id_path)
        relation_id = self.relation2id[:, 1]
        relation_name = self.relation2id[:, 0]
        relation_description_list = []
        rel_word_bag = []
        for i in range(len(relation_id)):
            rel_des = str(relation_name[i])
            rel_des_word = clean(rel_des)
            rel_word_bag += rel_des_word
            rel_word_bag = list(set(rel_word_bag))
            relation_description_list.append(rel_des_word)

        all_word_bag = list(set(rel_word_bag + self.entity_word_bag))
        write_word_bag(self.word_bag_path, all_word_bag)

        return relation_description_list, all_word_bag

    def _get_pre_trained_word_embedding(self):

        """ read word bag """

        all_word_dic = {}
        print(len(self.all_word_bag))
        for i in range(len(self.all_word_bag)):
            all_word_dic[self.all_word_bag[i]] = i
        pre_word_embedding = word2vector(all_word_dic, dim=self.word_dim)

        return pre_word_embedding, all_word_dic

    def get_name_embedding(self):
        """
        name
        :return: entity_name embedding , relation embedding
        """

        entity_name_init_embedding = get_des_embedding(self.pre_word_embedding, self.all_word_dic,
                                                       self.all_entity_name_des_list)
        relation_init_embedding = get_des_embedding(self.pre_word_embedding, self.all_word_dic,
                                                    self.all_relation_description_list)

        np.savetxt(self.file_path + 'entity_name_init_embedding_' + str(self.word_dim) + '.txt',
                   entity_name_init_embedding, fmt='%.5f', delimiter=',')
        np.savetxt(self.file_path + 'relation_name_init_embedding_' + str(self.word_dim) + '.txt',
                   relation_init_embedding, fmt='%.5f', delimiter=',')

        print("obtain init embedding of entity and relation ... ")

        return np.array(entity_name_init_embedding), np.array(relation_init_embedding)

    def get_mention_embeddings(self):
        """
        name + mention
        :return: entity_mention embedding , relation embedding
        """

        entity_men_init_embedding = get_des_embedding(self.pre_word_embedding, self.all_word_dic,
                                                      self.all_entity_mention_des_list)
        relation_init_embedding = get_des_embedding(self.pre_word_embedding, self.all_word_dic,
                                                    self.all_relation_description_list)

        np.savetxt(self.file_path + 'entity_mention_init_embedding_' + str(self.word_dim) + '.txt',
                   entity_men_init_embedding, fmt='%.5f', delimiter=',')
        np.savetxt(self.file_path + 'relation_mention_init_embedding_' + str(self.word_dim) + '.txt',
                   relation_init_embedding, fmt='%.5f', delimiter=',')

        print("obtain init embedding of entity and relation ... ")

        return np.array(entity_men_init_embedding), np.array(relation_init_embedding)

    def get_description_embeddings(self):
        """
        name + mention + new des
        :return: entity_description embedding , relation embedding
        """

        # _str = char.join(x.entity_des) + '\n'
        entity_des_init_embedding = get_des_embedding(self.pre_word_embedding, self.all_word_dic,
                                                      self.all_entity_description_word_list)

        relation_init_embedding = get_des_embedding(self.pre_word_embedding, self.all_word_dic,
                                                    self.all_relation_description_list)

        np.savetxt(self.file_path + 'entity_description_init_embedding_' + str(self.word_dim) + '.txt',
                   entity_des_init_embedding, fmt='%.5f', delimiter=',')
        np.savetxt(self.file_path + 'relation_description_init_embedding_' + str(self.word_dim) + '.txt',
                   relation_init_embedding, fmt='%.5f', delimiter=',')

        print("obtain init embedding of entity and relation ... ")

        return np.array(entity_des_init_embedding), np.array(relation_init_embedding)

    def get_triple_des(self, _h, _r, _t):

        h_des, r_des, t_des = get_hrt_description_embedding(_h, _r, _t, self.entity_res, self.relation2id)
        return h_des, r_des, t_des

    def er_des_print(self):
        print(self.entity2id_path)


def obtain_train_triple_des(file_path, en_rel_des):
    print("obtain_train_triple_des ... \n")
    train_data_set_path = file_path + 'train2id.txt'
    train = read_data2id(train_data_set_path)
    h = train[:, 0].tolist()
    _h = [int(h[i]) for i in range(len(h))]
    t = train[:, 1].tolist()
    _t = [int(t[i]) for i in range(len(t))]
    r = train[:, 2].tolist()
    _r = [int(r[i]) for i in range(len(r))]

    """
    Error getting word vector
    The reason is that the thesaurus does not cover all the words described in entity and relation/
    """

    h_des, r_des, t_des = en_rel_des.get_triple_des(_h, _r, _t)

    write_triple_descriptions(file_path + 'train_triple_des_4num_2step.txt', h_des, r_des, t_des)


def obtain_valid_triple_des(file_path, en_rel_des):
    print("obtain_valid_triple_des ... \n")
    valid_data_set_path = file_path + 'valid2id.txt'
    valid = read_data2id(valid_data_set_path)
    h = valid[:, 0].tolist()
    _h = [int(h[i]) for i in range(len(h))]
    t = valid[:, 1].tolist()
    _t = [int(t[i]) for i in range(len(t))]
    r = valid[:, 2].tolist()
    _r = [int(r[i]) for i in range(len(r))]

    h_des, r_des, t_des = en_rel_des.get_triple_des(_h, _r, _t)

    write_triple_descriptions(file_path + 'valid_triple_des_4num_2step.txt', h_des, r_des, t_des)


def obtain_test_triple_des(file_path, en_rel_des):
    print("obtain_test_triple_des ... \n")
    test_data_set_path = file_path + 'test2id.txt'
    test = read_data2id(test_data_set_path)
    h = test[:, 0].tolist()
    _h = [int(h[i]) for i in range(len(h))]
    t = test[:, 1].tolist()
    _t = [int(t[i]) for i in range(len(t))]
    r = test[:, 2].tolist()
    _r = [int(r[i]) for i in range(len(r))]

    h_des, r_des, t_des = en_rel_des.get_triple_des(_h, _r, _t)

    write_triple_descriptions(file_path + 'test_triple_des_4num_2step.txt', h_des, r_des, t_des)


if __name__ == "__main__":
    file_path = '../benchmarks/WN18RR/'
    Paras = {
        'num_neighbours': 'all',
        'num_step': 1,
        'word_dim': 300,
        'all_triples_path': file_path + 'train.tsv',
        'entity2Obj_path': file_path + 'ID_Name_Mention.txt',
        'entity2id_path': file_path + 'entity2id.txt',
        'relation2id_path': file_path + 'relation2id.txt',
        'word_bag_path': file_path + 'word_bag.txt',
        'entity_description_des_path': file_path + 'entity2new_description_des_all_nums_1step.txt',
        'entity_mention_des_path': file_path + 'entity2new_mention_des_all_nums_2step.txt',
    }

    en_rel_des = ERDes(_Paras=Paras)
    en_rel_des.pre_process()
    ent_mention_init_embedding, rel_mention_init_embedding = en_rel_des.get_mention_embeddings()
    ent_description_init_embedding, rel_description_init_embedding = en_rel_des.get_description_embeddings()

    # en_rel_des.get_mention_embeddings()
    # en_rel_des.get_description_embeddings()
    # train
    # obtain_train_triple_des(file_path, en_rel_des)
    # valid
    # obtain_valid_triple_des(file_path, en_rel_des)
    # test
    # obtain_test_triple_des(file_path, en_rel_des)
