import sys

import numpy as np
import pandas as pd
from create_description.utilities import Enti, write_to_file, entity_text_process, relation_text_process,clean
import ast
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import threading, time


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        # time.sleep(2)
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except Exception:
            return None


def read_ent_rel_2id(data_id_paht):
    data = pd.read_csv(data_id_paht)  #
    data = np.array(data)
    data_id = []
    for i in range(len(data)):
        _tmp = data[i][0]
        tmp = _tmp.split('\t')
        if tmp:
            id_list = []
            for s in tmp:
                id_list.append(s)
            data_id.append(id_list)
    data = np.array(data_id)
    return data


def read_data2id(data_id_paht):
    data = pd.read_csv(data_id_paht)  #
    data = np.array(data)
    data_id = []
    for i in range(len(data)):
        _tmp = data[i][0]
        tmp = _tmp.split(' ')
        if tmp:
            id_list = []
            for s in tmp:
                id_list.append(s)
            data_id.append(id_list)
    data = np.array(data_id)
    return data


def read_entity_description(entity_description):
    """
    14344(index) 	/m/0wsr(symbol) 	 Atlanta Falcons(label)	 American football team (description)
    :param entity_obj_path:
    :return:
    """
    f = open(entity_description, encoding='utf-8')

    x_obj = []
    for d in f:
        d = d.strip()
        if d:
            d = d.split('\t')

            elements = []
            for n in d:
                elements.append(n.strip())
            d = elements
            x_obj.append(d)
    f.close()
    return x_obj


def set_entity_description_obj(entity_des):
    all_entity_description_list = []
    entity_description_list = []


    for i in range(len(entity_des)):

        print(i)
        entity_id = entity_des[i][0]
        symbol = entity_des[i][1]
        name = entity_des[i][2]
        mention = "that is " + entity_des[i][3]
        import ast
        neighbours_data = ast.literal_eval(entity_des[i][4])  # "list" -> list

        if len(neighbours_data) == 0:
            neighbours = ['the entity has not neighbours']
        else:
            neighbours = neighbours_data

        id2vector = np.random.rand(10)

        # en_des = str(symbol) + '$' + str(name) + '$' + str(mention) + '$' + str(neighbours)
        en_des = str(symbol) + '$' + str(mention) + '$' + str(neighbours)

        entity_des_word_list = entity_text_process(en_des)  # get entity des 's word list

        entity = Enti(_id=entity_id, _symbol=symbol, _label=name, _mention=mention, _neighbours=neighbours,
                      _entity2vec=id2vector, _entity_des_word_list=entity_des_word_list)

        en_des_word_list = entity.get_entity_description()
        all_entity_description_list.append(en_des_word_list)

        entity_description_list.append(entity)

    print("len(all_entity_description_list) : \n", len(all_entity_description_list))

    word_bag = []
    all_word_dic = []
    pre_word_embedding = []
    print("len(word_bag)", len(word_bag))

    return entity_description_list, all_entity_description_list, word_bag, all_word_dic, pre_word_embedding

def obtain_all_entity_resource(entity_description_path):

    """obtain entity description"""

    entity_description = read_entity_description(entity_description_path)  # read original entity description

    all_entity_description_obj, all_entity_description_word_list, all_word_bag, all_word_bag_dic, pre_trained_word_embedding = set_entity_description_obj(
        entity_description)  # set entity object

    re = {'all_entity_description_obj': all_entity_description_obj,
          'all_entity_description_word_list': all_entity_description_word_list, 'all_word_bag': all_word_bag,
          'all_word_bag_dic': all_word_bag_dic, 'pre_trained_word_embedding': pre_trained_word_embedding}

    return re


def obtain_each_relation_description(head_en_obj, tail_en_obj):
    pass


def get_hrt_description_embedding(_h, _r, _t, entity_res, relation2id):
    """
    :param _h: head index
    :param _r: relation index
    :param _t: tail index
    :param ret: entity resource
    :param relation2id: relation id and name
    :return: the word embedding of _h,_r, and _t,

     entity_res = {'all_entity_obj_list': all_entity_obj_list,
                    'all_entity_description_word_list': all_entity_description_word_list,
                    'all_word_bag': all_word_bag,
                    'all_word_bag_dic': all_word_bag_dic,
                    'pre_trained_word_embedding': pre_trained_word_embedding
                    }

    """
    char = " "

    all_entity_res_obj = entity_res['all_entity_obj_list']
    all_entity_des_word = entity_res['all_entity_description_word_list']
    pre_word_embedding = entity_res['pre_trained_word_embedding']
    word_bag = entity_res['all_word_bag_dic']

    head_index = _h
    tail_index = _t
    relation_index = _r

    head_obj = [all_entity_res_obj[i] for i in head_index]
    tail_obj = [all_entity_res_obj[i] for i in tail_index]

    head_description_list = [" ".join(all_entity_des_word[i]) for i in head_index]  # get head entity description

    tail_description_list = [" ".join(all_entity_des_word[i]) for i in tail_index]  # get tail entity

    relation_description_list = []  # get relation descriptions

    relation_name = relation2id[relation_index, 0]


    for i in range(len(relation_name)):

        rel_des = str(relation_name[i]) + ', ' + 'which is between ' + head_obj[i].label + ' and ' + tail_obj[
            i].label + ';' \
                  + head_obj[i].get_random_neighbour() + ';' + tail_obj[i].get_random_neighbour()
        # print("rel_des ", rel_des)
        relation_description_list.append(rel_des)


    relation_description_word_list = relation_text_process(relation_description_list)

    return head_description_list, relation_description_word_list,tail_description_list

if __name__ == "__main__":
    entity_description_path = "../benchmarks/FB15K/all_entity_description_3.txt"

    ret = obtain_all_entity_resource(entity_description_path)

    # ret = {'all_entity_description_obj': all_entity_description_obj,
    #       'all_entity_description_word_list': all_entity_description_word_list, 'all_word_bag': all_word_bag,
    #       'all_word_bag_dic': all_word_bag_dic, 'pre_trained_word_embedding': pre_trained_word_embedding}

    # number_of_entity = len(entity_description_obj)
    # print(number_of_entity)
    # for i in range(number_of_entity):
    #     tmp_en = entity_description_obj[i]
    #     entity_str = tmp_en.id + '\t' + tmp_en.symb + '\t' + tmp_en.label + '\t' + tmp_en.description
    #     print(entity_str)
    #     entity_des = tmp_en.get_entity_description()
    #     print("entity_des :\n ",entity_des)
    #
    #     print("all_entity_description_list :\n",all_entity_description_list[i])
    #     import time
    #     time.sleep(2)

    # word_bag_path = "./FB15K/word_bag.txt"
    # word_bag, pre_word_embedding = get_word2vec(word_bag_path)

    """
    Obtain entity2id ,relation2id, and train2id. (25 Mar)
    """
    # entity2id_path = "../benchmarks/FB15K/entity2id.txt"
    relation2id_path = "../benchmarks/FB15K/relation2id.txt"
    # train_id_path = "../benchmarks/FB15K/train2id.txt"

    # entity2id = read_ent_rel_2id(entity2id_path)
    relation2id = read_ent_rel_2id(relation2id_path)

    # train2id = read_data2id(train_id_path)

    # entity_description_obj = ret['all_entity_description_obj']
    # word_bag_dic = ret['all_word_bag_dic']
    # pre_word_embedding = ret['pre_trained_word_embedding']
    # get_triples_description(train2id, relation2id, entity_description_obj, word_bag_dic, pre_word_embedding)

    _h = [0, 2, 4, 6, 8, 10]
    _r = [0, 1, 2, 3, 4, 5]
    _t = [1, 3, 5, 7, 9, 11]

    h_des_init_embedding, r_des_init_embedding, t_des_init_embedding = get_hrt_description_embedding(_h, _r, _t, ret,
                                                                                                     relation2id)
