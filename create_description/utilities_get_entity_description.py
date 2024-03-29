# Obtain ID + Description + neighbor structure，and conduct word embedding（that is entity description）

import time

import numpy as np
import pandas as pd
import random
import os
from create_description.utilities import entity_text_process, constuct_entity_des,clean
import torch
import torchtext
import re
from create_description.utilities import Enti

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
        fopen_in = open(in_path, 'r', encoding='utf-8')
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
    return all_data, all_word


def read_file(in_path):
    """
    read training data (complex_triple2vector)
    :param in_path:
    :param all_data:  return data
    :return:
    """
    all_data = []

    try:
        fopen_in = open(in_path, 'r')
    except IOError as err:
        print('file open error: {0}'.format(err))
    else:
        for eachLine in fopen_in:
            if eachLine:
                each = eachLine.split(',')
                elements = []
                for n in each:
                    n = re.sub(r'[\[\]]', '', n)
                    elements.append(float(n))

                all_data.append(elements)
        fopen_in.close()

    print("read train data over! ")
    return pd.DataFrame(all_data)


def read_all_triples(path):
    f = open(path)
    x = []
    relation_set = []
    entity_set = []
    entityPair_set = []
    for d in f:
        d = d.strip()
        if d:
            d = d.split('\t')

            elements = []
            for n in d:
                elements.append(n.strip())
            d = elements

            x.append(d)
            relation_set.append(d[1])
            entity_set.append(d[0])
            entity_set.append(d[2])
            entityPair_set.append((d[0], d[2]))

    f.close()
    X = np.array(x)

    return X, relation_set, entity_set, set(entityPair_set)


def read_train_set(path):
    f = open(path)
    x = []
    relation_set = []
    h_entity_set = []
    t_entity_set = []
    entityPair_set = []
    for d in f:
        d = d.strip()
        if d:
            d = d.split('\t')

            elements = []
            for n in d:
                elements.append(n.strip())
            d = elements

            x.append(d)
            relation_set.append(d[1])
            h_entity_set.append(d[0])
            t_entity_set.append(d[2])
            entityPair_set.append((d[0], d[2]))

    f.close()
    X = np.array(x)
    return X, relation_set, h_entity_set, t_entity_set, entityPair_set


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


def read_entity2id(data_id_paht):
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
    data_id = np.array(data_id)
    return data_id


def read_entity2obj(entity_obj_path):
    """
    14344(index) 	/m/0wsr(symbol) 	 Atlanta Falcons(label)	 American football team (description)
    :param entity_obj_path:
    :return:
    """
    f = open(entity_obj_path)

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
    X = np.array(x_obj)

    return X

def obtain_multi_neighbours(current_entity_neighbours, entity_neighbours_dict,current_entity_neighbours_using_name,entity_neighbours_dict_using_name, num_neigs, num_step):

    num_neighbours = len(current_entity_neighbours)
    current_entity_des = []
    current_entity_des_using_name = []

    if num_neigs == "all" or num_neighbours <= num_neigs:
        for j in range(num_neighbours):

            _neighbour = current_entity_neighbours[j]
            _neighbour_using_name = current_entity_neighbours_using_name[j]

            if num_step == 1:
                current_entity_des.append(_neighbour)
                current_entity_des_using_name.append(_neighbour_using_name)

            else:
                for n in range(num_step - 1):
                    one_of_neighs_tail = _neighbour.split(" ")[-1]
                    """ choose randomly a neighbour as the second-step neighbours """
                    current_tail_neighbours = entity_neighbours_dict[one_of_neighs_tail]
                    current_tail_neighbours_using_name = entity_neighbours_dict_using_name[one_of_neighs_tail]

                    num_current_tail_neighbours = len(current_tail_neighbours)

                    if num_current_tail_neighbours == 0:
                        # _neighbour = _neighbour + ", and " + "it has not next neighbours"
                        _neighbour = _neighbour
                        _neighbour_using_name = _neighbour_using_name
                        break

                    elif num_current_tail_neighbours == 1:
                        next_step_neighbour = current_tail_neighbours[0]
                        _neighbour = _neighbour + ", and " + next_step_neighbour

                        next_step_neighbour_using_name = current_tail_neighbours_using_name[0]
                        _neighbour_using_name = _neighbour_using_name + ", and " + next_step_neighbour_using_name

                    else:
                        index = np.random.random_integers(0, num_current_tail_neighbours - 1)
                        next_step_neighbour = current_tail_neighbours[index]
                        _neighbour = _neighbour + ", and " + next_step_neighbour

                        next_step_neighbour_using_name = current_tail_neighbours_using_name[index]
                        _neighbour_using_name = _neighbour_using_name + ", and " + next_step_neighbour_using_name

                current_entity_des.append(_neighbour)
                current_entity_des_using_name.append(_neighbour_using_name)
    else:

        all_neighbours_index = [i for i in range(num_neighbours)]
        neighbours_index = random.sample(all_neighbours_index, num_neigs)

        for j in neighbours_index:
            _neighbour = current_entity_neighbours[j]
            _neighbour_using_name = current_entity_neighbours_using_name[j]

            if num_step == 1:
                current_entity_des.append(_neighbour)
                current_entity_des_using_name.append(_neighbour_using_name)

            else:
                for n in range(num_step - 1):
                    one_of_neighs_tail = _neighbour.split(" ")[-1]
                    """ choose randomly a neighbour as the second-step neighbours """
                    current_tail_neighbours = entity_neighbours_dict[one_of_neighs_tail]
                    current_tail_neighbours_using_name = entity_neighbours_dict_using_name[one_of_neighs_tail]

                    num_current_tail_neighbours = len(current_tail_neighbours)

                    if num_current_tail_neighbours == 0:
                        # _neighbour = _neighbour + ", and " + "it has not next neighbours"
                        _neighbour = _neighbour
                        _neighbour_using_name = _neighbour_using_name
                        break

                    elif num_current_tail_neighbours == 1:
                        next_step_neighbour = current_tail_neighbours[0]
                        _neighbour = _neighbour + ", and " + next_step_neighbour

                        next_step_neighbour_using_name = current_tail_neighbours_using_name[0]
                        _neighbour_using_name = _neighbour_using_name + ", and " + next_step_neighbour_using_name

                    else:
                        index = np.random.random_integers(0, num_current_tail_neighbours - 1)
                        next_step_neighbour = current_tail_neighbours[index]
                        _neighbour = _neighbour + ", and " + next_step_neighbour

                        next_step_neighbour_using_name = current_tail_neighbours_using_name[index]
                        _neighbour_using_name = _neighbour_using_name + ", and " + next_step_neighbour_using_name


                current_entity_des.append(_neighbour)
                current_entity_des_using_name.append(_neighbour_using_name)

    return current_entity_des,current_entity_des_using_name


def obtain_inverse_relations(X, all_tail, entity_symbol):

    # obtain forward neighbours for each entity
    entity_inverse_index_list = [i for i, x in enumerate(all_tail) if x == entity_symbol]  # Count neighbors
    each_entity_neighbours = len(entity_inverse_index_list)
    print("entity_inverse_index_list : ", each_entity_neighbours)

    # get neighbour ： relation  +  tail entity
    entity_neighbours = X[entity_inverse_index_list, 0:2].tolist()


    """
     /m/01xpxv has a inverse relation of /people/person/ethnicity with /m/0cn68.
     str1 = " has a inverse relation of "
     str2 =  " with "
     neighbour = entity_symbol + str1 +  entity_neighbours[i][0] + str2 + entity_neighbours[i][1]
    """

    """ create neighbours """
    str1 = " has a inverse relationship of "
    str2 = " with "

    str_entity_inverse_neighbours = []
    if len(entity_neighbours) == 0:
        str_entity_inverse_neighbours = []
    else:
        for k in range(len(entity_neighbours)):  # obtain all first-neighbours for the entity
            n_entity_symbol = entity_neighbours[k][0]  # obtain tail entity's ID
            neighbour = entity_symbol + str1 + entity_neighbours[k][1] + str2 + n_entity_symbol
            str_entity_inverse_neighbours.append(str(neighbour))

    return str_entity_inverse_neighbours


def obtain_entity_res(X, sub_x_obj, entity2name,_entity_set, num_neigs, num_step):
    print("obtain_entity_res ...")
    """
    :param X: triples
    :param sub_x_obj: there are 14515 entities have label and des.
    :param entity_set: there are 14951 entities

    :param num_neigs: how many neighbours
    :param num_step: how many steps
    :return: 14951 entity objects
    """
    entity_symbol_set = _entity_set
    num_entity = len(entity_symbol_set)
    all_head = list(X[:, 0])  # obtain all head entity and then get them neighbours

    all_tail = list(X[:, 2])  # # obtain all tail entity and then get them neighbours

    all_entity_obj_list = []
    all_entity_description_des_list = []
    all_entity_mention_des_list = []
    all_entity_name_des_list = []

    entity_neighbours_dict = {}
    entity_neighbours_dict_using_name = {}

    neighbours_tail_list_dic = {}
    s = 0
    n = 0
    max_num_neighbours = 0
    num_neighbours_list = []
    for i in range(num_entity):

        # print("---", i, "---")
        entity_symbol = entity_symbol_set[i]

        entity_index_list = [i for i, x in enumerate(all_head) if x == entity_symbol]  # Count neighbors
        each_entity_neighbours = len(entity_index_list)
        # print("entity_index_list : ", each_entity_neighbours)
        if each_entity_neighbours == 0:
            s += 1
        if each_entity_neighbours == 1:
            n += 1

        entity_neighbours = X[entity_index_list, 1:3].tolist()

        """
         /m/01xpxv has a relation of /people/person/ethnicity with /m/0cn68.
         str1 = " has a relation of "
         str2 =  " with "
         neighbour = entity_symbol + str1 +  entity_neighbours[i][0] + str2 + entity_neighbours[i][1]
        """

        str1 = " has a relationship of "
        str2 = " with "
        # neighbours_tail_list = []
        str_entity_neighbours = []
        str_entity_neighbours_using_name = []

        if len(entity_neighbours) == 0:
            str_entity_neighbours = []
            str_entity_neighbours_using_name = []
            # neighbours_tail_list = []
        else:
            for k in range(len(entity_neighbours)):
                n_entity_symbol = entity_neighbours[k][1]

                neighbour = entity_symbol + str1 + entity_neighbours[k][0] + str2 + n_entity_symbol
                neighbour_using_name = entity2name[entity_symbol] + str1 + entity_neighbours[k][0] + str2 + entity2name[n_entity_symbol]

                str_entity_neighbours.append(str(neighbour))
                str_entity_neighbours_using_name.append(str(neighbour_using_name))

        # combine forward and inverse neighbours
        entity_neighbours_dict[entity_symbol] = str_entity_neighbours
        entity_neighbours_dict_using_name[entity_symbol] = str_entity_neighbours_using_name

        num_neighbours = len(entity_neighbours_dict[entity_symbol])
        num_neighbours_list.append(num_neighbours)
        if num_neighbours > max_num_neighbours:
            max_num_neighbours = num_neighbours

    print("max the number of neighbours :", max_num_neighbours)

    """
    there are 724 entities have not neighbours and 700 entities have one neighbour.
    Consider using inverse relations to construct sentence.
    """

    """
    # Design method to encapsulate multi-step neighbors
    for /m/01_30_
    /m/01_30_ has a relationship of /business/business_operation/industry with /m/020mfr, and /m/020mfr has a ......

    """
    word_bag = [',', '.', 'None', 'has', 'a', 'relationship', 'of', 'with', 'which', 'is', 'between', 'and']


    symbol = list(sub_x_obj[:, 1])
    symbol_dic = {}
    for j in range(len(symbol)):
        symbol_dic[symbol[j]] = j

    print("obtain multi-step neighbours ... \n")
    for i in range(num_entity):
        entity_symbol = entity_symbol_set[i]

        # Take out the neighbors of the current entity, which is a list containing several neighbors
        current_entity_neighbours = entity_neighbours_dict[entity_symbol]
        current_entity_neighbours_using_name = entity_neighbours_dict_using_name[entity_symbol]

        current_entity_des = []
        current_entity_des_using_name = []
        num_neighbours = len(current_entity_neighbours)
        if num_neighbours == 0:

            current_entity_des = ['it has not neighbours']

            # current_entity_des_using_name = ['it has not neighbours']
            current_entity_des_using_name = []

        else:
            current_entity_des, current_entity_des_using_name = obtain_multi_neighbours(current_entity_neighbours, entity_neighbours_dict,current_entity_neighbours_using_name, entity_neighbours_dict_using_name ,num_neigs,num_step)
        # other attributes


        entity_id = i

        if entity_symbol in symbol_dic.keys():
            index = symbol_dic.get(entity_symbol)
            entity_name = sub_x_obj[index, 2]
            entity_mention = sub_x_obj[index, 3]
        else:
            entity_name = entity_symbol
            entity_mention = entity_symbol

        # if entity_symbol in symbol:
        #     index = symbol.index(entity_symbol)
        #     entity_name = sub_x_obj[index, 2]
        #     entity_mention = sub_x_obj[index, 3]
        # else:
        #     entity_name = entity_symbol
        #     entity_mention = entity_symbol

        """"
        word list encapsulation for entity description
        des = str(self.label) + '$' + str(self.description) + '$' + str(self.neighbours)
        """

        entity_description_des_word_list , entity_mention_des_list, entity_name_list = constuct_entity_des(entity_name, entity_mention, current_entity_des_using_name)  # get entity des 's word list

        word_bag += entity_description_des_word_list

        word_bag = list(set(word_bag))

        entity_id2vec = np.random.rand(10)

        # entity_des_word_list = []
        entity = Enti(_id=entity_id, _symbol=entity_symbol, _label=entity_name, _mention=entity_mention,
                      _neighbours=current_entity_des_using_name,
                      _entity2vec=entity_id2vec, _entity_mention_des_word_list = entity_mention_des_list,
                      _entity_description_des_word_list=entity_description_des_word_list)

        all_entity_obj_list.append(entity)
        all_entity_description_des_list.append(entity_description_des_word_list)
        all_entity_mention_des_list.append(entity_mention_des_list)
        all_entity_name_des_list.append(entity_name_list)

    all_word_dic = {}
    for i in range(len(word_bag)):
        all_word_dic[i] = word_bag[i]

    pre_word_embedding = []
    return all_entity_obj_list, all_entity_description_des_list,all_entity_mention_des_list, all_entity_name_des_list, word_bag,num_neighbours_list


def write_train_set():
    pass


def write_triples2vector(_triple2vector):
    out_path = "./FB15K/all_complex_triple2vector.txt"
    fobj = open(out_path, 'a+')

    fobj.writelines('%s\n' % _triple2vector)

    fobj.close()


def write_to_file(out_path, all_data):
    ls = os.linesep

    try:
        fobj = open(out_path, 'w')
    except IOError as err:

        print('file open error: {0}'.format(err))

    else:

        fobj.writelines('%s\n' % x for x in all_data)

        fobj.close()

    print('WRITE FILE DONE!')


def write_to_file_entity_obj(out_path, all_data):
    ls = os.linesep

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for x in all_data:
            #
            _str = str(x.id) + '\t' + str(x.symb) + '\t' + str(x.label) + '\t' + str(x.description) + '\t' + str(
                x.neighbours) + '\n'

            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')


def write_entity2vec(out_path, all_data, entity_set):
    ls = os.linesep

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for x in all_data:
            _str = str(x.id) + '\t' + str(x.symb) + '\t' + str(x.entity2vec) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')


if __name__ == "__main__":


    train_path = './FB15K/all_triples.txt'
    set_entity_obj_path = './FB15K/all_entity_description_name_mention_des_n20s_1.txt'
    entity2vec_path = './FB15K/all_entity2id_randomly_vector_7.txt'

    # obtain firstly entity2Obj，ikg_data_processing
    entity_obj_path = './FB15K/entity2Obj.txt'
    X, relation_set, entity_set, entityPair_set = read_all_triples(train_path)

    # print("data details")
    # print(X.shape)
    # print(len(relation_set))
    # print(len(entity_set))
    # print("read_train_set ->  over ! ")

    sub_x_obj = read_entity2obj(entity_obj_path)  # 14515 entities have label and des , and about 436 has not desc...
    print("read_entity_obj ->  over ! ")
    relation_set = list(set(relation_set))
    entity2id_path = "./FB15K/entity2id.txt"
    entity_id_read_file = read_entity2id(entity2id_path)
    entity_id_set = entity_id_read_file[:, 0].tolist()
    """
    set_entity_obj     14951 entities and its description
    all_entity2vec_set 14951 randomly generate vector as entity id
    """
    entity_obj_list, all_entity_description_list = obtain_entity_res(X, sub_x_obj, entity_id_set)
    print("set_entity_obj ->  over ! ")
    write_to_file_entity_obj(set_entity_obj_path, entity_obj_list)
    print("write set_entity_obj  ->  over ! ")

    print("=========OVER==========")
