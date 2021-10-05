import os
import numpy as np
import time
import pandas as pd
import re
import math
import ast

# get_write_ini_embedding
# this py read embedding
"""
technology：
Tensor.detach().numpy().tolist() -> tensor become list
"[1,2,3,4,5]" --> [1,2,3,4,5]
1.
import ast
data_new = ast.literal_eval(data)
2.
answer_start = eval(answer_start)
:return: str_list -> list
"""


def read_train_valid_test_id(data_id_paht):
    data = pd.read_csv(data_id_paht)  #
    data = np.array(data)

    data_id = []
    for i in range(len(data)):
        _tmp = data[i][0]
        tmp = _tmp.split(' ')
        if tmp:
            id_list = []
            for s in tmp:
                id_list.append(int(s))
            data_id.append(id_list)

    data_id = np.array(data_id)
    return data_id


def read_init_embs(in_enti_path, in_rel_path):
    ini_ent = []
    ini_rel = []
    with open(in_rel_path) as f:
        for each_line in f:
            each_line = each_line.strip()
            if each_line:
                eachline_list = each_line.split('\t')
                str_embed = eachline_list[2]
                embed = ast.literal_eval(str_embed)  # str -> list
                ini_rel.append(embed)

                # if embed:
                #     each = embed.split(',')
                #     elements = []
                #     for n in each:
                #         n = re.sub(r'[\[\]]', '', n)
                #         elements.append(n)
                #
                #     ini_rel.append(elements,)

        with open(in_enti_path) as f:
            for each_line in f:
                each_line = each_line.strip()
                if each_line:
                    eachline_list = each_line.split('\t')
                    embed = eachline_list[2]
                    if embed:
                        each = embed.split(',')
                        elements = []
                        for n in each:
                            n = re.sub(r'[\[\]]', '', n)
                            if math.isnan(float(n)):
                                n = 0
                            elements.append(n)
                        ini_ent.append(elements)

    return np.array(ini_ent, dtype=np.float32), np.array(ini_rel, dtype=np.float32)





def read_new_init_embs(in_enti_path, in_rel_path):

    print("read my word embedding input...")

    init_entity_arr = pd.read_csv(in_enti_path,header=None)
    init_rel_arr = pd.read_csv(in_rel_path,header=None)

    init_entity_arr = np.array(init_entity_arr, dtype=np.float32)
    init_rel_arr = np.array(init_rel_arr, dtype=np.float32)

    return init_entity_arr,init_rel_arr

def advance_read_init_embs():
    data = "[168, 169, 170, 171, 172, 174, 185, 187, 159]"
    print("way 1")
    print(type(data))  #
    data_new = ast.literal_eval(data)
    print(type(data_new))  #
    print(data_new)  #
    print("way 2")
    print(type(data))  #
    data_new = eval(data)
    print(type(data_new))  #
    print(data_new)  #


def get_entityPair(train_data):
    entity_pair = []
    for i in range(len(train_data)):
        entity_pair.append((train_data[i, 0], train_data[i, 1]))

    entity_pair_set = list(set(entity_pair))
    index = []
    train_entity_pair = []
    for i in range(len(entity_pair_set)):
        print(i)
        tmp_entity_pair = []
        idx = [j for j, x in enumerate(entity_pair) if x == entity_pair_set[i]]

        index = index + idx
        tmp_data = train_data[idx]
        h = tmp_data[0][0]
        r_set = list(tmp_data[:,2])
        t = tmp_data[0][1]

        tmp_entity_pair.append(h)
        tmp_entity_pair.append(r_set)
        tmp_entity_pair.append(t)

        train_entity_pair.append(tmp_entity_pair)

        # print("tmp_entity_pair",tmp_entity_pair)

    print("len(index)", len(index))
    return train_entity_pair

def get_entityNeighbours(train_id):

    head_id = list(train_id[:, 0])
    tail_id = list(train_id[:, 1])
    all_entity = list(set(head_id + tail_id))
    # print(all_entity)
    train_entity_neighbours = []

    for i in all_entity:
        print(i)
        index = [j for j, x in enumerate(head_id) if x == i]
        reverse_index = [j for j, x in enumerate(tail_id) if x == i]
        if len(index) != 0:
            tmp_train_id = train_id[index, :]
            # print("tmp_train_id",tmp_train_id)
            tmp = []
            for j in range(len(tmp_train_id)):
                r = tmp_train_id[j][2]
                t = tmp_train_id[j][1]
                tmp.append((r,t))
        else:
            tmp = []

        if len(reverse_index) != 0:

            inverse_tmp_train_id = train_id[reverse_index, :]
            # print("inverse_tmp_train_id",inverse_tmp_train_id)

            inverse_tmp = []
            for j in range(len(inverse_tmp_train_id)):
                r = inverse_tmp_train_id[j][2]
                h = inverse_tmp_train_id[j][0]
                inverse_tmp.append((r,h))

        else:
            inverse_tmp = []
        # print("tmp",len(tmp))
        # print("inverse_tmp",len(inverse_tmp))
        i_neighs = [i,tmp,inverse_tmp]

        train_entity_neighbours.append(i_neighs)
        # print(i_neighs)
        # print("====================")

    return train_entity_neighbours

def write_initi_embedding(init_o, init_embedding, out_path, out_new_id):
    try:
        fobj = open(out_path, 'w')
        fobj_newi2 = open(out_new_id, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for i in range(len(init_o)):
            _str = str(i) + '\t' + str(init_o[i]) + '\t' + str(list(init_embedding[i])) + '\n'
            fobj.writelines('%s' % _str)

            _str_id = str(i) + '\t' + str(init_o[i]) + '\n'
            fobj_newi2.writelines('%s' % _str_id)

        fobj.close()
        fobj_newi2.close()

    print('WRITE FILE DONE!')

def write_train_entity_neighbours(path,data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:

        header_str = " entity_id " + '\t' + "neighbours_set" + '\t' + "inverse_neighbours_set" + '\n'
        fobj.writelines('%s' % header_str)

        for i in range(len(data)):

            _str = str(data[i][0]) + '\t' + str(data[i][1]) + '\t' + str(data[i][2])+ '\n'
            fobj.writelines('%s' % _str)

        fobj.close()


    print('WRITE FILE DONE!')

def write_train_entity_pairs(path,data):

    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:

        header_str = " head_id " + '\t' + "relation_set" + '\t' + "tail_id" + '\n'
        fobj.writelines('%s' % header_str)

        for i in range(len(data)):

            _str = str(data[i][0]) + '\t' + str(data[i][1]) + '\t' + str(data[i][2])+ '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')

def entity_id_symbol_label_description(entity2id_path,entity2obj_path):

    entity = []
    entity_id = []

    entity2obj = []
    entity2obj_ent = []

    entity_id_symbol_label_des = []

    with open(entity2id_path) as f:
        for each_line in f:
            each_line = each_line.strip()
            if each_line:
                eachline_list = each_line.split('\t')
                if (len(eachline_list) != 2):
                    continue
                entity.append(eachline_list[0])
                entity_id.append(int(eachline_list[1]))

    with open(entity2obj_path) as f:
        for each_line in f:
            each_line = each_line.strip()
            if each_line:
                eachline_list = each_line.split('\t')
                syb = eachline_list[1].strip()
                lab = eachline_list[2].strip()
                des = eachline_list[3].strip()

                entity2obj.append([syb,lab,des])

                entity2obj_ent.append(syb)

    print("len entity2obj",len(entity2obj))
    print("entity2obj[0]",entity2obj[0])

    print("entity ",entity[14950])
    print("entity_id",entity_id[14950])

    # print("entity2obj_ent",entity2obj_ent)


    lab = "label is None"
    des = "description is None"

    for i in range(len(entity_id)):
        tmp_id_sy_ds  =[]
        if entity[i] in entity2obj_ent:

            index = entity2obj_ent.index(entity[i])
            tmp_id_sy_ds.append(i)

            tmp_id_sy_ds = tmp_id_sy_ds + entity2obj[index]

        else:

            tmp_id_sy_ds.append(i)

            tmp_id_sy_ds = tmp_id_sy_ds + [entity[i],lab,des]

        entity_id_symbol_label_des.append(tmp_id_sy_ds)

    # for i in range(len(entity_id)):
    #
    #     print(entity_id_symbol_label_des[i])

    print(len(entity_id_symbol_label_des))
    return entity_id_symbol_label_des

def write_entity_id_symbol_label_description_set(path,data):


    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for i in range(len(data)):

            _str = str(data[i][0]) + '\t' + str(data[i][1]) + '\t' + str(data[i][2])+ '\t' + str(data[i][3]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')





def test():
    score = 1
    ave_out = [score for i in range(20)]
    print(ave_out)

def read_transe_out_embs(out_ent_path,out_rel_path):

    print("read TransE out embedding...")

    entity_embedding = []
    entity_id = []
    with open(out_ent_path) as f:
        for each_line in f:
            each_line = each_line.strip()
            if each_line:
                each_line_list = each_line.split('\t')
                entity_id.append(each_line_list[0])

                entity_embedding.append(ast.literal_eval(each_line_list[1]))

    re_entity = np.array(entity_embedding, dtype=np.float32)

    relation_embedding = []
    relation_id = []
    with open(out_rel_path) as f:
        for each_line in f:
            each_line = each_line.strip()
            if each_line:
                each_line_list = each_line.split('\t')
                relation_id.append(each_line_list[0])

                relation_embedding.append(ast.literal_eval(each_line_list[1]))

    re_relation = np.array(relation_embedding, dtype=np.float32)

    # print(re_entity)
    # print(re_entity.shape)
    #
    # print(re_relation)
    # print(re_relation.shape)

    return re_entity, re_relation

if __name__ == "__main__":

    out_transE_entity_emb = './benchmarks/FB15K/out_transE_entity_embedding100.txt'
    out_transE_relation_emb = './benchmarks/FB15K/out_transE_relation_embedding100.txt'
    read_transe_out_embs(out_transE_entity_emb, out_transE_relation_emb)



    # entity_embs_path = './data/FB15K/init_entity_embedding.txt'
    # rel_embs_path = './data/FB15K/init_relation_embedding.txt'
    # entity_embs, rel_embs = read_init_embs(entity_embs_path,rel_embs_path)

    # new_entity_embs_path = './data/FB15K/new_init_entity_embedding.txt'
    # new_rel_embs_path = './data/FB15K/new_init_relation_embedding.txt'
    #
    # obtain init entity and relation embedding
    # np.savetxt(new_entity_embs_path,entity_embs,fmt='%f',delimiter=',')
    # np.savetxt(new_rel_embs_path,rel_embs,fmt='%f',delimiter=',')
    #



    # init_entity_arr,init_rel_arr = read_new_init_embs(new_entity_embs_path,new_rel_embs_path)
    # print("init_entity_arr.shape",init_entity_arr.shape)
    # print("init_entity_arr\n",init_entity_arr)
    #
    # print("init_rel_arr.shape",init_rel_arr.shape)
    # print("init_entity_arr\n",init_entity_arr)

    # advance_read_init_embs()

    # test()
    # train_triples_id = './data/FB15K/train2id.txt'
    # valid_triples_id = './data/FB15K/valid2id.txt'
    # test_triples_id = './data/FB15K/test2id.txt'
    # read train2id file
    # train_id = read_train_valid_test_id(train_triples_id)
    # print(train_id)
    # print(train_id.shape)

    # print(len(all_entity))
    # get entityPairs --> [0,[1,2,3],5]
    # train_entity_pairs_set = get_entityPair(train_id)
    # train_entity_pairs_path = './data/FB15K/train_entity_pairs_set.txt'
    # write_train_entity_pairs(train_entity_pairs_path,train_entity_pairs_set)

    # print("len(entityPair_set)",len(entityPair_set))
    # print("(entityPair_set)",(entityPair_set[0]))
    # print("len(set(entityPair_set))",len(set(entityPair_set)))

    # obtain entity neighbours --> [entity_id , neighbours, inverse_neighbours]. neighbours is out-coming edge.
    # inverse_neighbours, which means head entity as t,incoming edge
    # train_entity_neighbours_set = get_entityNeighbours(train_id)
    #
    # print("train_entity_neighbours_set over !")
    #
    # train_entity_neighbours_path = './data/FB15K/train_entity_neighbours_set.txt'
    # write_train_entity_neighbours(train_entity_neighbours_path,train_entity_neighbours_set)

    # valid_id = read_train_valid_test_id(valid_triples_id)
    # print(valid_id.shape)
    #
    # test_id = read_train_valid_test_id(test_triples_id)
    # print(test_id.shape)

    # import random
    # s = random.sample([i for i in range(train_id.shape[0])], 10)
    # print(s)
    # print(type(train_id[s,0]))

    # entity2id_path = './data/FB15K/entity2id.txt'
    # entity2obj_path = './data/FB15K/entity2Obj.txt'
    # entity_id_symbol_label_description_path = './data/FB15K/entity_id_symbol_label_description.txt'
    # get entity2id_symbol_label_des ->  0	/m/027rn	Dominican Republic	country in the Caribbean
    # entity_id_symbol_label_description_set = entity_id_symbol_label_description(entity2id_path,entity2obj_path)
    # write_entity_id_symbol_label_description_set(entity_id_symbol_label_description_path,entity_id_symbol_label_description_set)



    print("main over !")
