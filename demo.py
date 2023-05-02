import numpy as np
import torch

from utilities import read_new_init_embs,read_transe_out_embs

import logging

logger = logging.getLogger(__name__)

def read_data():

    out_transE_entity_emb = './benchmarks/FB15K/out_transE_entity_embedding100.txt'
    out_transE_relation_emb = './benchmarks/FB15K/out_transE_relation_embedding100.txt'

    pre_train_entity_id, pre_train_rel_id = read_transe_out_embs(out_transE_entity_emb,out_transE_relation_emb)
    print("out_transE_entity_emb ",pre_train_entity_id.shape)
    print("out_transE_relation_emb ",pre_train_rel_id.shape)


    new_entity_embs_path = './benchmarks/FB15K/new_init_entity_embedding_mention_description_id0_des300.txt'
    new_rel_embs_path = './benchmarks/FB15K/new_init_relation_embedding_mention_description_id0_des300.txt'
    print("entity_embs ",new_entity_embs_path)
    print("relation_embs ",new_rel_embs_path)

    entity_embs, rel_embs = read_new_init_embs(new_entity_embs_path,new_rel_embs_path)

    print("entity_embs.shape ", entity_embs.shape)
    print("rel_embs.shape ",rel_embs.shape)

    print("=============")


    _out_ent_embs = torch.from_numpy(pre_train_entity_id)
    _out_rel_embs = torch.from_numpy(pre_train_rel_id)



    init_ent_embs = torch.from_numpy(entity_embs)
    init_rel_embs = torch.from_numpy(rel_embs)

    entity_embedding = torch.cat([_out_ent_embs,init_ent_embs],dim=1)
    relation_embedding = torch.cat([_out_rel_embs,init_rel_embs],dim=1)

    # print(entity_embedding.shape)
    # print(relation_embedding.shape)
    logger.info(entity_embedding.shape)

def read_files(rank_path):
    f = open(rank_path)
    f.readline()

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

    return np.array(x_obj,dtype=int)

def static_frequency(dic_path, data):
    d = data
    c = dict.fromkeys(d, 0)
    for x in d:
        c[x] += 1
    # sorted_x = sorted(c.items(), key=lambda d: d[1], reverse=True)

    sorted_x = sorted(c.items(), key=lambda d: int(d[0]), reverse=False)

    # write_entity_relation_frequency(path=dic_path, data=sorted_x)

    num = len(sorted_x)
    file = open(dic_path, 'w')
    file.writelines('%s\n' % num)
    i = 0
    for e in sorted_x:
        file.write(str(e[0]) + '\t' + str(e[1]) + '\n')
        i += 1

    file.close()

    return sorted_x


if __name__ == "__main__":

    data = read_files("./benchmarks/FB15K237/each_entity_num_neighbours")
    num_nei = data[:,0].tolist()
    print(num_nei)
    print(np.mean(num_nei))
    static_frequency("./benchmarks/FB15K237/each_entity_num_neighbours_frequency.txt",data[:,0].tolist())



