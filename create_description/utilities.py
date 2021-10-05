
import os
import numpy as np
import pandas as pd
import ast
import sys
#被引用模块所在的路径
sys.path.append("../")
from text_analytics.text_analytics.text_analytics import text_analytics

ta = text_analytics()
class Rela(object):
    pass

class Enti(object):
    def __init__(self, _id, _symbol, _label, _mention, _neighbours, _entity2vec,_entity_mention_des_word_list,_entity_description_des_word_list = None):
        self.id = str(_id)

        self.symbol = _symbol
        self.label = _label
        self.mention = _mention
        self.neighbours = _neighbours
        self.entity2vec = _entity2vec
        self.entity_mention_des = _entity_mention_des_word_list
        self.entity_description_des = _entity_description_des_word_list

    def print_enti(self):
        print("id: ", self.id, '\n'
                               "symbol: ", self.symbol,
              "label: ", self.label,
              "description: ", self.mention)

    def get_random_neighbour(self):
        """
        randomly return a neighbours
        """
        num_neighbours = len(self.neighbours)

        if num_neighbours == 0:
            res = "the entity has not neighbours"
        else:
            index = np.random.random_integers(0, num_neighbours - 1)  # readomly select a neighbour
            res = str(self.neighbours[index])

        return res

    def get_des(self):

        des = str(self.symbol) + '$' + str(self.label) + '$' + str(self.mention) + '$' + str(self.neighbours)

        return des

    def set_entity_des(self,_entity_des):
        self.entity_des = _entity_des

    def get_entity_description(self):
        return self.entity_des


def write_to_file(out_path,all_data):

    ls = os.linesep

    try:
        fobj = open(out_path,  'w')
    except IOError as err:

        print('file open error: {0}'.format(err))

    else:

        fobj.writelines('%s\n' % x for x in all_data)

        fobj.close()

    print('WRITE FILE DONE!')

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

def relation_text_process(rel_str_list):
    """
    given a relation , which was transformed into a word vector
    """
    # rel_str = "/film/actor/film./film/performance/film, which is between /m/07nznf and /m/014lc_;/m/07nznf has a relationship of /award/award_winner/awards_won./award/award_honor/award with /m/02g3ft;/m/014lc_ has a relationship of /film/film/release_date_s./film/film_regional_release_date/film_release_region with /m/0f8l9c"
    relation_des_word_list = []

    for rel_str in rel_str_list:

        rel_str = rel_str.split(";")
        relation_mention = rel_str[0]
        relation_neighbours = rel_str[1:]

        relation_mention_list = relation_mention.split(" ")

        # print("relation_mention_list",relation_mention_list)
        relation_mention = ta.clean(relation_mention_list[0])
        # print("relation_mention",relation_mention)

        two_entity = relation_mention_list[1:]

        beg = two_entity.index('between')
        end = two_entity.index('and')

        head_enti = two_entity[beg+1:end]
        tail_enti = two_entity[end+1:]


        relation_mention += ['which', 'is', 'between']

        head_enti_list = []
        if len(head_enti) == 1 and '/m/' in head_enti[0]:
            head_enti_list.append(head_enti[0])
        else:

            for z in range(len(head_enti)):

                head_enti_list += ta.clean(head_enti[z])
        # tail_en = ta.clean(())
        relation_mention += head_enti_list

        relation_mention += ['and']

        # print("relation_mention",relation_mention)

        tail_enti_list = []
        if len(tail_enti) == 1 and '/m/' in tail_enti[0]:
            tail_enti_list.append(tail_enti[0])
        else:

            for z in range(len(tail_enti)):

                tail_enti_list += ta.clean(tail_enti[z])
        # tail_en = ta.clean(())
        relation_mention += tail_enti_list

        # print("relation_mention",relation_mention)

        relation_description_list = []
        relation_description_list += relation_mention
        relation_description_list.append(".")

        neighbours_li = []
        #
        for i in range(len(relation_neighbours)):

            re_list = relation_neighbours[i].split(" ") # 分解每个邻居
            # re_list = li[i].split(" ") # 分解每个邻居
            # re_list = li[i]

            if 'has' in re_list and 'with' in re_list:

                beg = re_list.index('has')
                end = re_list.index('with')

                # print(beg,end)
                sub_re_list = re_list[beg:end+1]
                # print(sub_re_list)

                w_list = []
                # 加入头实体
                n_head = re_list[:beg]
                #
                n_head_list = []
                if len(n_head) == 1 and '/m/' in n_head[0]:
                    n_head_list.append(n_head[0])
                else:

                    for z in range(len(n_head)):

                        n_head_list += ta.clean(n_head[z])
                # tail_en = ta.clean(())
                head_en = n_head_list
                w_list += head_en #


                # 处理关系
                for j in range(len(sub_re_list)):

                    w_list += ta.clean(sub_re_list[j])

                # 处理尾实体
                n_tail = re_list[end+1: ]
                n_tail_list = []
                if len(n_tail) == 1 and '/m/' in n_tail[0]:
                    n_tail_list.append(n_tail[0])
                else:

                    for z in range(len(n_tail)):

                        n_tail_list += ta.clean(n_tail[z])
                # tail_en = ta.clean(())
                tail_en = n_tail_list

                w_list += tail_en # 取尾巴实体
                # print(i)
                # w_list = ta.clean(re_list)

                relation_description_list += w_list
                relation_description_list.append(".")


                neighbours_li.append(w_list)

                # print("w_list",w_list)

            else:

                sub_re_list = re_list
                # print(sub_re_list)
                # no_neighbour = ta.clean(sub_re_list)
                relation_description_list += sub_re_list
                relation_description_list.append(".")

            # re_list = relation_neighbours[i].split(" ") # 分解每个邻居
            #
            # sub_re_list = re_list[1:-1]
            #
            # w_list = [re_list[0]] # 取头实体
            #
            # for j in range(len(sub_re_list)):
            #     w_list += ta.clean(sub_re_list[j])
            #
            # w_list.append(re_list[-1]) # 取尾巴实体
            #
            # print(i,w_list)
            #
            # relation_description_list += w_list
            # relation_description_list.append(".")
        # print('relation_description_list',relation_description_list)
        relation_des_word_list.append(" ".join(relation_description_list))

    return relation_des_word_list


def adv_entity_text_process(ent_str):
    """
    given a entity , which was transformed into a word vector
    """

    str = ent_str.split("$")

    # str = ta.clean(str)
    # entity_symbol = str[0]
    entity_name = ta.clean(str[0])
    entity_des = ta.clean(str[1])

    li = ast.literal_eval(str[2])

    entity_description_list = [entity_name]

    entity_description_list += entity_des

    neighbours_li = []

    for i in range(len(li)):

        re_list = li[i].split(" ") # 分解每个邻居
        # print(re_list)
        sub_re_list = re_list[1:-1]
        # print(sub_re_list)
        w_list = [re_list[0]] # 取头实体
        for j in range(len(sub_re_list)):
            w_list += ta.clean(sub_re_list[j])

        w_list.append(re_list[-1]) # 取尾巴实体
        # print(i)
        # print(w_list)
        entity_description_list += w_list
        neighbours_li.append(w_list)

    return entity_description_list


def entity_text_process(ent_str):
    """
    given a entity , which was transformed into a word vector
    """

    # ent_str = "/m/07nznf;Bryan Singer;American film director, writer and producer;" \
    #       "['/m/07nznf has a relationship of /people/person/nationality with /m/09c7w0', " \
    #       "'/m/07nznf has a relationship of /award/award_nominee/award_nominations./award/award_nomination/nominated_for with /m/04p5cr', " \
    #       "'/m/07nznf has a relationship of /medicine/notable_person_with_medical_condition/condition with /m/029sk', " \
    #       "'/m/07nznf has a relationship of /award/award_nominee/award_nominations./award/award_nomination/award_nominee with /m/08xwck', " \
    #       "'/m/07nznf has a relationship of /award/award_nominee/award_nominations./award/award_nomination/nominated_for with /m/016fyc', " \
    #       "'/m/07nznf has a relationship of /film/actor/film./film/performance/film with /m/014lc_', " \
    #       "'/m/07nznf has a relationship of /film/producer/films_executive_produced with /m/01qb5d', " \
    #       "'/m/07nznf has a relationship of /people/person/profession with /m/0dxtg', " \
    #       "'/m/07nznf has a relationship of /film/film_story_contributor/film_story_credits with /m/01qb5d', " \
    #       "'/m/07nznf has a relationship of /base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency with /m/09nqf', " \
    #       "'/m/07nznf has a relationship of /film/director/film with /m/02qhlwd', " \
    #       "'/m/07nznf has a relationship of /people/person/education./education/education/institution with /m/065y4w7', '/m/07nznf has a relationship of /people/person/profession with /m/01d_h8', '/m/07nznf has a relationship of /film/actor/film./film/performance/film with /m/01qb5d', '/m/07nznf has a relationship of /film/producer/film with /m/044g_k', '/m/07nznf has a relationship of /tv/tv_producer/programs_produced./tv/tv_producer_term/producer_type with /m/0ckd1', '/m/07nznf has a relationship of /film/producer/film with /m/016fyc', '/m/07nznf has a relationship of /people/person/profession with /m/03gjzk', '/m/07nznf has a relationship of /people/person/ethnicity with /m/041rx', '/m/07nznf has a relationship of /film/producer/film with /m/02qhlwd', '/m/07nznf has a relationship of /film/film_story_contributor/film_story_credits with /m/044g_k', '/m/07nznf has a relationship of /film/director/film with /m/044g_k', '/m/07nznf has a relationship of /award/award_nominee/award_nominations./award/award_nomination/award with /m/040njc', '/m/07nznf has a relationship of /base/popstra/celebrity/friendship./base/popstra/friendship/participant with /m/015v3r', '/m/07nznf has a relationship of /film/producer/film with /m/0cd2vh9', '/m/07nznf has a relationship of /film/director/film with /m/0d90m', '/m/07nznf has a relationship of /award/award_nominee/award_nominations./award/award_nomination/award with /m/0fbtbt', '/m/07nznf has a relationship of /film/director/film with /m/01qb5d', '/m/07nznf has a relationship of /film/director/film with /m/016fyc', '/m/07nznf has a relationship of /film/film_story_contributor/film_story_credits with /m/0d90m', '/m/07nznf has a relationship of /people/person/education./education/education/institution with /m/01hb1t', '/m/07nznf has a relationship of /people/person/profession with /m/02jknp', '/m/07nznf has a relationship of /people/person/place_of_birth with /m/02_286', '/m/07nznf has a relationship of /film/film_story_contributor/film_story_credits with /m/0cd2vh9', '/m/07nznf has a relationship of /tv/tv_producer/programs_produced./tv/tv_producer_term/program with /m/04p5cr', '/m/07nznf has a relationship of /award/award_winner/awards_won./award/award_honor/award with /m/02g3ft', '/m/07nznf has a relationship of /award/award_winner/awards_won./award/award_honor/honored_for with /m/0d90m', '/m/07nznf has a relationship of /people/person/gender with /m/05zppz', '/m/07nznf has a relationship of /award/award_nominee/award_nominations./award/award_nomination/award_nominee with /m/0h53p1', '/m/07nznf has a relationship of /base/popstra/celebrity/friendship./base/popstra/friendship/participant with /m/01k53x', '/m/07nznf has a relationship of /award/award_nominee/award_nominations./award/award_nomination/award_nominee with /m/013pk3', '/m/07nznf has a relationship of /award/award_winner/awards_won./award/award_honor/honored_for with /m/044g_k', '/m/07nznf has a relationship of /people/person/profession with /m/02hrh1q']"

    str = ent_str.split("$")

    # str = ta.clean(str)
    entity_symbol = str[0]
    # entity_name = ta.clean(str[1])
    entity_des = ta.clean(str[1])
    # print(str[2])

    # print(ta.clean(ast.literal_eval(str[3])[0]))
    # print("=====errs=====")
    # print(str)

    li = ast.literal_eval(str[2])

    # print("neighbours : ",len(li))

    entity_description_list = []

    entity_description_list.append(entity_symbol)


    # entity_description_list += entity_name

    entity_description_list += entity_des

    entity_description_list.append(".")

    neighbours_li = []

    for i in range(len(li)):

        re_list = li[i].split(" ") # 分解每个邻居
        # print(re_list)
        sub_re_list = re_list[1:-1]
        # print(sub_re_list)
        w_list = [re_list[0]] # 取头实体
        for j in range(len(sub_re_list)):
            w_list += ta.clean(sub_re_list[j])

        w_list.append(re_list[-1]) # 取尾巴实体
        w_list.append(".")
        # print(i)
        # print(w_list)
        entity_description_list += w_list
        neighbours_li.append(w_list)

    # print(neighbours_li)
    # print("neighbours : ", len(neighbours_li))

    # print(entity_description_list)
    # print(len(entity_description_list))

    return entity_description_list

    # print(eval(str[3])[0])


    # 做一个词库，包含所有实体和关系


def constuct_entity_des(entity_name,entity_mention,current_entity_des_using_name):
    """
    given a entity , which was transformed into a word vector
    """

    # _str = ent_str.split("$")

    # str = ta.clean(str)
    # entity_symbol = str[0]

    _entity_name = []
    if '/m/' in entity_name:
        _entity_name.append(entity_name)

    else:
        _entity_name = ta.clean(entity_name)

    _entity_mention = []
    if '/m/' in entity_mention:
        _entity_mention.append(entity_mention)
    elif entity_mention == 'None':
        _entity_mention = _entity_name

    else:
        _entity_mention = ta.clean(entity_mention)

    # li = ast.literal_eval(_str[2])

    entity_description_list = []
    entity_mention_des_list = []

    entity_description_list += _entity_name

    entity_description_list += [',', 'that','is']

    entity_description_list += _entity_mention
    entity_description_list += ['.']

    entity_mention_des_list += entity_description_list

    if len(current_entity_des_using_name) == 0:

        return entity_description_list, entity_mention_des_list
    else:

        for i in range(len(current_entity_des_using_name)):

            re_list = current_entity_des_using_name[i].split(" ")

            if 'has' in re_list and 'with' in re_list:

                beg = re_list.index('has')
                end = re_list.index('with')

                sub_re_list = re_list[beg: end+1]
                w_list = []
                # 加入头实体
                w_list += _entity_name
                # 处理关系
                for j in range(len(sub_re_list)):

                    w_list += ta.clean(sub_re_list[j])

                # 处理尾实体
                n_tail = re_list[end+1:]
                n_tail_list = []
                if len(n_tail) == 1 and '/m/' in n_tail[0]:
                    n_tail_list.append(n_tail[0])
                else:
                    for z in range(len(n_tail)):
                        n_tail_list += ta.clean(n_tail[z])
                tail_en = n_tail_list
                w_list += tail_en # 取尾巴实体

                entity_description_list += w_list
            else:
                # no_neighbour = ta.clean(str())
                sub_re_list = re_list

                entity_description_list += sub_re_list
            entity_description_list += ['.']

        return entity_description_list, entity_mention_des_list


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
    print(len(word_to_idx))

    pretrained_embeddings = np.random.uniform(-0.5, 0.5, (len(word_dict), dim))

    # word2vec = load_w2v_vec('./data/GoogleNews-vectors-negative300.bin', word_to_idx)
    word2vec = load_golve_vec(word_to_idx, dim)
    print(len(word2vec))
    print(word_to_idx['the'])
    for word, vector in word2vec.items():  # 初始化每个词
        print(word)

        print(word_to_idx[word])
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

