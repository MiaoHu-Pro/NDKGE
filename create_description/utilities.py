import os
import numpy as np
import pandas as pd
import ast
import sys

# 被引用模块所在的路径
sys.path.append("../")

# from text_analytics.text_analytics.text_analytics import text_analytics
# ta = text_analytics()

import os
import re
import cytoolz as ct
from gensim.parsing import preprocessing


def clean(line):
    function_words_1 = ["the", "of", "and", "to", "a", "in", "i", "he", "that", "was", "it", "his", "you", "with", "as",
                        "for", "had", "is", "her", "not", "but", "at", "on", "she", "be", "have", "by", "which", "him",
                        "they", "this", "from", "all", "were", "my", "we", "one", "so", "said", "me", "there", "or",
                        "an", "are", "no", "would", "their", "if", "been", "when", "do", "who", "what", "them", "will",
                        "out", "up", "then", "more", "could", "into", "man", "now", "some", "your", "very", "did",
                        "has", "about", "time", "can", "little", "than", "only", "upon", "its", "any", "other", "see",
                        "our", "before", "two", "know", "over", "after", "down", "made", "should", "these", "must",
                        "such", "much", "us", "old", "how", "come", "here", "never", "may", "first", "where", "go", "s",
                        "came", "men", "way", "back", "himself", "own", "again", "say", "day", "long", "even", "too",
                        "think", "might", "most", "through", "those", "am", "just", "make", "while", "went", "away",
                        "still", "every", "without", "many", "being", "take", "last", "shall", "yet", "though",
                        "nothing", "get", "once", "under", "same", "off", "another", "let", "tell", "why", "left",
                        "ever", "saw", "look", "seemed", "against", "always", "going", "few", "got", "something",
                        "between", "sir", "thing", "also", "because", "yes", "each", "oh", "quite", "both", "almost",
                        "soon", "however", "having", "t", "whom", "does", "among", "perhaps", "until", "began",
                        "rather", "herself", "next", "since", "anything", "myself", "nor", "indeed", "whose", "thus",
                        "along", "others", "till", "near", "certain", "behind", "during", "alone", "already", "above",
                        "often", "really", "within", "used", "use", "itself", "whether", "around", "second", "across",
                        "either", "towards", "became", "therefore", "able", "sometimes", "later", "else", "seems",
                        "ten", "thousand", "don", "certainly", "ought", "beyond", "toward", "nearly", "although",
                        "past", "seem", "mr", "mrs", "dr", "thou", "except", "none", "probably", "neither", "saying",
                        "ago", "ye", "yourself", "getting", "below", "quickly", "beside", "besides", "especially",
                        "thy", "thee", "d", "unless", "three", "four", "five", "six", "seven", "eight", "nine",
                        "hundred", "million", "billion", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
                        "ninth", "tenth", "amp", "m", "re", "u", "via", "ve", "ll", "th", "lol", "pm", "things", "w",
                        "didn", "doing", "doesn", "r", "gt", "n", "st", "lot", "y", "im", "k", "isn", "ur", "hey",
                        "yeah", "using", "vs", "dont", "ok", "v", "goes", "gone", "lmao", "happen", "wasn", "gotta",
                        "nd", "okay", "aren", "wouldn", "couldn", "cannot", "omg", "non", "inside", "iv", "de",
                        "anymore", "happening", "including", "shouldn", "yours", ]
    function_words_2 = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                        'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                        'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                        'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
                        'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                        'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
                        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm',
                        'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn',
                        'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    function_words = function_words_2 + function_words_1
    # Remove links, hashtags, at-mentions, mark-up, and "RT"
    line = re.sub(r"http\S+", "", line)
    line = re.sub(r"@\S+", "", line)
    line = re.sub(r"#\S+", "", line)
    line = re.sub("<[^>]*>", "", line)
    line = line.replace(" RT", "").replace("RT ", "")

    # Remove punctuation and extra spaces
    line = ct.pipe(line,
                   preprocessing.strip_tags,
                   preprocessing.strip_punctuation,
                   preprocessing.strip_numeric,
                   preprocessing.strip_non_alphanum,
                   preprocessing.strip_multiple_whitespaces
                   )

    # Strip and lowercase
    line = line.lower().strip().lstrip().split()

    # line = [x for x in line if x not in function_words]

    return line


class Rela(object):
    pass


class Enti(object):
    def __init__(self, _id, _symbol, _label, _mention, _neighbours, _entity2vec, _entity_mention_des_word_list,
                 _entity_description_des_word_list=None):
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

    def set_entity_des(self, _entity_des):
        self.entity_des = _entity_des

    def get_entity_description(self):
        return self.entity_des


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
        relation_mention = clean(relation_mention_list[0])
        # print("relation_mention",relation_mention)

        two_entity = relation_mention_list[1:]

        beg = two_entity.index('between')
        end = two_entity.index('and')

        head_enti = two_entity[beg + 1:end]
        tail_enti = two_entity[end + 1:]

        relation_mention += ['which', 'is', 'between']

        head_enti_list = []
        if len(head_enti) == 1 and '/m/' in head_enti[0]:
            head_enti_list.append(head_enti[0])
        else:

            for z in range(len(head_enti)):
                head_enti_list += clean(head_enti[z])
        # tail_en = clean(())
        relation_mention += head_enti_list

        relation_mention += ['and']

        # print("relation_mention",relation_mention)

        tail_enti_list = []
        if len(tail_enti) == 1 and '/m/' in tail_enti[0]:
            tail_enti_list.append(tail_enti[0])
        else:

            for z in range(len(tail_enti)):
                tail_enti_list += clean(tail_enti[z])
        # tail_en = clean(())
        relation_mention += tail_enti_list

        relation_description_list = []
        relation_description_list += relation_mention
        relation_description_list.append(".")

        neighbours_li = []
        #
        for i in range(len(relation_neighbours)):

            re_list = relation_neighbours[i].split(" ")  # split each neighbours

            if 'has' in re_list and 'with' in re_list:

                beg = re_list.index('has')
                end = re_list.index('with')

                # print(beg,end)
                sub_re_list = re_list[beg:end + 1]
                # print(sub_re_list)

                w_list = []
                # add head entity
                n_head = re_list[:beg]

                n_head_list = []
                if len(n_head) == 1 and '/m/' in n_head[0]:
                    n_head_list.append(n_head[0])
                else:

                    for z in range(len(n_head)):
                        n_head_list += clean(n_head[z])

                head_en = n_head_list
                w_list += head_en  #

                # dealing with relation
                for j in range(len(sub_re_list)):
                    w_list += clean(sub_re_list[j])

                # deal with tail entity
                n_tail = re_list[end + 1:]
                n_tail_list = []
                if len(n_tail) == 1 and '/m/' in n_tail[0]:
                    n_tail_list.append(n_tail[0])
                else:

                    for z in range(len(n_tail)):
                        n_tail_list += clean(n_tail[z])
                # tail_en = clean(())
                tail_en = n_tail_list

                w_list += tail_en  #

                relation_description_list += w_list
                relation_description_list.append(".")

                neighbours_li.append(w_list)
            else:

                sub_re_list = re_list
                relation_description_list += sub_re_list
                relation_description_list.append(".")
        # print('relation_description_list',relation_description_list)
        relation_des_word_list.append(" ".join(relation_description_list))

    return relation_des_word_list


def adv_entity_text_process(ent_str):
    """
    given a entity , which was transformed into a word vector
    """

    str = ent_str.split("$")
    entity_name = clean(str[0])
    entity_des = clean(str[1])

    li = ast.literal_eval(str[2])

    entity_description_list = [entity_name]

    entity_description_list += entity_des

    neighbours_li = []

    for i in range(len(li)):

        re_list = li[i].split(" ")
        sub_re_list = re_list[1:-1]
        w_list = [re_list[0]]
        for j in range(len(sub_re_list)):
            w_list += clean(sub_re_list[j])

        w_list.append(re_list[-1])
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
    entity_symbol = str[0]
    entity_des = clean(str[1])

    li = ast.literal_eval(str[2])
    entity_description_list = []

    entity_description_list.append(entity_symbol)

    entity_description_list += entity_des

    entity_description_list.append(".")

    neighbours_li = []

    for i in range(len(li)):

        re_list = li[i].split(" ")
        # print(re_list)
        sub_re_list = re_list[1:-1]
        # print(sub_re_list)
        w_list = [re_list[0]]
        for j in range(len(sub_re_list)):
            w_list += clean(sub_re_list[j])

        w_list.append(re_list[-1])
        w_list.append(".")
        entity_description_list += w_list
        neighbours_li.append(w_list)

    return entity_description_list

def constuct_entity_des(entity_name, entity_mention, current_entity_des_using_name):
    """
    given a entity , which was transformed into a word vector
    """

    _entity_name = []
    if '/m/' in entity_name:
        _entity_name.append(entity_name)

    else:
        _entity_name = clean(entity_name)

    _entity_mention = []
    if '/m/' in entity_mention:
        _entity_mention.append(entity_mention)
    elif entity_mention == 'None':
        _entity_mention = _entity_name

    else:
        _entity_mention = clean(entity_mention)

    entity_name_list = []
    entity_description_list = []
    entity_mention_des_list = []

    entity_name_list += _entity_name

    # entity_description_list += _entity_name
    # entity_description_list += [',', 'that', 'is']
    # entity_description_list += _entity_mention
    # entity_description_list += ['.']
    # entity_mention_des_list += entity_description_list

    # mention is only mention
    entity_mention_des_list += _entity_mention
    if len(current_entity_des_using_name) == 0:
        # if there are no any neighboures, the des is name + mention
        entity_description_list += _entity_name
        entity_description_list += [',', 'that', 'is']
        entity_description_list += _entity_mention
        return entity_description_list, entity_mention_des_list, entity_name_list
    else:

        for i in range(len(current_entity_des_using_name)):

            re_list = current_entity_des_using_name[i].split(" ")

            if 'has' in re_list and 'with' in re_list:

                beg = re_list.index('has')
                end = re_list.index('with')

                sub_re_list = re_list[beg: end + 1]
                w_list = []

                w_list += _entity_name

                for j in range(len(sub_re_list)):
                    w_list += clean(sub_re_list[j])


                n_tail = re_list[end + 1:]
                n_tail_list = []
                if len(n_tail) == 1 and '/m/' in n_tail[0]:
                    n_tail_list.append(n_tail[0])
                else:
                    for z in range(len(n_tail)):
                        n_tail_list += clean(n_tail[z])
                tail_en = n_tail_list
                w_list += tail_en

                entity_description_list += w_list
            else:
                sub_re_list = re_list

                entity_description_list += sub_re_list
            entity_description_list += ['.']

        return entity_description_list, entity_mention_des_list,entity_name_list


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
    word_to_idx = word_dict  # all words in dataset
    print(len(word_to_idx))
    pretrained_embeddings = np.random.uniform(-0.5, 0.5, (len(word_dict), dim))
    # word2vec = load_w2v_vec('./data/GoogleNews-vectors-negative300.bin', word_to_idx)
    word2vec = load_golve_vec(word_to_idx, dim)
    print(len(word2vec))
    print(word_to_idx['the'])
    for word, vector in word2vec.items():  #
        print(word)
        print(word_to_idx[word])
        pretrained_embeddings[word_to_idx[word]] = vector

    # print test
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
    return all_data,all_word
