from sklearn.feature_extraction.text import CountVectorizer
from utils import *


def load_data(data_path, index_dir, max_word_num, seq_max_len, use_predicted_feature=False):
    # collect all users id and items id
    user_set = set()
    item_set = set()

    max_rating = 5
    min_rating = 1

    reviews = pickle.load(open(data_path, 'rb'))
    for review in reviews:
        user_set.add(review['user'])
        item_set.add(review['item'])
        rating = review['rating']
        if max_rating < rating:
            max_rating = rating
        if min_rating > rating:
            min_rating = rating

    # convert id to array index
    user_list = list(user_set)
    item_list = list(item_set)
    user2index = {x: i for i, x in enumerate(user_list)}
    item2index = {x: i for i, x in enumerate(item_list)}

    with open(index_dir + 'train.index', 'r') as f:
        line = f.readline()
        indexes = [int(x) for x in line.split(' ')]
    doc_list = []
    for idx in indexes:
        rev = reviews[idx]
        (fea, adj, tem, sco) = rev['template']
        doc_list.append(tem)
    word2index, word_list = get_word2index(doc_list, max_word_num)

    def format_data(data_type):
        with open(index_dir + data_type + '.index', 'r') as f:
            line = f.readline()
            indexes = [int(x) for x in line.split(' ')]
        tuple_list = []
        fea_set = set()
        for idx in indexes:
            rev = reviews[idx]
            u = user2index[rev['user']]
            i = item2index[rev['item']]
            r = rev['rating']
            (fea, adj, tem, sco) = rev['template']
            w_list = [word2index.get(w, word2index['<UNK>']) for w in tem.split(' ')]
            w_list.append(word2index['<EOS>'])
            if len(w_list) > seq_max_len:
                w_list = w_list[:seq_max_len]
            if use_predicted_feature != 0 and data_type == 'test':
                fea = rev['predicted']
            fea_id = word2index.get(fea, word2index['<UNK>'])
            fea_set.add(fea_id)

            if sco == 1:
                sco = 5
            tuple_list.append([u, i, r, fea_id, w_list, fea, tem, sco])
        return tuple_list, fea_set

    train_tuple_list, fea_set_tr = format_data('train')
    validation_tuple_list, fea_set_va = format_data('validation')
    test_tuple_list, fea_set_te = format_data('test')
    user2items_test = {}
    for x in test_tuple_list:
        u = x[0]
        i = x[1]
        if u in user2items_test:
            user2items_test[u].add(i)
        else:
            user2items_test[u] = {i}

    feature_set = set()
    feature_set = feature_set | fea_set_tr
    feature_set = feature_set | fea_set_va
    feature_set = feature_set | fea_set_te

    return train_tuple_list, validation_tuple_list, test_tuple_list, max_rating, min_rating, user2index, item2index, word2index, user_list, item_list, word_list, feature_set, user2items_test


def get_word2index(doc_list, max_word_num):
    def split_words_by_space(text):
        return text.split(' ')

    vectorizer = CountVectorizer(max_features=max_word_num, analyzer=split_words_by_space)
    vectorizer.fit(doc_list)
    word_list = vectorizer.get_feature_names()
    word_list.extend(['<UNK>', '<GO>', '<EOS>', '<PAD>'])
    word2index = {w: i for i, w in enumerate(word_list)}

    return word2index, word_list
