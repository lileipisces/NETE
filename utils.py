from bleu import compute_bleu
from rouge import rouge
import numpy as np
import datetime
import random
import pickle
import math
import os


def get_now_time():
    """a string of current time"""
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def mean_absolute_error(predicted, max_r, min_r):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        total += abs(sub)

    return total / len(predicted)


def mean_square_error(predicted, max_r, min_r):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_square_error(predicted, max_r, min_r)
    return math.sqrt(mse)


def split_data(data_path, save_dir, ratio_str):
    '''
    :param data_path: pickle file, a list of all instances
    :param save_dir: save the indexes
    :param ratio_str: in the format of train:validation:test
    '''

    # process rating and review
    user2item = {}
    item2user = {}
    user2item2idx = {}
    reviews = pickle.load(open(data_path, 'rb'))
    for idx, review in enumerate(reviews):
        u = review['user']
        i = review['item']

        if u in user2item:
            user2item[u].append(i)
        else:
            user2item[u] = [i]
        if i in item2user:
            item2user[i].append(u)
        else:
            item2user[i] = [u]

        if u in user2item2idx:
            user2item2idx[u][i] = idx
        else:
            user2item2idx[u] = {i: idx}

    # split data
    train_set = set()
    for (u, item_list) in user2item.items():
        i = random.choice(item_list)
        train_set.add(user2item2idx[u][i])
    for (i, user_list) in item2user.items():
        u = random.choice(user_list)
        train_set.add(user2item2idx[u][i])

    total_num = len(reviews)
    ratio = [float(r) for r in ratio_str.split(':')]
    train_num = int(ratio[0] / sum(ratio) * total_num)
    validation_num = int(ratio[1] / sum(ratio) * total_num)

    index_list = list(range(total_num))
    while len(train_set) < train_num:
        train_set.add(random.choice(index_list))
    remains_list = list(set(index_list) - train_set)

    validation_set = set()
    while len(validation_set) < validation_num:
        validation_set.add(random.choice(remains_list))
    test_set = set(remains_list) - validation_set

    def write_to_file(path, data_set):
        idx_list = [str(x) for x in data_set]
        with open(path, 'w', encoding='utf-8') as f:
            f.write(' '.join(idx_list))

    # save data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(get_now_time() + 'writing index data to {}'.format(save_dir))
    write_to_file(save_dir + 'train.index', train_set)
    write_to_file(save_dir + 'validation.index', validation_set)
    write_to_file(save_dir + 'test.index', test_set)


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def chop_before_eos(word2index, ids):
    end = len(ids)
    for idx, i in enumerate(ids):
        if i == word2index['<EOS>']:
            end = idx
            break
    return ids[:end]


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(feature_list)

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for (fea_list, fea) in zip(feature_batch, test_feature):
        for f in fea_list:
            if f == fea:
                count += 1
                break

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    feature_list = []
    for fb in feature_batch:
        feature_list.extend(fb)

    return len(set(feature_list)) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            for k in y:
                if k in x:
                    total_count += 1

    denominator = list_len * (list_len - 1) / 2

    return total_count / denominator


def ids2tokens(word_list, ids):
    result = []
    for i in ids:
        result.append(word_list[i])

    return result


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def ids2sentence(word2index, word_list, ids):
    result = []
    for i in ids:
        if i != word2index['<EOS>']:
            result.append(word_list[i])
        else:
            break
    return ' '.join(result)


def pad_sequence_4_generation(sequence_batch, pad_int):
    '''
    Pad sentences with <PAD> so that each sentence of a batch has the same length
    :param sequence_batch: a list of lists
    :return: 2d numpy matrix, 1d numpy vector
    '''
    seq_len = [len(sequence) for sequence in sequence_batch]
    max_seq_len = max(seq_len)
    new_batch = [sequence + [pad_int] * (max_seq_len - len(sequence)) for sequence in sequence_batch]
    new_batch = np.asarray(new_batch, dtype=np.int32)
    new_seq_len = np.asarray(seq_len, dtype=np.int32)

    return new_batch, new_seq_len


def evaluate_ndcg(user2items_test, user2items_top):
    top_k = len(list(user2items_top.values())[0])
    dcgs = [1 / math.log(i + 2) for i in range(top_k)]

    ndcg = 0
    for u, test_items in user2items_test.items():
        rank_list = user2items_top[u]
        dcg_u = 0
        for idx, item in enumerate(rank_list):
            if item in test_items:
                dcg_u += dcgs[idx]
        ndcg += dcg_u

    return ndcg / (sum(dcgs) * len(user2items_test))


def evaluate_precision_recall_f1(user2items_test, user2items_top):
    top_k = len(list(user2items_top.values())[0])

    precision_sum = 0
    recall_sum = 0  # it is also named hit ratio
    f1_sum = 0
    for u, test_items in user2items_test.items():
        rank_list = user2items_top[u]
        hits = len(test_items & set(rank_list))
        pre = hits / top_k
        rec = hits / len(test_items)
        precision_sum += pre
        recall_sum += rec
        if (pre + rec) > 0:
            f1_sum += 2 * pre * rec / (pre + rec)

    precision = precision_sum / len(user2items_test)
    recall = recall_sum / len(user2items_test)
    f1 = f1_sum / len(user2items_test)

    return precision, recall, f1
