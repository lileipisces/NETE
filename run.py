from module import NETE_r, NETE_t
from load_data import load_data
from utils import *
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument('-gd', '--gpu_device', type=str, help='device(s) on GPU, default=0', default='0')
parser.add_argument('-dp', '--data_path', type=str, help='path for loading pickle data', default=None)
parser.add_argument('-dr', '--data_ratio', type=str, help='ratio of train:validation:test', default='8:1:1')
parser.add_argument('-id', '--index_dir', type=str, help='create new indexes if the directory is empty, otherwise load indexes', default=None)

parser.add_argument('-rn', '--rating_layer_num', type=int, help='rating prediction layer number, default=4', default=4)
parser.add_argument('-ld', '--latent_dim', type=int, help='latent dimension of users and items, default=200', default=200)
parser.add_argument('-wd', '--word_dim', type=int, help='dimension of word embeddings, default=200', default=200)
parser.add_argument('-rd', '--rnn_dim', type=int, help='dimension of RNN hidden states, default=256', default=256)
parser.add_argument('-sm', '--seq_max_len', type=int, help='seq max len of a text, default=15', default=15)
parser.add_argument('-wn', '--max_word_num', type=int, help='number of words in vocabulary, default=20000', default=20000)
parser.add_argument('-dk', '--dropout_keep', type=float, help='dropout ratio in RNN, default=0.8', default=0.8)

parser.add_argument('-en', '--max_epoch_num', type=int, help='max epoch number, default=100', default=100)
parser.add_argument('-bs', '--batch_size', type=int, help='batch size, default=128', default=128)
parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate, default=0.0001', default=0.0001)
parser.add_argument('-rr', '--reg_rate', type=float, help='regularization rate, default=0.0001', default=0.0001)

parser.add_argument('-pf', '--use_predicted_feature', type=int, help='use predicted features from PMI when testing, 0 means no, otherwise yes', default=0)
parser.add_argument('-pp', '--prediction_path', type=str, help='the path for saving predictions', default=None)
parser.add_argument('-tk', '--top_k', type=int, help='select top k to evaluate, default=5', default=5)
args = parser.parse_args()


print('-----------------------------ARGUMENTS-----------------------------')
for arg in vars(args):
    value = getattr(args, arg)
    if value is None:
        value = ''
    print('{:30} {}'.format(arg, value))
print('-----------------------------ARGUMENTS-----------------------------')


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
if args.data_path is None:
    sys.exit(get_now_time() + 'provide data_path for loading data')
if args.index_dir is None:
    sys.exit(get_now_time() + 'provide index_dir for saving and loading indexes')
if args.prediction_path is None:
    sys.exit(get_now_time() + 'provide prediction_path for saving predicted text')
if not os.path.exists(args.index_dir) or len(os.listdir(args.index_dir)) == 0:
    split_data(args.data_path, args.index_dir, args.data_ratio)


train_tuple_list, validation_tuple_list, test_tuple_list, max_rating, min_rating, user2index, item2index, word2index, \
user_list, item_list, word_list, feature_id_set, user2items_test = load_data(args.data_path, args.index_dir, args.max_word_num,
                                                                             args.seq_max_len, args.use_predicted_feature)
mean_r = (max_rating + min_rating) / 2
sentiment_num = 2


model_r = NETE_r(train_tuple_list, len(user_list), len(item_list), args.rating_layer_num, args.latent_dim, args.learning_rate,
                 args.batch_size, args.reg_rate)
# first train rating prediction module
previous_loss = 1e10
rating_validation, rating_test = None, None
for en in range(1, args.max_epoch_num + 1):
    print(get_now_time() + 'iteration {}'.format(en))

    train_loss = model_r.train_one_epoch()
    print(get_now_time() + 'loss on train set: {}'.format(train_loss))
    validation_loss = model_r.validate(validation_tuple_list)
    print(get_now_time() + 'loss on validation set: {}'.format(validation_loss))

    # early stop setting
    if validation_loss > previous_loss:
        print(get_now_time() + 'early stopped')
        break
    previous_loss = validation_loss

    rating_validation = model_r.get_prediction(validation_tuple_list)
    rating_test = model_r.get_prediction(test_tuple_list)

# evaluating
predicted_rating = []
for (x, r_p) in zip(test_tuple_list, rating_test):
    predicted_rating.append((x[2], r_p))
test_rmse = root_mean_square_error(predicted_rating, max_rating, min_rating)
print(get_now_time() + 'RMSE on test set: {}'.format(test_rmse))
test_mae = mean_absolute_error(predicted_rating, max_rating, min_rating)
print(get_now_time() + 'MAE on test set: {}'.format(test_mae))

user2items_top = model_r.get_prediction_ranking(args.top_k, list(user2items_test.keys()), len(item_list))
ndcg = evaluate_ndcg(user2items_test, user2items_top)
print(get_now_time() + 'NDCG on test set: {}'.format(ndcg))
precision, recall, f1 = evaluate_precision_recall_f1(user2items_test, user2items_top)
print(get_now_time() + 'Precision on test set: {}'.format(precision))
print(get_now_time() + 'HR on test set: {}'.format(recall))
print(get_now_time() + 'F1 on test set: {}'.format(f1))


# replace the ground-truth sentiments with predicted ratings
new_validation_list = []
new_test_list = []
for (x, r_p) in zip(validation_tuple_list, rating_validation):
    x[-1] = r_p
    new_validation_list.append(x)
for (x, r_p) in zip(test_tuple_list, rating_test):
    x[-1] = r_p
    new_test_list.append(x)
validation_tuple_list = new_validation_list
test_tuple_list = new_test_list

# then start to train the explanation generation module
model = NETE_t(train_tuple_list, len(user_list), len(item_list), word2index, mean_r, sentiment_num, args.latent_dim,
               args.word_dim, args.rnn_dim, args.learning_rate, args.batch_size, args.seq_max_len)
# early stop setting
previous_loss = 1e10
seq_prediction = None
for en in range(1, args.max_epoch_num + 1):
    print(get_now_time() + 'iteration {}'.format(en))

    train_loss = model.train_one_epoch(args.dropout_keep)
    print(get_now_time() + 'loss on train set: {}'.format(train_loss))
    validation_loss = model.validate(validation_tuple_list)
    print(get_now_time() + 'loss on validation set: {}'.format(validation_loss))

    # early stop setting
    if validation_loss > previous_loss:
        print(get_now_time() + 'early stopped')
        break
    previous_loss = validation_loss

    seq_prediction = model.get_prediction(test_tuple_list)


ids_predict = []
for s_p in seq_prediction:
    ids = chop_before_eos(word2index, s_p)
    ids_predict.append(ids)

PUS, NUS = unique_sentence_percent(ids_predict)
print(get_now_time() + 'USN on test set: {}'.format(NUS))
print(get_now_time() + 'USR on test set: {}'.format(PUS))

feature_batch = feature_detect(ids_predict, feature_id_set)
# DIV really takes time
DIV = feature_diversity(feature_batch)
print(get_now_time() + 'DIV on test set: {}'.format(DIV))
FCR = feature_coverage_ratio(feature_batch, feature_id_set)
print(get_now_time() + 'FCR on test set: {}'.format(FCR))

feature_test = []
ids_test = []
for x in test_tuple_list:
    # [u, i, r, fea_id, w_list, fea, tem, p_r]
    feature_test.append(x[3])
    ids_test.append(x[4])
FMR = feature_matching_ratio(feature_batch, feature_test)
print(get_now_time() + 'FMR on test set: {}'.format(FMR))

token_predict = [ids2tokens(word_list, ids) for ids in ids_predict]
token_test = [ids2tokens(word_list, ids) for ids in ids_test]
BLEU_1 = bleu_score(token_test, token_predict, n_gram=1, smooth=False)
print(get_now_time() + 'BLEU-1 on test set: {}'.format(BLEU_1))
BLEU_4 = bleu_score(token_test, token_predict, n_gram=4, smooth=False)
print(get_now_time() + 'BLEU-4 on test set: {}'.format(BLEU_4))

text_predict = [' '.join(tokens) for tokens in token_predict]
text_test = [' '.join(tokens) for tokens in token_test]
ROUGE = rouge_score(text_test, text_predict)  # a dictionary
print(get_now_time() + 'ROUGE on test set:')
for (k, v) in ROUGE.items():
    print('{}: {}'.format(k, v))

formatted_out = []
for (x, s_p) in zip(test_tuple_list, seq_prediction):
    text = ids2sentence(word2index, word_list, s_p)
    formatted_out.append('{}\n{}, {}, {}\n{}\n\n'.format(x[6], x[5], x[2], x[7], text))
with open(args.prediction_path + '.test.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(formatted_out))
print(get_now_time() + 'saved predicted text on test set')
