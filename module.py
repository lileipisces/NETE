from utils import *
import tensorflow as tf
import numpy as np
import random
import math
import heapq


def decode_train(cell_w, cell_f, seq_max_len, initial_state, seq_embeddings, feature_emb, latent_dim, mapping_layer):
    seq_embed = tf.TensorArray(dtype=tf.float32, size=seq_max_len)
    seq_embed = seq_embed.unstack(seq_embeddings)

    def condition(step, new_state, new_output):  # arguments returned from the body function, cannot be removed
        return step < seq_max_len

    def loop_body(step, state, output):
        inputs = seq_embed.read(step)
        output_w, state_w = cell_w(inputs=inputs, state=state)
        output_f, state_f = cell_f(inputs=feature_emb, state=state)
        gamma = fusion_unit(state_w, state_f, latent_dim)
        gamma = tf.clip_by_value(gamma, clip_value_min=0.0, clip_value_max=1.0)
        new_state = (1.0 - gamma) * state_w + gamma * state_f  # (batch_size, hidden_size)
        logits = mapping_layer(new_state)  # (batch_size, vocab_size)
        new_output = output.write(index=step, value=logits)
        return step + 1, new_state, new_output

    outputs = tf.TensorArray(dtype=tf.float32, size=seq_max_len)
    loop_init = [tf.constant(value=0, dtype=tf.int32), initial_state, outputs]

    _, _, last_out = tf.while_loop(cond=condition, body=loop_body, loop_vars=loop_init)  # (seq_max_len, batch_size, vocab_size)
    final_out = tf.transpose(last_out.stack(), perm=[1, 0, 2])  # (batch_size, seq_max_len, vocab_size)

    return final_out


def decode_infer(cell_w, cell_f, seq_max_len, initial_state, start_token, feature_emb, latent_dim, mapping_layer, word_embeddings):
    def condition(step, new_token, new_state, new_word_out):  # arguments returned from the body function, cannot be removed
        return step < seq_max_len

    def loop_body(step, token, state, word_out):
        inputs = tf.nn.embedding_lookup(word_embeddings, token)  # (batch_size, word_dim)
        output_w, state_w = cell_w(inputs=inputs, state=state)
        output_f, state_f = cell_f(inputs=feature_emb, state=state)
        gamma = fusion_unit(state_w, state_f, latent_dim)
        gamma = tf.clip_by_value(gamma, clip_value_min=0.0, clip_value_max=1.0)
        new_state = (1.0 - gamma) * state_w + gamma * state_f  # (batch_size, hidden_size)
        logits = mapping_layer(new_state)  # (batch_size, vocab_size)
        new_token = tf.argmax(logits, axis=1, output_type=tf.int32)  # (batch_size,)
        new_word_out = word_out.write(index=step, value=new_token)
        return step + 1, new_token, new_state, new_word_out

    word_ids = tf.TensorArray(dtype=tf.int32, size=seq_max_len)
    loop_init = [tf.constant(value=0, dtype=tf.int32), start_token, initial_state, word_ids]

    _, _, _, word_ids_out = tf.while_loop(cond=condition, body=loop_body, loop_vars=loop_init)
    word_ids_out = tf.transpose(word_ids_out.stack(), perm=[1, 0])  # (batch_size, seq_max_len)

    return word_ids_out


def fusion_unit(state_w, state_f, latent_dim):
    with tf.variable_scope('fusion_unit', reuse=tf.AUTO_REUSE):
        state_w_ = tf.layers.dense(inputs=state_w, units=latent_dim, activation=tf.nn.tanh, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                   use_bias=False, name='state_w_')  # (batch_size, latent_dim)
        state_f_ = tf.layers.dense(inputs=state_f, units=latent_dim, activation=tf.nn.tanh, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                   use_bias=False, name='state_f_')  # (batch_size, latent_dim)
        state_w_f = tf.concat(values=[state_w_, state_f_], axis=1)  # (batch_size, hidden_size * 2)
        gamma = tf.layers.dense(inputs=state_w_f, units=1, activation=tf.nn.sigmoid, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                use_bias=False, name='gamma')  # (batch_size, 1)
    return gamma


class NETE_r:
    def __init__(self, train_tuple_list, user_num, item_num, rating_layer_num=4, latent_dim=200, learning_rate=0.0001,
                 batch_size=128, reg_rate=0.0001):

        self.train_tuple_list = train_tuple_list
        self.batch_size = batch_size

        graph = tf.Graph()
        with graph.as_default():
            # input
            self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')  # (batch_size,)
            self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
            self.rating = tf.placeholder(dtype=tf.float32, shape=[None], name='rating')

            # embeddings
            user_embeddings = tf.get_variable('user_embeddings', shape=[user_num, latent_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))
            item_embeddings = tf.get_variable('item_embeddings', shape=[item_num, latent_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))

            # rating prediction
            user_feature = tf.nn.embedding_lookup(user_embeddings, self.user_id)  # (batch_size, latent_dim)
            item_feature = tf.nn.embedding_lookup(item_embeddings, self.item_id)
            hidden = tf.concat(values=[user_feature, item_feature], axis=1)  # (batch_size, latent_dim * 2)
            for k in range(rating_layer_num):
                hidden = tf.layers.dense(inputs=hidden, units=latent_dim * 2, activation=tf.nn.sigmoid, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         bias_initializer=tf.constant_initializer(0.0), name='layer-{}'.format(k))  # (batch_size, latent_dim * 2)
            prediction = tf.layers.dense(inputs=hidden, units=1, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         bias_initializer=tf.constant_initializer(0.0), name='prediction')  # (batch_size, 1)
            self.predicted_rating = tf.reshape(prediction, shape=[-1])  # (batch_size,)
            rating_loss = tf.losses.mean_squared_error(self.rating, self.predicted_rating)

            regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

            # optimization
            self.total_loss = rating_loss + reg_rate * regularization_cost
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)

            init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)
        self.sess.run(init)

    def train_one_epoch(self):
        sample_num = len(self.train_tuple_list)
        index_list = list(range(sample_num))
        random.shuffle(index_list)

        total_loss = 0

        step_num = int(math.ceil(sample_num / self.batch_size))
        for step in range(step_num):
            start = step * self.batch_size
            offset = min(start + self.batch_size, sample_num)

            user = []
            item = []
            rating = []
            for idx in index_list[start:offset]:
                x = self.train_tuple_list[idx]
                user.append(x[0])
                item.append(x[1])
                rating.append(x[2])
            user = np.asarray(user, dtype=np.int32)
            item = np.asarray(item, dtype=np.int32)
            rating = np.asarray(rating, dtype=np.float32)

            feed_dict = {self.user_id: user,
                         self.item_id: item,
                         self.rating: rating}
            _, loss = self.sess.run([self.optimizer, self.total_loss], feed_dict=feed_dict)
            total_loss += loss * (offset - start)

        return total_loss / sample_num

    def validate(self, tuple_list):
        sample_num = len(tuple_list)

        total_loss = 0

        step_num = int(math.ceil(sample_num / self.batch_size))
        for step in range(step_num):
            start = step * self.batch_size
            offset = min(start + self.batch_size, sample_num)

            user = []
            item = []
            rating = []
            for x in tuple_list[start:offset]:
                user.append(x[0])
                item.append(x[1])
                rating.append(x[2])
            user = np.asarray(user, dtype=np.int32)
            item = np.asarray(item, dtype=np.int32)
            rating = np.asarray(rating, dtype=np.float32)

            feed_dict = {self.user_id: user,
                         self.item_id: item,
                         self.rating: rating}
            loss = self.sess.run(self.total_loss, feed_dict=feed_dict)
            total_loss += loss * (offset - start)

        return total_loss / sample_num

    def get_prediction(self, tuple_list):
        sample_num = len(tuple_list)
        rating_prediction = []

        step_num = int(math.ceil(sample_num / self.batch_size))
        for step in range(step_num):
            start = step * self.batch_size
            offset = min(start + self.batch_size, sample_num)

            user = []
            item = []
            for x in tuple_list[start:offset]:
                user.append(x[0])
                item.append(x[1])
            user = np.asarray(user, dtype=np.int32)
            item = np.asarray(item, dtype=np.int32)

            feed_dict = {self.user_id: user,
                         self.item_id: item}
            rating_p = self.sess.run(self.predicted_rating, feed_dict=feed_dict)
            rating_prediction.extend(rating_p)

        return np.asarray(rating_prediction, dtype=np.float32)

    def get_prediction_ranking(self, top_k, users_test, item_num):
        user2items_train = {}
        for x in self.train_tuple_list:
            u = x[0]
            i = x[1]
            if u in user2items_train:
                user2items_train[u].add(i)
            else:
                user2items_train[u] = {i}

        user2items_top = {}
        for u in users_test:
            items = set(list(range(item_num))) - user2items_train[u]
            tuple_list = [[u, i] for i in items]
            predicted = self.get_prediction(tuple_list)
            item2rating = {}
            for i, p in zip(items, predicted):
                rating = p
                if rating == 0:
                    rating = random.random()
                item2rating[i] = rating
            top_list = heapq.nlargest(top_k, item2rating, key=item2rating.get)
            user2items_top[u] = top_list

        return user2items_top


class NETE_t:
    def __init__(self, train_tuple_list, user_num, item_num, word2index, mean_r=3, sentiment_num=2, latent_dim=200,
                 word_dim=200, rnn_size=256, learning_rate=0.0001, batch_size=128, seq_max_len=15):

        self.train_tuple_list = train_tuple_list
        self.word2index = word2index
        self.batch_size = batch_size
        self.seq_max_len = seq_max_len

        graph = tf.Graph()
        with graph.as_default():
            # input
            self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')  # (batch_size,)
            self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
            self.rating = tf.placeholder(dtype=tf.float32, shape=[None], name='rating')
            self.feature = tf.placeholder(dtype=tf.int32, shape=[None], name='feature')
            self.word_id_seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_id_seq')  # (batch_size, batch_max_len)
            self.seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name='seq_len')  # (batch_size,)
            self.batch_max_len = tf.placeholder(dtype=tf.int32, shape=[], name='batch_max_len')
            self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

            # embeddings
            user_embeddings = tf.get_variable('user_embeddings', shape=[user_num, latent_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))
            item_embeddings = tf.get_variable('item_embeddings', shape=[item_num, latent_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))
            sentiment_embeddings = tf.get_variable('sentiment_embeddings', shape=[sentiment_num, latent_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))
            word_embeddings = tf.get_variable('word_embeddings', shape=[len(self.word2index), word_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))

            # text generation
            b_size = tf.shape(input=self.user_id)[0]
            start_token = tf.fill(dims=[b_size], value=self.word2index['<GO>'])  # (batch_size,)
            ending = tf.strided_slice(input_=self.word_id_seq, begin=[0, 0], end=[b_size, -1], strides=[1, 1])  # remove the last column
            train_input = tf.concat(values=[tf.reshape(tensor=start_token, shape=[-1, 1]), ending], axis=1)  # add <GO> to the head of each sample
            train_input_emb = tf.nn.embedding_lookup(params=word_embeddings, ids=train_input)  # (batch_size, batch_max_len, word_dim)
            feature_emb = tf.nn.embedding_lookup(params=word_embeddings, ids=self.feature)  # (batch_size, word_dim)

            # encoder
            user_feature = tf.nn.embedding_lookup(user_embeddings, self.user_id)  # (batch_size, latent_dim)
            item_feature = tf.nn.embedding_lookup(item_embeddings, self.item_id)
            # sentiment feature
            one = tf.ones_like(self.rating, dtype=tf.int32)
            zero = tf.zeros_like(self.rating, dtype=tf.int32)
            sentiment_index = tf.where(self.rating < mean_r, x=zero, y=one)
            sentiment_feature = tf.nn.embedding_lookup(sentiment_embeddings, sentiment_index)
            encoder_input = tf.concat(values=[user_feature, item_feature, sentiment_feature], axis=1)  # (batch_size, word_dim * 3)
            initial_state = tf.layers.dense(inputs=encoder_input, units=rnn_size, activation=tf.nn.tanh, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), bias_initializer=tf.constant_initializer(0.0))  # (batch_size, rnn_size)

            # decoder
            word_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_size, kernel_initializer=tf.orthogonal_initializer(), reuse=tf.AUTO_REUSE)  # rnn_size: the dimension of h(t)
            word_decoder = tf.nn.rnn_cell.DropoutWrapper(cell=word_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)
            feature_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_size, kernel_initializer=tf.orthogonal_initializer(), reuse=tf.AUTO_REUSE)  # rnn_size: the dimension of h(t)
            feature_decoder = tf.nn.rnn_cell.DropoutWrapper(cell=feature_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)
            output_layer = tf.layers.Dense(units=len(self.word2index), kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), bias_initializer=tf.constant_initializer(0.0), name='output_layer')

            # decoding
            seq_emb = tf.transpose(train_input_emb, perm=[1, 0, 2])  # (batch_max_len, batch_size, word_dim)
            train_logits = decode_train(word_decoder, feature_decoder, self.batch_max_len, initial_state, seq_emb, feature_emb, latent_dim, output_layer)  # (batch_size, batch_max_len, vocab_size)
            self.argmax_tokens = decode_infer(word_decoder, feature_decoder, seq_max_len, initial_state, start_token, feature_emb, latent_dim, output_layer, word_embeddings)   # (batch_size, seq_max_len)

            # text generation loss
            masks = tf.sequence_mask(lengths=self.seq_len, maxlen=self.batch_max_len, dtype=tf.float32)  # only compute the loss of valid words, (batch_size, batch_max_len)
            text_loss = tf.contrib.seq2seq.sequence_loss(logits=train_logits, targets=self.word_id_seq, weights=masks)

            # optimization
            self.total_loss = text_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)

            init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)
        self.sess.run(init)

    def train_one_epoch(self, dropout_keep=0.8):
        sample_num = len(self.train_tuple_list)
        index_list = list(range(sample_num))
        random.shuffle(index_list)

        total_loss = 0

        step_num = int(math.ceil(sample_num / self.batch_size))
        for step in range(step_num):
            start = step * self.batch_size
            offset = min(start + self.batch_size, sample_num)

            user = []
            item = []
            rating = []
            feature = []
            word_seq = []
            for idx in index_list[start:offset]:
                x = self.train_tuple_list[idx]
                user.append(x[0])
                item.append(x[1])
                rating.append(x[7])
                feature.append(x[3])
                word_seq.append(x[4])
            user = np.asarray(user, dtype=np.int32)
            item = np.asarray(item, dtype=np.int32)
            rating = np.asarray(rating, dtype=np.float32)
            feature = np.asarray(feature, dtype=np.int32)
            word_seq, seq_len = pad_sequence_4_generation(word_seq, self.word2index['<PAD>'])

            feed_dict = {self.user_id: user,
                         self.item_id: item,
                         self.rating: rating,
                         self.feature: feature,
                         self.word_id_seq: word_seq,
                         self.seq_len: seq_len,
                         self.batch_max_len: max(seq_len),
                         self.dropout_keep_prob: dropout_keep}
            _, loss = self.sess.run([self.optimizer, self.total_loss], feed_dict=feed_dict)
            total_loss += loss * (offset - start)

        return total_loss / sample_num

    def validate(self, tuple_list):
        sample_num = len(tuple_list)

        total_loss = 0

        step_num = int(math.ceil(sample_num / self.batch_size))
        for step in range(step_num):
            start = step * self.batch_size
            offset = min(start + self.batch_size, sample_num)

            user = []
            item = []
            rating = []
            feature = []
            word_seq = []
            for x in tuple_list[start:offset]:
                user.append(x[0])
                item.append(x[1])
                rating.append(x[7])
                feature.append(x[3])
                word_seq.append(x[4])
            user = np.asarray(user, dtype=np.int32)
            item = np.asarray(item, dtype=np.int32)
            rating = np.asarray(rating, dtype=np.float32)
            feature = np.asarray(feature, dtype=np.int32)
            word_seq, seq_len = pad_sequence_4_generation(word_seq, self.word2index['<PAD>'])

            feed_dict = {self.user_id: user,
                         self.item_id: item,
                         self.rating: rating,
                         self.feature: feature,
                         self.word_id_seq: word_seq,
                         self.seq_len: seq_len,
                         self.batch_max_len: max(seq_len),
                         self.dropout_keep_prob: 1.0}
            loss = self.sess.run(self.total_loss, feed_dict=feed_dict)
            total_loss += loss * (offset - start)

        return total_loss / sample_num

    def get_prediction(self, tuple_list):
        sample_num = len(tuple_list)
        seq_prediction = []

        step_num = int(math.ceil(sample_num / self.batch_size))
        for step in range(step_num):
            start = step * self.batch_size
            offset = min(start + self.batch_size, sample_num)

            user = []
            item = []
            rating = []
            feature = []
            for x in tuple_list[start:offset]:
                user.append(x[0])
                item.append(x[1])
                rating.append(x[7])
                feature.append(x[3])
            user = np.asarray(user, dtype=np.int32)
            item = np.asarray(item, dtype=np.int32)
            rating = np.asarray(rating, dtype=np.float32)
            feature = np.asarray(feature, dtype=np.int32)

            feed_dict = {self.user_id: user,
                         self.item_id: item,
                         self.rating: rating,
                         self.feature: feature,
                         self.dropout_keep_prob: 1.0}
            predicted_ids = self.sess.run(self.argmax_tokens, feed_dict=feed_dict)
            if predicted_ids.shape[1] != self.seq_max_len:
                pad = np.full((offset - start, self.seq_max_len - predicted_ids.shape[1]), self.word2index['<PAD>'])
                predicted_ids = np.concatenate([predicted_ids, pad], axis=1)
            seq_prediction.append(predicted_ids)

        return np.concatenate(seq_prediction, axis=0)
