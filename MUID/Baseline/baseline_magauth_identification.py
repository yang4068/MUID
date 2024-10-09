import tensorflow as tf
import numpy as np
import random
import pickle
# import normalization
import time
import os
from identification import normalization
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine, euclidean
from scipy.special import kl_div
from itertools import combinations
from imblearn.over_sampling import RandomOverSampler

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

np.set_printoptions(threshold=np.inf)


class SourceModel():
    def __init__(
            self,
            train_data=None,
            s_test=None,
            k=2,
            batch_size=24,
            learning_rate=0.0001,
            training_epochs=600,
            param_file=False,
            is_train=True
    ):
        self.train = train_data
        self.s_test = s_test
        self.k = k
        self.batch_size = batch_size
        self.lr = learning_rate
        self.is_train = is_train
        self.training_epochs = training_epochs
        self.buildNetwork()
        print("Neural networks build!")
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if is_train is True:
            if param_file is True:
                self.saver.restore(self.sess, "./baseline/train.ckpt")
                print("loading neural-network params...")
                self.learn()
            else:
                print("learning with initialization!")
                self.learn()

        else:
            self.saver.restore(self.sess, "./baseline/train.ckpt")
            print("loading neural-network params...")

    def buildNetwork(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 10], name='distance_vector')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.y = tf.placeholder(tf.float32, shape=[None, self.k], name='true_label')
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.k], name='predict')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope('fc'):
            w_initializer = tf.random_normal_initializer(stddev=0.02)
            b_initializer = tf.constant_initializer(0.01)

            w_fc1 = tf.get_variable('wfc1', [10, 30], initializer=w_initializer)
            b_fc1 = tf.get_variable('bfc1', [30], initializer=b_initializer)
            self.logits1 = tf.nn.relu(tf.matmul(self.x, w_fc1) + b_fc1)

            w_fc2 = tf.get_variable('wfc2', [30, self.k], initializer=w_initializer)
            b_fc2 = tf.get_variable('bfc2', [self.k], initializer=b_initializer)
            self.res = tf.matmul(self.logits1, w_fc2) + b_fc2
            self.y_ = tf.nn.softmax(self.res)

        with tf.variable_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.res))
            self.loss = softmax_loss

        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        with tf.variable_scope('acc'):
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.y, 1)), tf.float32))

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

    def learn(self):
        batch_size = self.batch_size
        data_train = self.train
        epoch = self.training_epochs
        train_loss, train_acc = [], []
        test_acc = []
        init_time = time.time()

        for j in range(epoch):
            t_loss, t_acc = [], []
            s_given_posi = []
            s_given_nega = []

            random.shuffle(data_train)
            for step in range(0, len(data_train), batch_size):
                batch_xs, batch_ys, = [], []

                if (step + batch_size + 1) > len(data_train):
                    pp = len(data_train) - step
                    for batch in range(step, step + pp):
                        a = data_train[batch]
                        batch_xs.append(a[0])
                        batch_ys.append(a[1])
                else:
                    for batch in range(step, step + batch_size):
                        a = data_train[batch]
                        batch_xs.append(a[0])
                        batch_ys.append(a[1])

                batch_xs = np.array(batch_xs)
                batch_ys = np.array(batch_ys)
                batch_labels = np.argmax(batch_ys, 1)

                _, c, a, train_pred = self.sess.run([self.optimizer, self.loss, self.accuracy, self.y_],
                                                    feed_dict={self.x: batch_xs, self.labels: batch_labels,
                                                               self.y: batch_ys,
                                                               self.keep_prob: 0.5})

                t_loss.append(c)
                t_acc.append(a)

                s_given_posi.extend(train_pred[batch_labels == 1][:, 1])
                s_given_nega.extend(train_pred[batch_labels == 0][:, 1])

            train_loss.append(np.mean(t_loss))
            train_acc.append(np.mean(t_acc))

            s_g_p_distribution, bins = np.histogram(s_given_posi, bins=100, range=(0, 1), density=True)
            s_g_n_distribution, bins = np.histogram(s_given_nega, bins=100, range=(0, 1), density=True)
            self.llr = s_g_p_distribution / s_g_n_distribution

            test_pro = self.test(self.s_test)
            test_acc.append(test_pro)

            now_time = time.time()

            print("Total Epoch:", '%d' % (j + 1), '| time: ', now_time - init_time)
            print("loss: ", train_loss[-1], '| train_acc: ', train_acc[-1], "| s_acc: ", test_acc[-1])
            init_time = time.time()

        self.saver.save(self.sess, "./baseline/train.ckpt")
        print("save model!")

    def test(self, data_test):
        correct_count = 0
        test_sum = 0
        batch_xs = []
        batch_ys = []
        random.shuffle(data_test)
        for n in range(len(data_test)):
            a = data_test[n]
            batch_xs.append(a[0])
            batch_ys.append(a[1])
        batch_xs = np.array(batch_xs)
        prob = self.sess.run(self.y_, feed_dict={self.x: batch_xs, self.keep_prob: 1})

        class_pre = np.argmax(prob, axis=1)
        class_true = np.argmax(batch_ys, axis=1)
        for i in range(len(class_true)):
            if class_pre[i] == class_true[i]:
                correct_count += 1
            test_sum += 1
        correct_pro = float(correct_count) / float(test_sum)
        return correct_pro

    def changeLabel(self, one_hot):
        return np.argmax(one_hot)


def weight_variable(shape, name, stddev=0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.random_normal_initializer(stddev=stddev, dtype=dtype))
    return var


def bias_variable(shape, name, bias_start=0.01, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return var


def conv2d(x, output_channels, name, k_h=5, k_w=5, reuse=False):
    x_shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse):
        w = weight_variable(shape=[k_h, k_w, x_shape[-1], output_channels], name='weights')
        b = bias_variable([output_channels], name='biases')
        conv = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME') + b
        return conv


def lrelu(x, leak=0.02):
    return tf.maximum(x, leak * x)


def batchNormalization(data):
    data1 = normalization.MINMAXNormalization(data)
    return data1


def time_reshape(train_data):
    time_length = train_data.shape[1]
    freq_length = train_data.shape[0]
    interpolated_data = np.zeros((freq_length, 800))
    if time_length <= 800:
        for f in range(freq_length):
            interpolated_data[f, :] = np.interp(np.linspace(0, time_length, 800), range(time_length), train_data[f, :])
    else:
        interpolated_data = train_data[:, :800]
    return interpolated_data


def package_t(train_data, class_num, p_index):
    csi_rx = train_data[0]
    csi_norm = batchNormalization(csi_rx)
    flag = p_index
    class_vector = create_domain(class_num, flag)
    return [csi_norm, class_vector]


def package(train_data, class_num, p_index):
    csi_rx = train_data[0][0]
    csi_norm = batchNormalization(csi_rx)
    flag = p_index
    class_vector = create_domain(class_num, flag)
    return [csi_norm, class_vector]


def create_domain(domain_num, domian_flag):
    domian_vector = np.zeros([domain_num])
    domian_vector[domian_flag] = 1
    return domian_vector


def calculate_feature_distances(query_features, candidate_features):
    num_features = len(query_features)
    feature_distances = []
    for i in range(num_features):
        query_feature = query_features[i]
        candidate_feature = candidate_features[i]
        if i < 5:
            distance = kl_div(query_feature, candidate_feature).sum()
        elif i == 5 or i == 6:
            distance = euclidean(query_feature, candidate_feature)
        else:
            distance = cosine(query_feature, candidate_feature)

        feature_distances.append(distance)

    return feature_distances


def package_dist(pairs):
    data_l1, data_l2 = pairs
    distan = calculate_feature_distances(data_l1[0], data_l2[0])
    judge_label = 1 if data_l1[1] == data_l2[1] else 0

    return distan, judge_label


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    env_index = 0
    train_features_label_list, query_features_label_list, candidate_features_label_list = [], [], []
    X_train_oringin, X_test_oringin = [], []
    y_train_oringin, y_test_oringin = [], []

    all_data_path = '/home/ubuntu/PycharmProjects/yx/baseline_data/'
    for env in os.listdir(all_data_path)[0:1]:
        env_path = all_data_path + 'data_1_10_3p'
        for multi_index in range(0, 1):
            multi_path = env_path + '/features_v3_low/'

            temp_path = os.listdir(multi_path)
            temp_path.sort(key=lambda x: int(x[-1]))
            for perdir in temp_path:
                path = os.path.join(multi_path, perdir)
                temp_paths = os.listdir(path)
                temp_paths.sort(key=lambda x: int(x[11:-4]))
                length = len(temp_paths)

                train_val_data, data_label_list = [], []

                for filename in temp_paths:
                    data_path = os.path.join(path, filename)
                    with open(data_path, 'rb') as handle:
                        data_temp = pickle.load(handle)

                    label = perdir[-1]
                    data_label = [data_temp, label]
                    data_label_list.append(data_label)

                np.random.seed(1)
                np.random.shuffle(data_label_list)
                train_features_label, test_features_label = train_test_split(data_label_list, test_size=0.5,
                                                                             shuffle=False)
                train_features_label_list.extend(train_features_label)

                query_features_label, candidate_features_label = train_test_split(train_features_label, test_size=0.5,
                                                                                  shuffle=False)
                query_features_label_list.extend(query_features_label)
                candidate_features_label_list.extend(candidate_features_label)

        env_index += 1

    train_combination_list = list(combinations(train_features_label_list, 2))
    for pair in train_combination_list:
        train_dist, label = package_dist(pair)
        X_train_oringin.append(train_dist)
        y_train_oringin.append(label)

    oversampler = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_oringin, y_train_oringin)

    for d_l1 in query_features_label_list:
        for d_l2 in candidate_features_label_list:
            test_dist, label = package_dist([d_l1, d_l2])
            X_test_oringin.append(test_dist)
            y_test_oringin.append(label)

    X_test_resampled, y_test_resampled = oversampler.fit_resample(X_test_oringin, y_test_oringin)

    X_train = [[x, create_domain(2, z)] for x, z in zip(X_train_resampled, y_train_resampled)]
    X_test = [[x, create_domain(2, z)] for x, z in zip(X_test_resampled, y_test_resampled)]

    SourceModel(train_data=X_train, s_test=X_test, is_train=True)
