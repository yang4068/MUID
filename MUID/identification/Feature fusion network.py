import tensorflow as tf
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import time
import os
import normalization
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
np.set_printoptions(threshold=np.inf)


class SelfAttention():
    def __init__(self, mode='embedded', intermediate_dim=None, add_residual=True):
        self.mode = mode
        self.intermediate_dim = intermediate_dim
        self.add_residual = add_residual

    def __call__(self, ip):
        """
        Returns:Tensor: Output tensor of the Non-Local block with the same shape as the input.
        ip: tf.Tensor  Input tensor.
        """
        if self.mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
        self.input_shape = ip.get_shape().as_list()

        rank = len(self.input_shape)
        print('-----------------self_attention')
        print('rank=', rank)
        print('input_shape', self.input_shape)
        if rank not in [3, 4, 5]:
            raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')
        elif rank == 3:
            batchsize, dims, channels = self.input_shape
        elif rank == 4:
            batchsize, height, width, channels = self.input_shape
        else:
            batchsize, *dims, channels = self.input_shape

        # verify correct intermediate dimension specified
        if self.intermediate_dim is None:
            self.intermediate_dim = channels // 2
            if self.intermediate_dim < 1:
                self.intermediate_dim = 1
        else:
            self.intermediate_dim = int(self.intermediate_dim)
            if self.intermediate_dim < 1:
                raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

        if self.mode == 'gaussian':  # Gaussian instantiation
            x1 = tf.reshape(ip, (-1, channels))  # xi
            x2 = tf.reshape(ip, (-1, channels))  # xj
            f = tf.matmul(x1, x2, transpose_b=True)
            f = tf.nn.softmax(f)

        elif self.mode == 'dot':  # Dot instantiation
            # theta path
            self.theta = self.convND(ip, rank, self.intermediate_dim)
            self.theta = tf.reshape(self.theta, (-1, height * width, self.intermediate_dim))

            # phi path
            self.phi = self.convND(ip, rank, self.intermediate_dim)
            self.phi = tf.reshape(self.phi, (-1, height * width, self.intermediate_dim))

            self.f = tf.matmul(self.theta, self.phi, transpose_b=True)

            size = self.f.get_shape().as_list()

            # scale the values to make it size invariant
            self.f = (1. / float(size[-1])) * self.f

        else:  # Embedded Gaussian instantiation
            # theta path
            self.theta = self.convND(ip, rank, self.intermediate_dim)
            self.theta = tf.reshape(self.theta, (-1, height * width, self.intermediate_dim))

            # phi path
            self.phi = self.convND(ip, rank, self.intermediate_dim)
            self.phi = tf.reshape(self.phi, (-1, height * width, self.intermediate_dim))

            # if self.compression > 1:
            #     # shielded computation
            #     phi = tf.layers.MaxPooling1D(self.compression)(phi)

            self.f = tf.matmul(self.theta, self.phi, transpose_b=True)
            self.f = tf.reshape(self.f, (-1, height * width * height * width))
            self.f = tf.nn.softmax(self.f, axis=-1)
            self.f = tf.reshape(self.f, (-1, height * width, height * width))

        # g path
        self.g = self.convND(ip, rank, self.intermediate_dim)
        self.g = tf.reshape(self.g, (-1, height * width, self.intermediate_dim))

        # if self.compression > 1 and self.mode == 'embedded':
        #     # shielded computation
        #     g = tf.layers.MaxPooling1D(self.compression)(g)

        # compute output path
        self.y = tf.matmul(self.f, self.g)

        # reshape to input tensor format
        if rank == 3:
            self.y = tf.reshape(self.y, (dims, self.intermediate_dim))
        else:
            self.y = tf.reshape(self.y, (-1, height, width, self.intermediate_dim))

        # project filters
        self.y = self.convND(self.y, rank, channels)

        # residual connection
        if self.add_residual:
            self.y = ip + self.y

        return self.y

    def convND(self, ip, rank, channels):
        assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

        if rank == 3:
            x = tf.layers.conv1d(ip, channels, 1, padding='same', use_bias=False,
                                 kernel_initializer=tf.initializers.he_normal())
        elif rank == 4:
            x = tf.layers.conv2d(ip, channels, 1, padding='same', use_bias=False, data_format='channels_last',
                                 kernel_initializer=tf.initializers.he_normal())
        else:
            x = tf.layers.conv3d(ip, channels, (1, 1, 1), padding='same', use_bias=False,
                                 kernel_initializer=tf.initializers.he_normal())
        return x


class SourceModel():
    def __init__(
            self,
            train_data=None,
            s_test=None,
            m=100,
            m2=100,
            n=800,
            k=6,
            batch_size=24,
            learning_rate=0.0001,
            training_epochs=200,
            param_file=False,
            is_train=True
    ):
        self.con2_3_s = None
        self.con1_3_s = None
        self.train = train_data
        self.s_test = s_test
        self.m, self.m2, self.n, self.k = m, m2, n, k
        self.batch_size = batch_size
        self.lr = learning_rate
        self.is_train = is_train
        self.training_epochs = training_epochs
        self.buildNetwork()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if is_train is True:
            if param_file is True:
                self.saver.restore(self.sess, "./2p_env1/train.ckpt")
                print("loading neural-network params...")
                self.learn()
            else:
                print("learning with initialization!")
                self.learn()
        else:
            self.saver.restore(self.sess, "./2p_env1/train.ckpt")
            print("loading neural-network params...")
            test_pro = self.test(self.s_test)
            print('test acc:', test_pro)

    def buildNetwork(self):
        self.x_1 = tf.placeholder(tf.float32, shape=[None, self.m, self.n, 1], name='doppler_origin')
        self.x_2 = tf.placeholder(tf.float32, shape=[None, self.m2, self.n, 1], name='acf_origin')

        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.y = tf.placeholder(tf.float32, shape=[None, self.k], name='true_label_vector')
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.k], name='predict_vector')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope('stream1'):
            with tf.variable_scope('sharedModel_1'):
                w_initializer = tf.random_normal_initializer(stddev=0.02)
                b_initializer = tf.constant_initializer(0.01)

                w_e_conv1_1 = tf.get_variable('w1', [5, 5, 1, 16], initializer=w_initializer)
                b_e_conv1_1 = tf.get_variable('b1', [16, ], initializer=b_initializer)
                con1_1_s = lrelu(tf.add(self.conv2d(self.x_1, w_e_conv1_1), b_e_conv1_1))

                w_e_conv1_2 = tf.get_variable('w2', [5, 5, 16, 32], initializer=w_initializer)
                b_e_conv1_2 = tf.get_variable('b2', [32, ], initializer=b_initializer)
                con1_2_s = lrelu(tf.add(self.conv2d(con1_1_s, w_e_conv1_2), b_e_conv1_2))

                w_e_conv1_3 = tf.get_variable('w3', [5, 5, 32, 64], initializer=w_initializer)
                b_e_conv1_3 = tf.get_variable('b3', [64, ], initializer=b_initializer)
                con1_3_s = lrelu(tf.add(self.conv2d(con1_2_s, w_e_conv1_3), b_e_conv1_3))

            with tf.variable_scope('fineTuning_1'):
                w_e_conv1_4 = tf.get_variable('w4', [5, 5, 64, 128], initializer=w_initializer)
                b_e_conv1_4 = tf.get_variable('b4', [128, ], initializer=b_initializer)
                self.con1_4_s = lrelu(tf.add(self.conv2d(con1_3_s, w_e_conv1_4), b_e_conv1_4))

            with tf.variable_scope('self_attention_1'):
                non_local_block_1 = SelfAttention(intermediate_dim=None, mode='embedded', add_residual=True)
                self.z1 = non_local_block_1(self.con1_4_s)

        with tf.variable_scope('stream2'):
            with tf.variable_scope('sharedModel_2'):
                w_initializer = tf.random_normal_initializer(stddev=0.02)
                b_initializer = tf.constant_initializer(0.01)

                w_e_conv2_1 = tf.get_variable('w1', [5, 5, 1, 16], initializer=w_initializer)
                b_e_conv2_1 = tf.get_variable('b1', [16, ], initializer=b_initializer)
                con2_1_s = lrelu(tf.add(self.conv2d(self.x_2, w_e_conv2_1), b_e_conv2_1))

                w_e_conv2_2 = tf.get_variable('w2', [5, 5, 16, 32], initializer=w_initializer)
                b_e_conv2_2 = tf.get_variable('b2', [32, ], initializer=b_initializer)
                con2_2_s = lrelu(tf.add(self.conv2d(con2_1_s, w_e_conv2_2), b_e_conv2_2))

                w_e_conv2_3 = tf.get_variable('w3', [5, 5, 32, 64], initializer=w_initializer)
                b_e_conv2_3 = tf.get_variable('b3', [64, ], initializer=b_initializer)
                con2_3_s = lrelu(tf.add(self.conv2d(con2_2_s, w_e_conv2_3), b_e_conv2_3))

            with tf.variable_scope('fineTuning_2'):
                w_e_conv2_4 = tf.get_variable('w4', [5, 5, 64, 128], initializer=w_initializer)
                b_e_conv2_4 = tf.get_variable('b4', [128, ], initializer=b_initializer)
                self.con2_4_s = lrelu(tf.add(self.conv2d(con2_3_s, w_e_conv2_4), b_e_conv2_4))

            with tf.variable_scope('self_attention_2'):
                non_local_block_2 = SelfAttention(intermediate_dim=None, mode='embedded', add_residual=True)
                self.z2 = non_local_block_2(self.con2_4_s)

        with tf.variable_scope('branch_attention'):
            sum_map = tf.add(self.z1, self.z2)
            self.avg_pool = tf.reduce_mean(sum_map, axis=[1, 2], keepdims=True)

            with tf.variable_scope('attention_fc1'):
                self.avg_pool = tf.reshape(self.avg_pool, (-1, 128))
                w1_attention = tf.get_variable('w1_attention', [128, 32], initializer=w_initializer)
                b1_attention = tf.get_variable('b1_attention', [32], initializer=b_initializer)
                self.se_fc1 = tf.nn.relu(
                    tf.layers.batch_normalization(tf.add(tf.matmul(self.avg_pool, w1_attention), b1_attention)))

            with tf.variable_scope('attention_fc2_1'):
                w2_1_attention = tf.get_variable('w2_1_attention', [32, 128], initializer=w_initializer)
                b2_1_attention = tf.get_variable('b2_1_attention', [128], initializer=b_initializer)
                self.se_fc2_1 = tf.add(tf.matmul(self.se_fc1, w2_1_attention), b2_1_attention)

            with tf.variable_scope('attention_fc2_2'):
                w2_2_attention = tf.get_variable('w2_2_attention', [32, 128], initializer=w_initializer)
                b2_2_attention = tf.get_variable('b2_2_attention', [128], initializer=b_initializer)
                self.se_fc2_2 = tf.add(tf.matmul(self.se_fc1, w2_2_attention), b2_2_attention)

            weight = tf.stack([self.se_fc2_1, self.se_fc2_2])
            attention_weights = tf.nn.softmax(weight, axis=0)
            self.attention_weights = tf.reshape(attention_weights[0], (-1, 1, 1, 128))
            self.weighted_features_1 = tf.multiply(self.z1, self.attention_weights)
            self.weighted_features_2 = tf.multiply(self.z2, 1 - self.attention_weights)

        with tf.variable_scope('tailed'):
            self.combined_features = tf.add(self.weighted_features_1, self.weighted_features_2)
            self.flatten = tf.reshape(self.combined_features, (-1, 7 * 50 * 128))
            w_fc_1 = tf.get_variable('wc1', [7 * 50 * 128, 500], initializer=w_initializer)
            b_fc_1 = tf.get_variable('bc1', [500], initializer=b_initializer)
            self.logits = tf.nn.relu(tf.matmul(self.flatten, w_fc_1) + b_fc_1)
            self.logits = tf.nn.dropout(self.logits, self.keep_prob)

            w_fc_2 = tf.get_variable('wc3', [500, self.k], initializer=w_initializer)
            b_fc_2 = tf.get_variable('bc3', [self.k], initializer=b_initializer)
            self.res = tf.matmul(self.logits, w_fc_2) + b_fc_2
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
        ckpt_address = "./2p_env1/train.ckpt"
        batch_size = self.batch_size
        data_train = self.train
        epoch = self.training_epochs
        train_loss, train_acc = [], []
        test_acc = []
        init_time = time.time()
        for j in range(epoch):
            t_loss, t_acc = [], []
            random.shuffle(data_train)
            for step in range(0, len(data_train), batch_size):
                batch_xs_doppler, batch_xs_acf, batch_ys, = [], [], []

                if (step + batch_size + 1) > len(data_train):
                    pp = len(data_train) - step
                    for batch in range(step, step + pp):
                        a = data_train[batch]
                        batch_xs_doppler.append(a[0])
                        batch_xs_acf.append(a[1])
                        batch_ys.append(a[2])
                else:
                    for batch in range(step, step + batch_size):
                        a = data_train[batch]
                        batch_xs_doppler.append(a[0])
                        batch_xs_acf.append(a[1])
                        batch_ys.append(a[2])

                batch_xs_doppler = np.array(batch_xs_doppler)
                batch_xs_acf = np.array(batch_xs_acf)
                batch_ys = np.array(batch_ys)
                batch_labels = np.argmax(batch_ys, 1)

                batch_xs_doppler = np.reshape(batch_xs_doppler, [-1, self.m, self.n, 1])
                batch_xs_acf = np.reshape(batch_xs_acf, [-1, self.m2, self.n, 1])
                batch_ys = batch_ys.astype(np.float32)
                batch_ys = np.nan_to_num(batch_ys)
                batch_ys = np.reshape(batch_ys, [-1, self.k])

                _, c, a, = self.sess.run([self.optimizer, self.loss, self.accuracy],
                                         feed_dict={self.x_1: batch_xs_doppler, self.x_2: batch_xs_acf,
                                                    self.labels: batch_labels, self.y: batch_ys,
                                                    self.keep_prob: 0.5})

                t_loss.append(c)
                t_acc.append(a)

            train_loss.append(np.mean(t_loss))
            train_acc.append(np.mean(t_acc))

            test_pro = self.test(self.s_test)
            test_acc.append(test_pro)

            now_time = time.time()
            print("Total Epoch:", '%d' % (j + 1), '| time: ', now_time - init_time)
            print("loss_1: ", train_loss[-1], '| train_acc: ', train_acc[-1], "| s_acc: ", test_acc[-1])
            init_time = time.time()

        self.saver.save(self.sess, ckpt_address)
        print("save model!")

        # plt.figure()
        # plt.title('classification_accuracy')
        # plt.xlabel('epoch')
        # plt.ylabel('accuracy')
        # plt.plot(range(len(test_acc)), test_acc, label='accuracy')
        # plt.grid(True, linestyle='--', alpha=0.4)
        # plt.legend()
        # plt.xlim(0, len(test_acc))
        # plt.ylim(0, 1)
        # plt.show()

    def test(self, data_test):
        correct_count = 0
        test_sum = 0
        batch_xs_doppler, batch_xs_acf, batch_ys, = [], [], []
        random.shuffle(data_test)
        for n in range(len(data_test)):
            a = data_test[n]
            batch_xs_doppler.append(a[0])
            batch_xs_acf.append(a[1])
            batch_ys.append(a[2])

        batch_xs_doppler = np.array(batch_xs_doppler)
        batch_xs_acf = np.array(batch_xs_acf)
        batch_xs_doppler = np.reshape(batch_xs_doppler, [-1, self.m, self.n, 1])
        batch_xs_acf = np.reshape(batch_xs_acf, [-1, self.m2, self.n, 1])

        prob = self.sess.run(self.y_,
                             feed_dict={self.x_1: batch_xs_doppler, self.x_2: batch_xs_acf, self.keep_prob: 1})

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
    for f in range(freq_length):
        interpolated_data[f, :] = np.interp(np.linspace(0, time_length, 800), range(time_length), train_data[f, :])
    return interpolated_data


def package_t(train_data, class_num, p_index):
    doppler_mat = train_data[0]
    acf_mat = train_data[1]
    doppler_matnorm = batchNormalization(doppler_mat)
    acf_matnorm = batchNormalization(acf_mat)
    flag = p_index
    class_vector = create_domain(class_num, flag)

    return [doppler_matnorm, acf_matnorm, class_vector]


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


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    num_class = 6
    per = 0

    X_train, X_test = [], []
    all_data_path = '/home/ubuntu/PycharmProjects/yx/envs_new/'
    for env in os.listdir(all_data_path)[0:1]:
        env_path = all_data_path + 'data_1_10_2p'
        for multi_index in range(0, 1):
            doppler_file_path = env_path + '/6Doppler_spectrum_cali/'
            acf_file_path = env_path + '/7acf_feature_mrc_100/'

            temp_path = os.listdir(doppler_file_path)
            temp_path.sort(key=lambda x: int(x[-1]))
            for perdir in temp_path:
                doppler_path = os.path.join(doppler_file_path, perdir)
                acf_path = os.path.join(acf_file_path, perdir)

                temp_paths = os.listdir(doppler_path)
                temp_paths.sort(key=lambda x: int(x[10:-4]))
                length = len(temp_paths)
                train_val_data = []
                for filename in temp_paths:
                    doppler_data_path = os.path.join(doppler_path, filename)
                    acf_data_path = os.path.join(acf_path, filename)

                    with open(doppler_data_path, 'rb') as handle1:
                        doppler_data_temp = pickle.load(handle1)
                    doppler_data_interp = time_reshape(doppler_data_temp[0])

                    with open(acf_data_path, 'rb') as handle2:
                        acf_data_temp = pickle.load(handle2)
                    acf_data_interp = time_reshape(acf_data_temp)

                    label = perdir[-1]
                    data_label = [doppler_data_interp, acf_data_interp, label]

                    data_nor = package_t(data_label, num_class, per)
                    train_val_data.append(data_nor)

                np.random.seed(1)
                np.random.shuffle(train_val_data)
                train_data, test_data = train_test_split(train_val_data, test_size=0.25, shuffle=False)

                X_train.extend(train_data)
                X_test.extend(test_data)

                per += 1
            per = 0

    print(np.array(train_data).shape)
    print('train_data_len is:', len(X_train))
    print('test_data_len is:', len(X_test))
    SourceModel(train_data=X_train, s_test=X_test, is_train=True)
