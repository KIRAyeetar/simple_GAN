from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import tensorflow as tf


def run():
    iris_data = get_iris_data()
    g_input = np.random.rand(40, 6)
    print(iris_data)

    g_net = G_net(input_data=g_input, hidden_list=[10, 8, 4], hidden_activation_list=[tf.nn.relu, None, None])
    d_net = D_net(input_data=iris_data.iloc[:, 0:4], hidden_list=[5, 2], hidden_activation_list=[None, tf.nn.softmax], rnn_hidden_len=8, sequence_len=4, frame_len=1, rnn_layers_len=1)
    gan = GAN(g_net, d_net)
    gan.build(batch_size=32, lr_d=0.002, lr_g=0.003, steps=2000)


# =================
#  read iris data
# =================
def get_iris_data():
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data)
    iris_data['target'] = iris.target
    iris_data = iris_data[iris_data['target']==2]
    iris_data = iris_data.sample(frac=1, random_state=2018).reset_index(drop=True)
    return iris_data


# =================
#  G-net
# =================
class G_net(object):
    def __init__(self, input_data, hidden_list, hidden_activation_list):
        self.input_data = np.array(input_data)
        self.hidden_list = hidden_list
        self.hidden_activation_list = hidden_activation_list
        self.size = len(self.input_data)
        self.input_len = len(self.input_data[0])
        self.output_len = self.hidden_list[-1]

        self.w_list = [0]*len(hidden_list)
        self.b_list = [0]*len(hidden_list)
        pre_len = self.input_len
        for index in range(len(self.hidden_list)):
            self.w_list[index] = tf.Variable(tf.random_normal([pre_len, hidden_list[index]], stddev=1, seed=1))
            self.b_list[index] = tf.Variable(tf.random_normal([hidden_list[index]], stddev=1, seed=1))
            pre_len = hidden_list[index]

    def build(self, input_data):
        y = input_data
        for index in range(len(self.hidden_list)):
            w = self.w_list[index]
            b = self.b_list[index]
            act = self.hidden_activation_list[index]
            if act is not None:
                y = act(tf.matmul(y, w) + b)
            else:
                y = tf.matmul(y, w) + b
        return y


# =================
#  D-net
# =================
class D_net(object):
    def __init__(self, input_data, hidden_list, hidden_activation_list, rnn_hidden_len, sequence_len, frame_len, rnn_layers_len=1):
        self.input_data = np.array(input_data)
        self.hidden_activation_list = hidden_activation_list
        self.hidden_list = hidden_list
        self.rnn_hidden_len = rnn_hidden_len
        self.rnn_layers_len = rnn_layers_len
        self.sequence_len = sequence_len
        self.frame_len = frame_len
        self.size = len(self.input_data)
        self.input_len = len(self.input_data[0])
        self.output_len = self.hidden_list[-1]

        self.w_list = [0] * len(hidden_list)
        self.b_list = [0] * len(hidden_list)
        pre_len = self.rnn_hidden_len
        # pre_len = self.input_len
        for index in range(len(self.hidden_list)):
            self.w_list[index] = tf.Variable(tf.truncated_normal([pre_len, hidden_list[index]], stddev=1, seed=1))
            self.b_list[index] = tf.Variable(tf.truncated_normal([hidden_list[index]], stddev=1, seed=1))
            pre_len = hidden_list[index]

    def build(self, input_data):
        input_data = tf.reshape(input_data, shape=[-1, self.sequence_len, self.frame_len])
        rnn_cell = [tf.nn.rnn_cell.BasicLSTMCell(self.rnn_hidden_len, state_is_tuple=True, forget_bias=0.8, reuse=tf.AUTO_REUSE, activation=tf.nn.tanh)] * self.rnn_layers_len
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cell, state_is_tuple=True)
        output_data, states = tf.nn.dynamic_rnn(rnn_cell, input_data, dtype=tf.float32)
        y = output_data[:, -1, :]
        for index in range(len(self.hidden_list)):
            w = self.w_list[index]
            b = self.b_list[index]
            act = self.hidden_activation_list[index]
            if act is not None:
                y = act(tf.matmul(y, w) + b)
            else:
                y = tf.matmul(y, w) + b
        return y

        # y = input_data
        # for index in range(len(self.hidden_list)):
        #     w = self.w_list[index]
        #     b = self.b_list[index]
        #     act = self.hidden_activation_list[index]
        #     if act is not None:
        #         y = act(tf.matmul(y, w) + b)
        #     else:
        #         y = tf.matmul(y, w) + b
        # return y


# =================
#  GAN-net
# =================
class GAN(object):
    def __init__(self, g_net, d_net):
        self.g_net = g_net
        self.d_net = d_net

    def build(self, batch_size, lr_d, lr_g, steps):
        g_x = tf.placeholder(tf.float32, [None, self.g_net.input_len])
        d_x = tf.placeholder(tf.float32, [None, self.d_net.input_len])

        y_01_g = tf.placeholder(tf.float32, [None, self.d_net.output_len])
        y_10_g = tf.placeholder(tf.float32, [None, self.d_net.output_len])
        y_01_d = tf.placeholder(tf.float32, [None, self.d_net.output_len])

        g_out = self.g_net.build(g_x)
        d_out_g = self.d_net.build(g_out)
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_g, labels=y_01_g))

        d_out_true = self.d_net.build(d_x)
        d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_g, labels=y_10_g)) + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_true, labels=y_01_d))

        train_step_g = tf.train.AdamOptimizer(lr_g).minimize(g_loss, var_list=self.g_net.w_list+self.g_net.b_list)
        train_step_d = tf.train.AdamOptimizer(lr_d).minimize(d_loss, var_list=self.d_net.w_list+self.d_net.b_list)

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            for i in range(steps):
                start_g = (i * batch_size) % self.g_net.size
                end_g = start_g + batch_size

                start_d = (i * batch_size) % self.d_net.size
                end_d = start_d + batch_size

                if end_g > self.g_net.size:
                    end_g = self.g_net.size
                if end_d > self.d_net.size:
                    end_d = self.d_net.size

                y_01_g_ = np.concatenate((np.zeros(shape=[end_g - start_g, 1]), np.ones(shape=[end_g - start_g, 1])), axis=1)
                y_10_g_ = np.concatenate((np.ones(shape=[end_g - start_g, 1]), np.zeros(shape=[end_g - start_g, 1])), axis=1)
                y_01_d_ = np.concatenate((np.zeros(shape=[end_d - start_d, 1]), np.ones(shape=[end_d - start_d, 1])), axis=1)

                sess.run(train_step_d, feed_dict={g_x: self.g_net.input_data[start_g: end_g, :], y_10_g: y_10_g_,
                                                  d_x: self.d_net.input_data[start_d: end_d, :], y_01_d: y_01_d_})

                sess.run(train_step_g, feed_dict={g_x: self.g_net.input_data[start_g: end_g, :], y_01_g: y_01_g_})

                g_loss_ = sess.run(g_loss, feed_dict={g_x: self.g_net.input_data[start_g: end_g, :], y_01_g: y_01_g_})
                d_loss_ = sess.run(d_loss, feed_dict={g_x: self.g_net.input_data[start_g: end_g, :], y_10_g: y_10_g_,
                                                      d_x: self.d_net.input_data[start_d: end_d, :], y_01_d: y_01_d_})

                d_out_g_ = sess.run(d_out_g, feed_dict={g_x: self.g_net.input_data})
                d_out_true_ = sess.run(d_out_true, feed_dict={d_x: self.d_net.input_data})
                correct_prediction = np.equal(np.argmax(d_out_g_, 1), np.ones(self.g_net.size))
                g_acc_ = np.mean(correct_prediction)
                correct_prediction = np.equal(np.argmax(d_out_true_, 1), np.ones(self.d_net.size))
                d_acc_ = np.mean(correct_prediction)

                if i % 10 == 0:
                    print("round=%d, g_acc=%f, d_acc=%f, g_loss=%f, d_loss=%f" % (i, g_acc_, d_acc_, g_loss_, d_loss_))

                if i % 100 == 0:
                    print(sess.run(g_out, feed_dict={g_x: self.g_net.input_data[start_g: end_g, :]}))
                    # print(sess.run(g_out, feed_dict={g_x: self.g_net.input_data[start_g: end_g, :], y_01_g: y_01_g_}))


if __name__ == '__main__':
    run()
