from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import tensorflow as tf


def run():
    iris_data = get_iris_data()
    g_input = np.random.rand(40, 4)
    print(iris_data)

    g_net = G_net(input_data=g_input, hidden_list=[10, 20, 4], hidden_activation_list=[tf.nn.relu, None, None])
    d_net = D_net(input_data=iris_data.iloc[:, 0:4], hidden_list=[3, 1], hidden_activation_list=[None, None], rnn_hidden_len=8, sequence_len=4, frame_len=1, rnn_layers_len=1)
    gan = GAN(g_net, d_net)
    gan.build(batch_size=32, lr_d=0.03, lr_g=0.03, steps=5000)


# =================
#  read iris data
# =================
def get_iris_data():
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data)
    iris_data['target'] = iris.target
    iris_data = iris_data[iris_data['target']==1]
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

        g_out = self.g_net.build(g_x)
        d_out_g = self.d_net.build(g_out)
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_g, labels=tf.ones_like(d_out_g)))

        d_out_true = self.d_net.build(d_x)
        d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_g, labels=tf.zeros_like(d_out_g))) + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_true, labels=tf.ones_like(d_out_true)))

        train_step_g = tf.train.AdamOptimizer(lr_g).minimize(g_loss)
        train_step_d = tf.train.AdamOptimizer(lr_d).minimize(d_loss)

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

                sess.run(train_step_d, feed_dict={g_x: self.g_net.input_data[start_g: end_g, :],
                                                  d_x: self.d_net.input_data[start_d: end_d, :]})
                sess.run(train_step_g, feed_dict={g_x: self.g_net.input_data[start_g: end_g, :]})

                g_loss_ = sess.run(g_loss, feed_dict={g_x: self.g_net.input_data[start_g: end_g, :]})
                d_loss_ = sess.run(d_loss, feed_dict={g_x: self.g_net.input_data[start_g: end_g, :],
                                                      d_x: self.d_net.input_data[start_d: end_d, :]})
                if i % 10 == 0:
                    print("round=%d, g_loss=%f, d_loss=%f" % (i, g_loss_, d_loss_))
                    # print(sess.run(g_out, feed_dict={g_x: self.g_net.input_data[start_g: end_g, :]}))
                if i % 500 == 0:
                    print(sess.run(d_out_g, feed_dict={g_x: self.g_net.input_data[start_g: end_g, :]}))

if __name__ == '__main__':
    run()
