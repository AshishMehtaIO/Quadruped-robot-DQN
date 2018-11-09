import tensorflow as tf
import numpy as np
import random
import serial
from collections import deque
import itertools


class QNetwork:
    '''Q Network for learning walking policy'''

    def __init__(self, ip_size=4, op_size=16, hidden1_size=8, hidden2_size=16, batch_size=128, lr=5e-4, y=0.95, e=0.5,
                 queue_len=512):
        self.ip_size = ip_size
        self.op_size = op_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.batch_size = batch_size
        self.lr = lr
        self.y = y
        self.e = e
        self.Rall = 0
        self.train_step = 1
        self.queue_len = queue_len
        self.circular_queue = deque([], maxlen=self.queue_len)

    def init_network(self, sess):
        self.ip = tf.placeholder(shape=[None, self.ip_size], dtype=tf.float32, name='Input_layer')
        self.W1 = tf.Variable(tf.random_uniform(shape=[self.ip_size, self.hidden1_size],
                                                dtype=tf.float32, name='Ip_hidden1_weights'))
        tf.summary.histogram('W_i_h1', self.W1)
        # self.b1 = tf.Variable(tf.zeros(self.hidden_size))
        # tf.summary.histogram('b_ih', self.b1)
        # self.hidden=tf.add(tf.matmul(self.ip,self.W1),self.b1)
        self.hidden1 = tf.matmul(self.ip, self.W1)
        self.W2 = tf.Variable(tf.random_uniform(shape=[self.hidden1_size, self.hidden2_size],
                                                dtype=tf.float32, name='Hidden1_hidden2_weights'))
        tf.summary.histogram('W_h1_h2', self.W2)
        self.hidden2 = tf.matmul(self.hidden1, self.W2)
        self.W3 = tf.Variable(tf.random_uniform(shape=[self.hidden2_size, self.op_size],
                                                dtype=tf.float32, name='Hidden2_output_weights'))
        tf.summary.histogram('W_h2_o', self.W3)
        # self.b2 = tf.Variable(tf.zeros(self.op_size))
        # tf.summary.histogram('b_ho', self.b2)
        # self.Qout=tf.add(tf.matmul(self.hidden,self.W2),self.b2)
        self.Qout = tf.matmul(self.hidden2, self.W3)
        self.predict = tf.argmax(self.Qout, 1)
        self.nextQ = tf.placeholder(shape=[None, self.op_size], dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.nextQ - self.Qout), reduction_indices=[1]))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.updateModel = self.trainer.minimize(self.loss)
        self.init_variables = tf.global_variables_initializer()
        # Node to save weights
        self.saver = tf.train.Saver()
        # Graph scalars in TfBoard
        self.total_reward = tf.Variable(self.Rall, dtype=tf.float32)
        self.randomness_var = tf.Variable(self.e, dtype=tf.float32)
        tf.summary.scalar('Total_Reward', self.total_reward)
        tf.summary.scalar('Randomness', self.randomness_var)
        tf.summary.scalar('Loss', self.loss)
        self.summ = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./', sess.graph)

    def collect_experiece(self, state_current, sess, S1):
        a, allQ = sess.run([self.predict, self.Qout], feed_dict={self.ip: state_current})
        if np.random.rand(1) < self.e:
            a[0] = random.randrange(0, self.op_size)
        # send action a to MCU
        S1.serial_write(a[0])
        input_char = S1.serial_read()
        r = input_char[2]
        if r > 127:
            r = r - 256
        state_next = dense_to_onehot(input_char[0], input_char[1])
        if np.array_equal(state_current, state_next):
            r = -100
        r = r / 100
        # save experience to queue and disk
        self.save_experience(state_current, a[0], r, state_next)
        # Reduce randomness
        self.e = self.e / 1.005
        self.train_step = self.train_step + 1
        return state_next, input_char

    def experience_replay(self, state_current_exp, action_exp, reward_exp, state_next_exp, sess):
        aBatch, allQ = sess.run([self.predict, self.Qout],
                                feed_dict={self.ip: np.reshape(state_current_exp, (self.batch_size, self.ip_size))})
        # Obtain the Q' values by feeding the new state through our network
        Q1 = sess.run(self.Qout, feed_dict={self.ip: np.reshape(state_current_exp, (self.batch_size, self.ip_size))})
        # Obtain maxQ' and set our target value for chosen action
        maxQ1Batch = Q1.max(axis=1)
        targetQBatch = allQ
        for targetQ, a, maxQ1, r in zip(targetQBatch, aBatch, maxQ1Batch, reward_exp):
            targetQ[a] = r + self.y * maxQ1
        # Train our network using target and predicted Q values
        sess.run(self.updateModel, feed_dict={self.ip: np.reshape(state_current_exp, (self.batch_size, self.ip_size)),
                                              self.nextQ: targetQBatch})
        # Add Reward
        self.Rall = self.Rall + np.sum(reward_exp)
        # update TfBoard Scalars
        sess.run(self.total_reward.assign(self.Rall))
        sess.run(self.randomness_var.assign(self.e))
        s = sess.run(self.summ, feed_dict={self.ip: np.reshape(state_current_exp, (self.batch_size, self.ip_size)),
                                           self.nextQ: targetQBatch})
        self.writer.add_summary(s, self.train_step)

    def save_weights(self, sess):
        save_path = self.saver.save(sess, "Saved_Weights/model.ckpt")
        print("Model saved successfully")

    def restore_weights(self, sess):
        self.saver.restore(sess, "Saved_Weights/model.ckpt")
        print("Model Restore Success")

    def save_experience(self, state_current, action, r, state_next):
        self.circular_queue.append(np.asarray([state_current, action, r, state_next]))
        file = open('saved_experience.csv', 'a')
        comma = ','
        for x in np.nditer(state_current):
            file.write(str(x))
            file.write(comma)
        file.write(str(action))
        file.write(comma)
        file.write(str(r))
        for x in np.nditer(state_next):
            file.write(comma)
            file.write(str(x))
        file.write(comma)
        file.write('\n')
        file.close()

    def load_experience_to_buffer(self):
        file = open('saved_experience.csv', 'r')
        for _ in range(self.queue_len):
            f = file.readline().split(',')
            state_current = np.asarray(f[0:4])
            action = f[4]
            r = f[5]
            state_next = np.asarray(f[6:10])
            self.circular_queue.append(np.asarray([state_current, action, r, state_next]))


class Serial_comm():
    ''' Serial Communication 	with the MCU'''

    def __init__(self, baud_rate, port, num_char):
        self.baud_rate = baud_rate
        self.port = port
        self.num_char = num_char

    def serial_read(self):
        ser = serial.Serial(self.port, self.baud_rate)
        while True:
            ser.reset_input_buffer()
            x = ser.readline()
            if (len(x) == self.num_char):
                break
        ser.close()
        return x[0:self.num_char - 2]

    def serial_write(self, x):
        ser = serial.Serial(self.port, self.baud_rate)
        ser.reset_output_buffer()
        if (x == 10):
            x = 26
        ser.write(bytes([x]))
        ser.close()


# Dense -  onehot conversion for 2 states, 2 actions
# def dense_to_onehot(prev,current):
# 	arr= np.empty([1,8],dtype=int)
# 	arr[0]=(prev&0b00001000)>>3
# 	arr[0,1]=(prev&0b00000100)>>2
# 	arr[0,2]=(prev&0b00000010)>>1
# 	arr[0,3]=(prev&0b00000001)
# 	arr[0,4]=(current&0b00001000)>>3
# 	arr[0,5]=(current&0b00000100)>>2
# 	arr[0,6]=(current&0b00000010)>>1
# 	arr[0,7]=(current&0b00000001)
# 	return arr

def dense_to_onehot(prev, current):
    arr = np.empty([1, 4], dtype=int)
    # arr[0]=(prev&0b00001000)>>3
    # arr[0,1]=(prev&0b00000100)>>2
    # arr[0,2]=(prev&0b00000010)>>1
    # arr[0,3]=(prev&0b00000001)
    arr[0] = (current & 0b00001000) >> 3
    arr[0, 1] = (current & 0b00000100) >> 2
    arr[0, 2] = (current & 0b00000010) >> 1
    arr[0, 3] = (current & 0b00000001)
    return arr


def onehot_to_dense(act):
    return np.argmax(act)


def train_on_quadbot():
    print('Making TF Model')
    Q1 = QNetwork()  # pass parameters: ip_size, op_size, hidden_size, lr, y, e
    S1 = Serial_comm(115200, '/dev/ttyACM0', 6)
    with tf.Session() as sess:
        # do not initialize always/ write code for restoring weights
        Q1.init_network(sess)
        if restore_model == 1:
            Q1.restore_weights(sess)
        else:
            sess.run(Q1.init_variables)
        # Receive initial state from MCU #prev, current, vel, done
        input_char = S1.serial_read()
        state_current = dense_to_onehot(input_char[0], input_char[1])
        if load_experience_to_buffer:
            Q1.load_experience_to_buffer()
        else:
            # Collect experience to initially fill queue
            for _ in range(Q1.queue_len):
                if input_char[3] == 0:  # terminate condition
                    state_current, input_char = Q1.collect_experiece(state_current, sess, S1)
                else:
                    break
        # Collect experience and replay every time step
        while input_char[3] == 0:
            state_current, input_char = Q1.collect_experiece(state_current, sess, S1)
            r_int = random.randint(0, Q1.queue_len - Q1.batch_size - 1)
            experience = np.asarray(list(itertools.islice(Q1.circular_queue, r_int, r_int + Q1.batch_size)))
            tupled_experience = [[] for _ in range(4)]
            for frame in experience:
                for i, elem in enumerate(frame):
                    tupled_experience[i].append(elem)
            state_current_exp = np.asarray(tupled_experience[0])
            action_exp = np.asarray(tupled_experience[1])
            reward_exp = np.asarray(tupled_experience[2])
            state_next_exp = np.asarray(tupled_experience[3])
            # print(state_current_exp,action_exp,reward_exp,state_next_exp)
            Q1.experience_replay(state_current_exp, action_exp, reward_exp, state_next_exp, sess)
        Q1.save_weights(sess)
        print(Q1.Rall)


restore_model = 0
load_experience_to_buffer = 1
train_on_quadbot()
