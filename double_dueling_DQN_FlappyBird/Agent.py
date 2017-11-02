import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os

#Implementing the network itself
class FrozenQNetwork():
    def __init__(self,act_num):
        with tf.variable_scope('Frozen_Q_Net'):
            self.w = 84
            self.h = 84
            self.d = 4

            xavier_init_conv2d = tf.contrib.layers.xavier_initializer_conv2d()
            xavier_init = tf.contrib.layers.xavier_initializer()

            self.flattened_batch_fi = tf.placeholder(shape=[None, 28224], dtype=tf.float32)#[bs,28224]
            self.batch_fi = tf.reshape(self.flattened_batch_fi, shape=[-1, self.w, self.h, self.d])#[bs,w=84,h=84,d=4]

            ## conv1 layer ##
            self.W_conv1 = tf.Variable(xavier_init_conv2d([8, 8, 4, 32]))
            self.b_conv1 = tf.Variable(tf.constant(0., shape=[32]))
            self.conv1 = tf.nn.conv2d(self.batch_fi, self.W_conv1, strides=[1, 4, 4, 1], padding='SAME')
            self.h_conv1 = tf.nn.relu(self.conv1 + self.b_conv1)#[bs,21,21,32]
            ## conv2 layer ##
            self.W_conv2 = tf.Variable(xavier_init_conv2d([4, 4, 32, 64]))
            self.b_conv2 = tf.Variable(tf.constant(0., shape=[64]))
            self.conv2 = tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, 2, 2, 1], padding='SAME')
            self.h_conv2 = tf.nn.relu(self.conv2 + self.b_conv2)#[bs,11,11,64]
            ## conv3 layer ##
            self.W_conv3 = tf.Variable(xavier_init_conv2d([3, 3, 64, 64]))
            self.b_conv3 = tf.Variable(tf.constant(0., shape=[64]))
            self.conv3 = tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            self. h_conv3 = tf.nn.relu(self.conv3 + self.b_conv3)#[bs,11,11,64]

            self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 7744])#[bs, 7744]
            ## fc4 layer ##
            self.W_fc4 = tf.Variable(xavier_init([7744, 1024]))
            self.b_fc4 = tf.Variable(tf.constant(0., shape=[1024]))
            self.h_fc4 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc4) + self.b_fc4)#[bs, 1024]
            ## fc5 layer ##
            self.W_fc5 = tf.Variable(xavier_init([1024, 64]))
            self.b_fc5 = tf.Variable(tf.constant(0., shape=[64]))
            self.h_fc5 = tf.matmul(self.h_fc4, self.W_fc5) + self.b_fc5#[bs, 64]

            self.streamA, self.streamV = tf.split(self.h_fc5, 2, 1)#[bs, 32],#[bs, 32]
            xavier_init = tf.contrib.layers.xavier_initializer()
            self.AW = tf.Variable(xavier_init([32,act_num])) #[32,act_num]
            self.VW = tf.Variable(xavier_init([32,1])) #[32,1]
            self.Advantage = tf.matmul(self.streamA,self.AW) #[bs,act_num]
            self.Value = tf.matmul(self.streamV,self.VW) #[bs,1]
            # Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))#[bs,act_num]


class TrainingQNetwork():
    def __init__(self, act_num):
        with tf.variable_scope('Train_Q_Net'):
            self.w = 84
            self.h = 84
            self.d = 4

            xavier_init_conv2d = tf.contrib.layers.xavier_initializer_conv2d()
            xavier_init = tf.contrib.layers.xavier_initializer()

            self.flattened_batch_fi = tf.placeholder(shape=[None, 28224], dtype=tf.float32)#[bs,28224]
            self.batch_fi = tf.reshape(self.flattened_batch_fi, shape=[-1, self.w, self.h, self.d])#[bs,w=84,h=84,d=4]
            ## conv1 layer ##
            self.W_conv1 = tf.Variable(xavier_init_conv2d([8, 8, 4, 32]))
            self.b_conv1 = tf.Variable(tf.constant(0., shape=[32]))
            self.conv1 = tf.nn.conv2d(self.batch_fi, self.W_conv1, strides=[1, 4, 4, 1], padding='SAME')
            self.h_conv1 = tf.nn.relu(self.conv1 + self.b_conv1)  # [bs,21,21,32]
            ## conv2 layer ##
            self.W_conv2 = tf.Variable(xavier_init_conv2d([4, 4, 32, 64]))
            self.b_conv2 = tf.Variable(tf.constant(0., shape=[64]))
            self.conv2 = tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, 2, 2, 1], padding='SAME')
            self.h_conv2 = tf.nn.relu(self.conv2 + self.b_conv2)  # [bs,11,11,64]
            ## conv3 layer ##
            self.W_conv3 = tf.Variable(xavier_init_conv2d([3, 3, 64, 64]))
            self.b_conv3 = tf.Variable(tf.constant(0., shape=[64]))
            self.conv3 = tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            self.h_conv3 = tf.nn.relu(self.conv3 + self.b_conv3)  # [bs,11,11,64]

            self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 7744])  # [bs, 7744]
            ## fc4 layer ##
            self.W_fc4 = tf.Variable(xavier_init([7744, 1024]))
            self.b_fc4 = tf.Variable(tf.constant(0., shape=[1024]))
            self.h_fc4 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc4) + self.b_fc4)  # [bs, 1024]
            ## fc5 layer ##
            self.W_fc5 = tf.Variable(xavier_init([1024, 64]))
            self.b_fc5 = tf.Variable(tf.constant(0., shape=[64]))
            self.h_fc5 = tf.matmul(self.h_fc4, self.W_fc5) + self.b_fc5  # [bs, 64]

            self.streamA, self.streamV = tf.split(self.h_fc5, 2, 1)  # [bs, 32],#[bs, 32]
            xavier_init = tf.contrib.layers.xavier_initializer()
            self.AW = tf.Variable(xavier_init([32, act_num]))  # [32,act_num]
            self.VW = tf.Variable(xavier_init([32, 1]))  # [32,1]
            self.Advantage = tf.matmul(self.streamA, self.AW)  # [bs,act_num]
            self.Value = tf.matmul(self.streamV, self.VW)  # [bs,1]
            # Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))  # [bs,act_num]
            self.predict = tf.argmax(self.Qout, 1)  # [bs]

            # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)  # bs
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)  # bs
            self.actions_onehot = tf.one_hot(self.actions, act_num, dtype=tf.float32)  #[bs,act_num]

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)  # bs

            self.td_error = tf.square(self.targetQ - self.Q)  # bs
            self.loss = tf.reduce_mean(self.td_error)  # 1
            self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self.trainer.minimize(self.loss)


class ExperienceMemory():
    def __init__(self, memory_size=10000):
        self.memory = []
        self.memory_size = memory_size

    def add(self, fi, a, r, Nfi, done):
        flattened_fi = np.reshape(fi, [28224])
        flattened_Nfi = np.reshape(Nfi, [28224])
        experience = np.reshape(np.array([flattened_fi,a,r,flattened_Nfi,done]),[1,5])
        if len(self.memory) + len(experience) >= self.memory_size:
            self.memory[0:(len(experience) + len(self.memory)) - self.memory_size] = []
        self.memory.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.memory, size)), [size, 5])


class Model():
    def __init__(self,path="./model"):
        self.saver = tf.train.Saver()
        self.path = path

    def store(self,sess,episode):

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.saver.save(sess, self.path + '/model.cptk', global_step=episode)

    def restore(self,sess):
        if not os.path.exists(self.path):
            print('no Model')
        else:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(self.path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)


class Chooser():
    def __init__(self,act_num,num_pre_train=1000):
        self.act_num = act_num
        self.initial_exploration = 1.  # 1. #initial
        self.final_exploration = 0.1
        self.e = self.initial_exploration
        self.final_exploration_frame = 50000
        self.num_pre_train=num_pre_train

    def choose_action(self,sess,training_net,fi,total_step):
        flattened_fi = np.reshape(fi, [28224])
        if total_step>self.num_pre_train:
            if (self.e > self.final_exploration):
                self.e -= (self.initial_exploration - self.final_exploration) / self.final_exploration_frame
            else:
                self.e = self.final_exploration

        if np.random.rand(1) < self.e or total_step < self.num_pre_train:
            a = np.random.randint(0, self.act_num)
            print('rand_act')
        else:
            a,act_value = sess.run([training_net.predict,training_net.Qout], feed_dict={training_net.flattened_batch_fi: [flattened_fi]})
            print('act_value:',['%.6f' %i for i in act_value[0]],'e:',self.e)
        return a,self.e


class Updater():
    def __init__(self):
        ur = 0.001
        frozen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Train_Q_Net')
        training_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Frozen_Q_Net')
        self.init_op_holder = []
        for idx, var in enumerate(frozen_params):
            op = frozen_params[idx].assign(training_params[idx].value())
            self.init_op_holder.append(op)

        self.op_holder = []
        for idx, var in enumerate(frozen_params):
            op = frozen_params[idx].assign((1 - ur) * var.value() + ur * training_params[idx].value())
            self.op_holder.append(op)

    def init_frozen_net(self,sess):
        for op in self.init_op_holder:
            sess.run(op)

    def update_frozen_net(self,sess):
        for op in self.op_holder:
            sess.run(op)


def train_traing_net(sess,training_net,frozen_net,Memory,batch_size=32,gamma=0.99):
    trainBatch = Memory.sample(batch_size)
    # Below we perform the Double-DQN update to the target Q-values
    Q_by_frozen_net = sess.run(frozen_net.Qout, feed_dict={frozen_net.flattened_batch_fi: np.vstack(trainBatch[:, 3])})#[bs,act_num]
    argmax = sess.run(training_net.predict,feed_dict={training_net.flattened_batch_fi: np.vstack(trainBatch[:, 3])}) # [bs]
    end_multiplier = -(trainBatch[:, 4] - 1)#array([bs])
    doubleQ = Q_by_frozen_net[range(batch_size), argmax]#array([bs])
    targetQ = trainBatch[:, 2] + (gamma * doubleQ * end_multiplier) #array([bs])
    # Update the network with our target values.
    _,loss = sess.run([training_net.updateModel, training_net.loss],
                 feed_dict={training_net.flattened_batch_fi: np.vstack(trainBatch[:, 0]),
                            training_net.targetQ: targetQ,
                            training_net.actions: trainBatch[:, 1]})
    return loss







