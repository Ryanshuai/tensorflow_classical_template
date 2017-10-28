"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
"""

import numpy as np
import tensorflow as tf


LOAD_MODEL = 'model/model1' #load model from here
SAVE_MODEL = 'model/model0/model.ckpt' #save model to here

# Deep Q Network off-policy
class Escaper_Agent:
    def __init__(self):
        self.n_actions = 2 #up down left right
        self.n_robot = 1

        self.batch_size = 32
        self.memory_size = 100000  # replay memory size
        self.history_length = 4 #agent history length
        self.frozen_network_update_frequency = 1000 #frozen network update frequency
        self.gamma = 0.99  # discount factor
        self.action_repeat = 4
        self.update_frequency = 4
        self.initial_exploration = 1. #1. #initial
        self.final_exploration = 0.1
        self.exploration = self.initial_exploration 
        self.final_exploration_frame = 100000
        self.replay_start_size = 1000
        #used by RMSProp
        self.lr = 0.00025
        self.min_squared_gradient = 0.01
        #counter and printer
        self.train_step_counter = 1  # total learning step
        self.memory_counter = 1
        self.update_counter = 0
        self.outloss = 0
        self.actions_value = [0,0]
        # w*h*m, this is the parameter of memory
        self.w = 84 #observation_w
        self.h = 84 #observation_h
        self.m = 4 #agent_history_length
        self.memory = {'fi': np.zeros(shape=[self.memory_size, self.w, self.h, self.m], dtype=np.uint8),#0-255
                  'a': np.zeros(shape=[self.memory_size, ], dtype=np.int8),
                  'r': np.zeros(shape=[self.memory_size, ], dtype=np.int8),
                  'Nfi': np.zeros(shape=[self.memory_size, self.w, self.h, self.m], dtype=np.uint8),
                  'done': np.zeros(shape=[self.memory_size, ], dtype=np.uint8)}

        self._build_net()# consist of [frozen_net, training_net]
        self.saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        
        # ------------------ load model ------------------
        ckpt = tf.train.get_checkpoint_state(LOAD_MODEL)
        if ckpt and ckpt.model_checkpoint_path:
            print('loading_model')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)


    def _build_net(self):
        def build_layers(s, collection_names):
            ## conv1 layer ##
            W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01), collections=collection_names)
            b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]), collections=collection_names)
            conv1 = tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding='SAME')
            h_conv1 = tf.nn.relu(conv1 + b_conv1)
            ## conv2 layer ##
            W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01), collections=collection_names)
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]), collections=collection_names)
            conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME')
            h_conv2 = tf.nn.relu(conv2 + b_conv2)
            ## conv3 layer ##
            W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01), collections=collection_names)
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]), collections=collection_names)
            conv3 = tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            h_conv3 = tf.nn.relu(conv3 + b_conv3)
            # [n_samples, 11, 11, 64] ->> [n_samples, 7744]
            h_conv3_flat = tf.reshape(h_conv3, [-1, 7744])
            ## fc4 layer ##
            W_fc4 = tf.Variable(tf.truncated_normal([7744, 512], stddev=0.01), collections=collection_names)
            b_fc4 = tf.Variable(tf.constant(0.01, shape=[512]), collections=collection_names)
            h_fc4 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc4) + b_fc4)
            ## fc5 layer ##
            W_fc5 = tf.Variable(tf.truncated_normal([512, self.n_actions*self.n_robot], stddev=0.01), collections=collection_names)
            b_fc5 = tf.Variable(tf.constant(0.01, shape=[self.n_actions*self.n_robot]), collections=collection_names)
            h_fc5 = tf.matmul(h_fc4, W_fc5) + b_fc5
            return h_fc5

        # ------------------ build frozen_net ------------------
        self.batch_Nfi = tf.placeholder(tf.float32, shape=[None, self.w, self.h, self.m]) / 255  # input Next State
        col_frozen_net = ['frozen_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        self.q_Nfi_from_frozen_net = build_layers(self.batch_Nfi, col_frozen_net)

        # ------------------ build training_net ------------------
        self.batch_fi = tf.placeholder(tf.float32, shape=[None, self.w, self.h, self.m]) / 255
        col_train_net = ['training_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        self.q_fi_from_training_net = build_layers(self.batch_fi, col_train_net)

        self.batch_a = tf.placeholder(tf.int32, [None, ])  # input Action
        a_one_hot = tf.one_hot(self.batch_a, depth=self.n_actions, dtype=tf.float32)
        self.q_fi_from_training_net_with_action = tf.reduce_sum(self.q_fi_from_training_net * a_one_hot, axis=1)  #dot product

        self.q_fi_suppose_by_frozen_net = tf.placeholder(tf.float32, shape=[None, ])
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_fi_suppose_by_frozen_net, self.q_fi_from_training_net_with_action))
        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


    def store_transition(self, fi, a, r, Nfi, done):
        index = self.memory_counter % self.memory_size
        self.memory['fi'][index] = fi
        self.memory['a'][index] = a
        self.memory['r'][index] = r
        self.memory['Nfi'][index] = Nfi
        self.memory['done'][index] = done
        self.memory_counter += 1


    def choose_action(self, observation):
        observation = observation[np.newaxis, :]#[84,84,4] - > [1,84,84,4]
        if np.random.uniform() < self.exploration: #exploration
            action = np.random.randint(0, self.n_actions)
        else:
            self.actions_value = self.sess.run(self.q_fi_from_training_net, feed_dict={self.batch_fi: observation})[0]
            action = np.argmax(self.actions_value)
        return action


    def learn(self):
        if self.train_step_counter % self.frozen_network_update_frequency == 0:
            t_params = tf.get_collection('frozen_net_params')
            e_params = tf.get_collection('training_net_params')
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
            self.saver.save(self.sess, SAVE_MODEL, global_step=self.train_step_counter)
            self.update_counter += 1
            
        if(self.exploration > self.final_exploration):
            self.exploration -= ( self.initial_exploration - self.final_exploration) / self.final_exploration_frame
        else:
            self.exploration = self.final_exploration

        
        if self.memory_counter > self.replay_start_size:
            # sample batch memory from all memory
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            #get q_fi_suppose_by_frozen_net
            q_Nfi_from_frozen_net = self.sess.run(self.q_Nfi_from_frozen_net, feed_dict={self.batch_Nfi: self.memory['Nfi'][sample_index]})
            end_multiplier = -(self.memory['done'][sample_index] - 1)
            q_fi_suppose_by_frozen_net = self.memory['r'][sample_index] + self.gamma * np.max(q_Nfi_from_frozen_net, axis=1) * end_multiplier
            # train training_network by q_fi_suppose_by_frozen_net
            _, self.outloss = self.sess.run([self._train_op, self.loss],
                feed_dict={self.q_fi_suppose_by_frozen_net : q_fi_suppose_by_frozen_net,
                            self.batch_fi: self.memory['fi'][sample_index],
                            self.batch_a: self.memory['a'][sample_index]})
            self.train_step_counter += 1
                            
        return self.actions_value[0],self.actions_value[1], self.exploration,self.train_step_counter, self.update_counter,self.outloss
