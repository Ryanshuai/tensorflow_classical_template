import Agent
import tensorflow as tf
from collections import Counter
from game import wrapped_flappy_bird as bird
import numpy as np
import cv2

counter = Counter({'total_steps':0,'train_steps':0,'episode':0,'step_in_episode':0,
                   'r_sum_in_episode':0,'loss':0,'exploration':0})
num_episodes = 10*1000
max_step_in_one_episode = 1000000000
update_freq = 4
num_pre_train=1000
save_mode_per_episode = 1000

tf.reset_default_graph()

env = bird.GameState()
training_net = Agent.TrainingQNetwork(act_num=2)
frozen_net = Agent.FrozenQNetwork(act_num=2)
memory = Agent.ExperienceMemory()
model = Agent.Model()
chooser = Agent.Chooser(act_num=2, num_pre_train=num_pre_train)
updater = Agent.Updater()


def next_step(a):
    action = np.zeros(shape=[2, ])
    action[a] = 1
    nextObservation = np.zeros(shape=[84, 84, 4], dtype = np.uint8)
    reward = 0
    reward_sum = 0
    terminal = False
    for i in range(4):
        next_image, reward, terminal = env.frame_step(action)
        reward_sum += reward
        # terminal = True, flappyBird is inited automatically
        if terminal:
            break
        next_image = cv2.cvtColor(cv2.resize(next_image, (84, 84)), cv2.COLOR_BGR2GRAY)
        nextObservation[:, :, i] = next_image
    return nextObservation, reward_sum , terminal


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    model.restore(sess)
    updater.init_frozen_net(sess)
    for episode in range(num_episodes):
        counter.update(('episode',))
        fi, r, done = next_step(0)

        counter['step_in_episode'] = 0
        counter['r_sum_in_episode'] = 0
        while counter['step_in_episode'] < max_step_in_one_episode:
            a,counter['exploration'] = chooser.choose_action(sess,training_net,fi,counter['total_steps'])
            Nfi,r,done = next_step(a)
            counter.update(('total_steps',))
            memory.add(fi,a,r,Nfi,done)
            fi = Nfi
            counter.update(('step_in_episode',))
            counter['r_sum_in_episode'] += r

            if counter['total_steps'] > num_pre_train and counter['total_steps'] % update_freq == 0:
                counter['loss'] = Agent.train_traing_net(sess, training_net, frozen_net, memory)
                print('frozen', sess.run(frozen_net.W_fc5))
                print('training', sess.run(training_net.W_fc5))
                counter.update(('train_steps',))
                updater.update_frozen_net(sess)
                print('frozen', sess.run(frozen_net.W_fc5))
                print('-----------------------------------------train_steps:', counter['train_steps'],'loss:', '%.8f' % counter['loss'])

            if done == True:
                break
        print('---------------------------------------------------------------------------------total_steps:',counter['total_steps'],'episode:',counter['episode'],'exploration:','%.3f'%counter['exploration'],'r_sum_in_episode:',counter['r_sum_in_episode'])

        if counter['episode'] % save_mode_per_episode == 0:
            model.store(sess, counter['episode'])

