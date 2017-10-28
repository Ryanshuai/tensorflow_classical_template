# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------

from game import wrapped_flappy_bird as game
from escaper_agent import Escaper_Agent
import numpy as np
import cv2

EPISODE_NUMS = 1000000
img_w = 84
img_h = 84
ACTION_UPDATE_FREQUENCY = 4
START_LEARN = 5000

def next_step(action):
    nextObservation = np.zeros(shape=[img_w, img_w, 4], dtype = np.uint8)
    rewardsum = 0
    terminal = False
    for i in range(ACTION_UPDATE_FREQUENCY):
        next_image, reward, terminal = flappyBird.frame_step(action)
        rewardsum += reward
        # terminal = True, flappyBird is inited automatically
        if terminal:
            break
        next_image = cv2.cvtColor(cv2.resize(next_image, (img_w, img_w)), cv2.COLOR_BGR2GRAY)
        nextObservation[:, :,i] = next_image
    return nextObservation, rewardsum, terminal


def playFlappyBird():
    for episode in range(EPISODE_NUMS):
        init_action = np.array([0, 1])  # input_actions[1] == 1: flap the bird
        observation, reward, terminal = next_step(init_action)
        score = 0
        while not terminal:
            action_index = brain.choose_action(observation)
            action = np.zeros(shape = 2)
            action[action_index] = 1
            nextObservation, reward, terminal = next_step(action)
            score += reward
            brain.store_transition(observation, action_index, reward, nextObservation, terminal)
            printer = brain.learn()
            observation = nextObservation
            if not terminal:
                print('DoNth:','%+.4f' % printer[0],'fly:','%+.4f' % printer[1],'e:', '%.4f' % printer[2],
              ' train_step:',printer[3], ' update:', printer[4],' loss:',printer[5])
            else:
                print('DoNth:','%+.4f' % printer[0],'fly:','%+.4f' % printer[1],'e:', '%.4f' % printer[2],
              ' train_step:',printer[3], ' update:', printer[4],' loss:',printer[5])
        print('\t-----  score:',score)

if __name__ == '__main__':
    flappyBird = game.GameState()
    brain = Escaper_Agent()
    playFlappyBird()

