#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
from gd_deep_q_learning import geometry
import random
import numpy as np
import tkinter
from collections import deque
import keyboard



GAME = 'dash'  # 로그 파일을 위한 재생되어지고 있던 게임
ACTIONS = 2  # 유효한 액션 의 수
GAMMA = 0.99  # 이전 관찰들로부터의 감소율
OBSERVE = 1000.  # 훈련 전에 관찰하기 위한 타임 스텝
EXPLORE = 3000000.  # 입실론 값을 강화하기 위한 최소 프레임수
FINAL_EPSILON = 0.005 # 입실론의 최종값
INITIAL_EPSILON = 0.1  # 입실론 값의 시작값
REPLAY_MEMORY = 1000000  # 기억해야 할 이전 변화의 수
BATCH = 32  # 미니배치의 크기
FRAME_PER_ACTION = 1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def create_network():
    # 네트워크 가중치
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # 입력 층
    s = tf.placeholder("float", [None, 80, 80, 4])

    # 숨겨진 층들
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # 판독 값 층
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1


def train_network(s, readout, sess):
    # 비용 함수를 정의한다
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # 에뮬레이터와 통신하기 위해 게임 스테이트를 초기화한다
    G = geometry.geometry()

    # 리플레이 메모리에 이전 관찰들을 저장한다
    D = deque()

    # 아무것도 안함으로써 첫 상태를 가져오고 이미지를 80*80*4크기로 사전 작업을 한다
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = G.frame_step(do_nothing)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 네트워크를 저장/불러온다
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())


    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    '''root = tkinter.Tk()
    T = tkinter.Text(root, height=12, width=110)
    T.tag_configure('big', font=('Verdana', 40, 'bold'))
    T.pack()
    root.update_idletasks()'''




    # 훈련 시작
    epsilon = INITIAL_EPSILON
    real_epsilon = INITIAL_EPSILON
    t = 0
    dead_t = 0
    notevaluate = True
    q_max = 0
    while True:
        # 욕심내어 액션 입실론을 선택한다
        if keyboard.is_pressed("q"):
            i = input("계속하려면 엔터를 누르세요..")
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= real_epsilon and notevaluate:
                print("----------랜덤 액션----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # 아무것도 안한다

        # 입실론 크기를 줄인다
        if epsilon > FINAL_EPSILON and t > OBSERVE and notevaluate:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 선택된 액션을 실행하고 다음 상태와 보상을 관찰한다
        x_t1, r_t, terminal = G.frame_step(a_t) # r_t : 간 거리..?,
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        if terminal:
            q_max = 0
            dead_t = t

        # D에 변화를 저장한다
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # 관찰이 끝날 시에만 훈련한다
        if t > OBSERVE:
            # 훈련을 진행할 미니배치의 견본을 뽑는다
            minibatch = random.sample(D, BATCH)

            # 배치 변수들을 가져온다
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # 터미널인 경우 보상과 같다
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # 점진적 하강 단계를 수행한다
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch}
            )

        # 오래된 값들을 업데이트한다
        s_t = s_t1
        t += 1

        # 매 10000번마다 경과를 저장한다
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)
            G.active = 5
            notevaluate = False

        if t % 10000 == 5:
            notevaluate = True

        # 출력 정보
        if t <= OBSERVE:
            state = "관찰"
        elif OBSERVE < t <= OBSERVE + EXPLORE:
            state = "탐구"
        else:
            state = "훈련"

        m = np.max(readout_t)

        q_max += m
        os = m / (q_max / (t - dead_t))
        if os < 0.7:
            real_epsilon = epsilon / os ** 4
        elif os < 1:
            real_epsilon = epsilon / os ** 3
        elif os < 1.3:
            real_epsilon = (FINAL_EPSILON - epsilon) / (1.3 - 1) * (os - 1) + epsilon
        else:
            real_epsilon = FINAL_EPSILON

        print("타임스텝", t, "/ 상태", state, "/ 입실론", real_epsilon, "/ 액션", action_index, "/ 보상", r_t, "/ Q 최대값 %e" % m)
        '''T.delete('1.0', tkinter.END)
        m = np.max(readout_t)
        #T.insert(tkinter.END, "MAX Q | " + str(m), 'big')
        if(max1 < m):
            max1 = m
        os = 1 - m / max1
        T.insert(tkinter.END, "MAX Q | " + str(m), 'big')
        T.insert(tkinter.END, "\n죽을 확률 | "+str(round(os * 100, 4)) + "%", 'big')
        T.tag_add("start", "2.8", "2.16")
        if os > 0.75 :
            T.tag_config("start", foreground="red")
        elif os > 0.5 :
            T.tag_config("start", foreground="orange")
        elif os > 0.25:
            T.tag_config("start", foreground="green")
        else:
            T.tag_config("start", foreground="blue")
        T.update()'''


        # 파일에 저장
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''


def play_game():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    s, readout, h_fc1 = create_network()
    train_network(s, readout, sess)


def main():
    play_game()


if __name__ == "__main__":
    main()


