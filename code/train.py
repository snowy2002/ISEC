import os
import copy
import numpy as np
import pandas as pd
import argparse
import time
from DQN import DQN
from utils import plot_learning_curve, create_directory
from environment import Environment
import torch
envpath = '.'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

parser = argparse.ArgumentParser()# 创建一个 ArgumentParser 对象
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/DQN/')# 添加参数并设置参数的全局默认值
parser.add_argument('--max_episodes', type=int, default=100) # 训练轮次
args = parser.parse_args() # 通过 parse_args() 方法解析参数。


def main():
    with open('code/data/base.in', 'r') as file:
        content = file.read()  # 读取整个文件内容
        base = list(map(float, content.split()))  # 处理浮点数
    fm = base[0] # 主频
    slot = int(base[1])  # 时隙
    num_m = int(base[2]) # 服务器数量
    rm = base[3]    # 传输速率

    # 设置DAG参数
    G = {}
    with open('code/data/Graph.in', 'r') as file:
        content = file.read()  # 读取整个文件内容
        Graph = list(map(float, content.split()))  # 处理浮点数
    vertex_num = int(Graph[0] + 1)
    edge_num = int(Graph[1])
    de = [0 for _ in range(vertex_num)]
    # DE = copy.deepcopy(de)
    for i in range(vertex_num):
        G[i] = []
    for i in range(2, len(Graph), 2):
        u = int(Graph[i])
        v = int(Graph[i+1])
        G[u].append(v)
        de[v] += 1
    
    with open('code/data/task.in', 'r') as file:
        content = file.read()  # 读取整个文件内容
        task = list(map(float, content.split()))  # 处理浮点数
    cost = task

    runtime = [0 for _ in range(num_m)]
    task = [0 for _ in range(vertex_num)]
    load_task = [-1 for _ in range(num_m)]
    deadline = [1000000 for _ in range(vertex_num)]

    env = Environment(slot=slot, time=0, fm=fm, runtime=runtime, num_m=num_m, 
                     load_task=load_task, rm=rm, task=task, G=G, de=de, alpha=2.2, beta=0.55,
                     vertex_num=vertex_num, edge_num=edge_num, cost=cost, deadline=deadline) 
    # alpha 1.0 - 1.6 
    # beta 0.4 0.6 0.8 1.0
    
    start = time.time() 
    agent = DQN(alpha=0.0003, state_dim=num_m+vertex_num, action_dim=vertex_num+1,
                fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.4, tau=0.005, epsilon=1,
                eps_end=0.05, eps_dec=1e-5, max_size=1000000, batch_size=256)# 创建智能体

    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])# 创建DQN下属文件夹'Q_eval', 'Q_target'
    total_rewards, avg_rewards, eps_history = [], [], []
    
    ave_time = 0 # 平均任务完成时间
    lc_rate = 0 # 任务完成率
    cnt_task = 0

    for episode in range(args.max_episodes):# 进行episode批次交互
        total_reward = 0
        done = False
        DE = copy.deepcopy(de)
        observation = env.reset(G, DE)# 重置环境
        while not done: # 当一局游戏未结束
            ob = np.array(observation)
            mask = copy.deepcopy(env.task)
            mask = mask + [1]
            mask = [0 if x == 2 else x for x in mask]
            action = agent.choose_action(ob, isTrain=True,mask=mask)# 智能体根据观察到的环境做出行动
            observation_, reward, done = env.step(action) # 智能体做出行动后环境给出的反馈
            cnt_ = env.cnt
            cnt_task = cnt_
            ob_ = np.array(observation_)
            agent.remember(ob, action, reward, ob_, done) # 智能体记录此前的环境、采取的动作以及环境给出的反馈信息
            agent.learn() # 智能体进行自主学习
            total_reward += reward # 将此次奖励计入总得分
            observation = observation_ # 观察到的环境发生变化

        ave_time = env.ave_time
        lc_rate = cnt_task / vertex_num

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:]) # 最近100次的总得分的均值
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
        print('EP:{} reward:{} avg_reward:{} epsilon:{}'.
            format(episode + 1, total_reward, avg_reward, agent.epsilon))

        if (episode + 1) % 50 == 0:
            agent.save_models(episode + 1)
    end=time.time()

    print("运行时间(s):",end - start)
    print("平均任务完成时间(ms):", ave_time)
    print("任务完成率：", lc_rate)

if __name__ == '__main__':
    main()
