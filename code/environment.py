from heapq import *
from math import *
from cyaron import *
import numpy as np
import random

class Environment:
    def __init__(self, slot, time, fm, runtime, 
                 load_task, rm, task, G, de, 
                 vertex_num, edge_num, cost, 
                 deadline, alpha = 1.2, num_m = 10, beta = 0.5) -> None:
        self.slot = slot # 时隙
        self.time = time # 当前时间(ms)
        self.fm = fm # 服务器的主频
        self.num_m = num_m # 服务器数量
        self.runtime = runtime # 每个服务器的运行时间
        self.load_task = load_task # 当前装载的任务
        self.rm = rm # 每个服务器的传输时延
        self.task = task # 任务列表（0：未装载且目前不可装载   1：未装载但目前可以装载   2：已经装载）
        self.vertex_num = vertex_num # 任务数量
        self.edge_num = edge_num # 边的数量
        self.G = G # 图
        self.de = de # 每个任务节点的入度，拓扑排序需要
        self.cost = cost # 每个任务节点的计算成本
        self.deadline = deadline # 每个任务节点的截止时间
        self.reward = 0 # 当前奖励
        self.state = [] # 当前状态
        self.done = False
        self.alpha = alpha # 奖励系数
        self.beta = beta # 奖励系数
        # self.score = sum(self.cost) / self.fm * 200

        self.run_time = 0 # 运行时间
        self.RPD = 0 # 运行时间百分比
        self.cnt = 0 # 任务指标完成率
        self.ave_time = 0 # 任务的平均完成时间
        self.ddl = sum(self.cost) / self.fm * 70 * self.beta # 任务截止时间


    def reset(self, G, de):
        self.G = G
        self.de = de
        self.time = 0
        self.reward = 0
        self.task = [0 for _ in range(self.vertex_num)]
        self.runtime = [0 for _ in range(self.num_m)]
        self.task[0] = 1
        self.state = self.task + self.runtime 
        self.load_task = [-1 for _ in range(self.num_m)] # 表示服务器装载任务为空
        self.load_task[0] = 0
        self.done = False


        self.run_time = 0 # 运行时间
        self.RPD = 0 # 运行时间百分比
        self.cnt = 0 # 任务完成个数
        self.ave_time = 0 # 任务的平均完成时间

        return self.state
    

    def step(self, action : int): # 0 ~ vertex_num : 表示将 action 这个节点装载到服务器上
        if(self.done == True):
            return self.state, 0, self.done
        self.time = self.time + self.slot # 每个时隙，时间增加
        id = -1 # 找到第一个可用的服务器
        for i in range(self.num_m): 
            self.runtime[i] = max(self.runtime[i] - self.slot, 0) # 每个服务器的运行1个时隙
            if self.runtime[i] == 0: # 如果服务器运行时间到，则将服务器上的任务卸载
                if(self.load_task[i] != -1): # 如果服务器上有任务
                    for j in self.G[self.load_task[i]]: # 更新入度
                        self.de[j] -= 1
                        if self.de[j] == 0: # 如果入度为0，则将任务标记为可装载
                            self.task[j] = 1
                    self.load_task[i] = -1 # 将服务器上的任务卸载
                id = i # 找到第一个可用的服务器
        # print("id:",id)
        if id == -1: # 如果没有可用的服务器，则此次装载失败
            return self.state, -0.1, self.done # 返回 当前状态，奖励，是否结束
        if action == self.vertex_num: # 如果没有可装载的任务，则此次装载失败
            return self.state, 0, self.done # 返回 当前状态，奖励，是否结束
        if(self.task[action] != 1):
            return self.state, 0, self.done # 返回 当前状态，奖励，是否结束
        self.load_task[id] = action # 将任务装载到服务器上
        self.runtime[id] += self.cost[action] / self.fm * 1000 # 将任务装载到服务器上(ms)
        self.ave_time += (self.time + self.runtime[id]) / (self.vertex_num - 1) # 计算任务的平均完成时间
        if(self.time + self.runtime[id] <= self.ddl):
            self.cnt += 1 # 计算任务完成个数
        self.task[action] = 2 # 将任务标记为已经装载
        
        if(all(x == 2 for x in self.task)): # 判断是否所有任务都装载完毕
            self.done = True
            self.reward = sum(self.cost) / self.fm * 100 * self.alpha - self.time # 奖励为1
        self.state = self.task + self.runtime  # 更新状态
        return self.state, self.reward, self.done # 返回 当前状态，奖励，是否结束
    
if __name__ == '__main__': 
    # 打开文件
    # 读取所有行到一个列表
    # 读取文件并转换为浮点数列表


    # 设置基本参数
    with open('base.in', 'r') as file:
        content = file.read()  # 读取整个文件内容
        base = list(map(float, content.split()))  # 处理浮点数
    fm = base[0] # 主频
    slot = int(base[1])  # 时隙
    num_m = int(base[2]) # 服务器数量
    rm = base[3]    # 传输速率

    # 设置DAG参数
    G = {}
    with open('Graph.in', 'r') as file:
        content = file.read()  # 读取整个文件内容
        Graph = list(map(float, content.split()))  # 处理浮点数
    vertex_num = int(Graph[0] + 1)
    edge_num = int(Graph[1])
    de = [0 for _ in range(vertex_num)]
    for i in range(vertex_num):
        G[i] = []
    for i in range(2, len(Graph), 2):
        u = int(Graph[i])
        v = int(Graph[i+1])
        G[u].append(v)
        de[v] += 1
    
    with open('task.in', 'r') as file:
        content = file.read()  # 读取整个文件内容
        task = list(map(float, content.split()))  # 处理浮点数
    cost = task

    # print(f"fm: {fm} slot: {slot} num_m: {num_m} rm: {rm}")
    # print(f"vertex_num: {vertex_num} edge_num: {edge_num}")
    # print(G)
    # print(de)
    # print(cost)

    runtime = [0 for _ in range(num_m)]
    task = [0 for _ in range(vertex_num)]
    load_task = [-1 for _ in range(num_m)]
    deadline = [1000000 for _ in range(vertex_num)]

    env = Environment(slot=slot, time=0, fm=fm, runtime=runtime, num_m=num_m, 
                     load_task=load_task, rm=rm, task=task, G=G, de=de, alpha=1.2, 
                     vertex_num=vertex_num, edge_num=edge_num, cost=cost, deadline=deadline)
    env.reset(G, de)

    cnt = 0
    f = 0
    while cnt < 10000000:
        cnt += 1
        _ = randint(0, vertex_num)
        if(_ == vertex_num):
            state, reward, done = env.step(_)
            f += reward
            # print(env.task)
            # print(reward, done, f)
            if done == 1: 
                break
        elif(env.task[_] != 1):
            cnt -= 1
            continue
        else:
            state, reward, done = env.step(_)
            f += reward
            # print(reward, done, f)
            if done == 1: 
                print("reward:",f)
                break