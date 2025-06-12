import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
import time

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")# 创建第一个GPU设备对象
# device
# print(device.)


class DeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)# 采用Adam优化器
        self.to(device)# 将模型转移到GPU上

    def forward(self, state):# 三层全连接层接两层relu函数激活
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))

        q = self.q(x)
        return q

    def save_checkpoint(self, checkpoint_file):# 保存模型训练参数
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):# 加载模型训练参数
        self.load_state_dict(T.load(checkpoint_file))


class DQN:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.01, eps_dec=5e-4,
                 max_size=1000000, batch_size=256):
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.action_space = [i for i in range(action_dim)]
        self.checkpoint_dir = ckpt_dir
        self.loss = 0

        self.q_eval = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                   fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=fc1_dim, fc2_dim=fc2_dim)

        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):# 更新网络参数
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, isTrain=True, mask=None):# 选择智能体行动
        
        state = T.tensor(observation, dtype=T.float).to(device)# 将张量转移至GPU
        if mask is not None:
            mask = T.tensor(mask, dtype=T.bool).to(device)

        actions = self.q_eval.forward(state)
        # print(actions.size(),"!!!!!!!!!!!!!")
        if mask is not None:
            masked_actions = actions.masked_fill(mask == 0, -100000000)
            action = T.argmax(masked_actions).item()# 选出神经网络给出的当前state下得分最高的action
            # print(masked_actions)
        else:
            action = T.argmax(actions).item()# 选出神经网络给出的当前state下得分最高的action
        
        if (np.random.random() < self.epsilon) and isTrain:# 有概率随机选取action
            if mask is not None:
                valid_actions = mask.nonzero().squeeze().cpu().numpy()
                action = np.random.choice(valid_actions)
            else:
                action = np.random.choice(self.action_space)
            #action = np.random.choice(valid_actions)

        return action

    def learn(self):# 智能体自学习
        if not self.memory.ready():
            return

        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()# 从经验回放池中抽取样例
        batch_idx = np.arange(self.batch_size)

        states_tensor = T.tensor(states, dtype=T.float).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)
        
        # print("terminals_tensor",terminals_tensor)

        with T.no_grad():
            q_ = self.q_eval.forward(next_states_tensor)
            action = T.argmax(q_, dim=-1) # 选出目标网络给出的下一state集合下得分最高的action集合
            q_ = self.q_target.forward(next_states_tensor)
            # q_ = self.q_eval.forward(next_states_tensor)
            # print("q_",q_)
            q_[terminals_tensor] = 0.0
            # target = rewards_tensor + self.gamma * T.max(q_, dim=-1)[0]
            target = rewards_tensor + self.gamma * q_[batch_idx, action]
        q = self.q_eval.forward(states_tensor)[batch_idx, actions]
        
        # time.sleep(1000)

        loss = F.mse_loss(q, target.detach())# 计算损失反向传播
        self.loss = loss
        # print("loss:", loss)
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self, episode):# 保存模型参数
        self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/DQN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/DQN_Q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')

    def load_models(self, episode):# 加载模型参数
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/DQN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/DQN_Q_target_{}.pth'.format(episode))
        print('Loading Q_target network successfully!')



