import numpy as np
import copy
from collections import deque
import gym
from gym import wrappers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt


import gym
import gym_floodit
# Q関数の定義


class QNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=16):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_action)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        h = F.elu(self.fc3(h))
        y = F.elu(self.fc4(h))
        return y

# リプレイバッファの定義


class ReplayBuffer:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque([], maxlen=memory_size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        batch_indexes = np.random.randint(0, len(self.memory), size=batch_size)
        states = np.array([self.memory[index]['state']
                           for index in batch_indexes])
        next_states = np.array([self.memory[index]['next_state']
                                for index in batch_indexes])
        rewards = np.array([self.memory[index]['reward']
                            for index in batch_indexes])
        actions = np.array([self.memory[index]['action']
                            for index in batch_indexes])
        dones = np.array([self.memory[index]['done']
                          for index in batch_indexes])
        return {'states': states, 'next_states': next_states, 'rewards': rewards, 'actions': actions, 'dones': dones}


class DqnAgent:
    def __init__(self, num_state, num_action, gamma=0.99, lr=0.001, batch_size=32, memory_size=50000):
        self.num_state = num_state
        self.num_action = num_action
        self.gamma = gamma  # 割引率
        self.batch_size = batch_size  # Q関数の更新に用いる遷移の数
        self.qnet = QNetwork(num_state, num_action)
        self.target_qnet = copy.deepcopy(self.qnet)  # ターゲットネットワーク
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_size)

    # Q関数を更新
    def update_q(self):
        batch = self.replay_buffer.sample(self.batch_size)
        q = self.qnet(torch.tensor(batch["states"], dtype=torch.float))
        targetq = copy.deepcopy(q.data.numpy())
        # maxQの計算
        maxq = torch.max(self.target_qnet(torch.tensor(
            batch["next_states"], dtype=torch.float)), dim=1).values
        # 以下のコメント部分は動画と異なっています（補足しました）
        # targetqのなかで，バッチのなかで実際に選択されていた行動 batch["actions"][i] に対応する要素に対して，Q値のターゲットを計算してセット
        # 注意：選択されていない行動のtargetqの値はqと等しいためlossを計算する場合には影響しない
        for i in range(self.batch_size):
            # 終端状態の場合はmaxQを0にしておくと学習が安定します（ヒント：maxq[i] * (not batch["dones"][i])）
            targetq[i, batch["actions"][i]] = batch["rewards"][i] + \
                self.gamma * maxq[i] * (not batch["dones"][i])
        self.optimizer.zero_grad()
        # lossとしてMSEを利用
        loss = nn.MSELoss()(q, torch.tensor(targetq))
        loss.backward()
        self.optimizer.step()
        # ターゲットネットワークのパラメータを更新
        self.target_qnet = copy.deepcopy(self.qnet)

    # Q値が最大の行動を選択
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(
            state, dtype=torch.float).view(-1, self.num_state)
        action = torch.argmax(self.qnet(state_tensor).data).item()
        return action

    # ε-greedyに行動を選択
    def get_action(self, state, episode):
        epsilon = 0.7 * (1/(episode+1))  # ここでは0.5から減衰していくようなεを設定
        if epsilon <= np.random.uniform(0, 1):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.num_action)
        return action


# 各種設定
num_episode = 300  # 学習エピソード数
memory_size = 50000  # replay bufferの大きさ
initial_memory_size = 500  # 最初に貯めるランダムな遷移の数

# ログ
episode_rewards = []
num_average_epidodes = 10

env = gym.make('floodit-v0')

agent = DqnAgent(
    env.observation_space.shape[0], env.action_space.n, memory_size=memory_size)

# 最初にreplay bufferにランダムな行動をしたときのデータを入れる
state = env.reset()
for step in range(initial_memory_size):
    action = env.action_space.sample()  # ランダムに行動を選択
    next_state, reward, done, _ = env.step(action)
    transition = {
        'state': state,
        'next_state': next_state,
        'reward': reward,
        'action': action,
        'done': int(done)
    }
    agent.replay_buffer.append(transition)
    state = env.reset() if done else next_state

for episode in range(num_episode):
    state = env.reset()  # envからは4次元の連続値の観測が返ってくる
    episode_reward = 0
    done = False
    while(not done):
        action = agent.get_action(state, episode)  # 行動を選択
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        transition = {
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'action': action,
            'done': int(done)
        }
        agent.replay_buffer.append(transition)
        agent.update_q()  # Q関数を更新
        state = next_state
        if done:
            break
    episode_rewards.append(episode_reward)
    if episode % 20 == 0:
        print("Episode %d finished | Episode reward %f" %
              (episode, episode_reward))

# 累積報酬の移動平均を表示
moving_average = np.convolve(episode_rewards, np.ones(
    num_average_epidodes)/num_average_epidodes, mode='valid')
plt.plot(np.arange(len(moving_average)), moving_average)
plt.title('DQN: average rewards in %d episodes' % num_average_epidodes)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.show()

env.close()
