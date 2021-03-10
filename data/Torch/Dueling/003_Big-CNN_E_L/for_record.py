# DQNを用いてFlood-Itを攻略

import matplotlib.pyplot as plt
import sys
import datetime
import gym
import numpy as np
import os
import re
import shutil
from stat import SF_IMMUTABLE

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import random

sys.path.append("../gym-floodit")
import gym_floodit  # nopep
sys.path.pop()


#!Seed値の固定


def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 処理速度は落ちる


SEED = 42  # ?適宜変更
fix_seed(SEED)


#!FloodItの初期化
# HIGH:H or MEDIUM:M or EASY:E
level = "E"  # ?適宜変更


class TorchFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        height, width = self.observation_space.shape
        channels = 1
        self.observation_space = gym.spaces.Box(
            low=0,
            high=5,
            shape=(channels, height, width),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return torch.as_tensor(obs, dtype=torch.float).unsqueeze(0)


env = gym.make(
    "floodit-v0",
    level=level,
    ignore_same_action=True,
    visualize=False,
    win_reward=10,
    lose_penalty=5,
    same_action_penalty=0.1,
    time_penalty=None,  # None=1 / (BOARDSIZE[self.level] * 2) / 10
    action_reward=None  # None=self.changed_square /(BOARDSIZE[self.level] * 2)
)
env = TorchFrame(env)  # ラッパーの追加
env.seed(SEED)
env.action_space.seed(SEED)

print("\n\n")
print("*" * 50 + "  FloodIt  " + "*" * 50)
print("action_space      : " + str(env.action_space))
print("observation_space : " + str(env.observation_space))
print("reward_range      : " + str(env.reward_range))
print()


#!各データ保存用フォルダの作成
summary = "Big-CNN"  # ?毎回変更

result_folder_path = "C:/Users/kator/OneDrive/ドキュメント/ResearchFloodit/data/Torch"
result_folder_path += "/Dueling2"  # ?適宜変更

file_and_folder = os.listdir(result_folder_path)
dir_list = [f for f in file_and_folder if os.path.isdir(
    os.path.join(result_folder_path, f))]
if (len(dir_list) == 0):
    last_num = 0
else:
    last_num = int(re.findall('^[0-9]+', dir_list[-1])[0])  # 先頭の数字のみ抽出
next_num = str(last_num + 1).zfill(3)
folder_name = next_num + "_" + summary + "_" + level + "_L"
folder_path = os.path.join(result_folder_path, folder_name)
os.makedirs(folder_path)
os.makedirs(os.path.join(folder_path, "checkpoints"))

shutil.copyfile(os.path.abspath(__file__), folder_path +
                "/for_record.py")  # 実行したファイルのコピーを作成
os.chmod(folder_path + "/for_record.py",
         SF_IMMUTABLE)  # change to read-only


#!Torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


#!Prioritized Experience Replayを実現するためのメモリクラス


class PrioritizedReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.index = 0
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.priorities[0] = 1.0

    def __len__(self):
        return len(self.buffer)

    # 経験をリプレイバッファに保存する． 経験は(obs, action, reward, next_obs, done)の5つ組を想定
    def push(self, experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience

        # 優先度は最初は大きな値で初期化しておき, 後でサンプルされた時に更新する
        self.priorities[self.index] = self.priorities.max()
        self.index = (self.index + 1) % self.buffer_size

    def sample(self, batch_size, alpha=0.6, beta=0.4):
        # 現在経験が入っている部分に対応する優先度を取り出し, サンプルする確率を計算
        priorities = self.priorities[: self.buffer_size if len(
            self.buffer) == self.buffer_size else self.index]
        priorities = priorities ** alpha
        prob = priorities / priorities.sum()

        # 上で計算した確率に従ってリプレイバッファ中のインデックスをサンプルする
        indices = np.random.choice(len(self.buffer), batch_size, p=prob)

        # 学習の方向性を補正するための重みを計算
        weights = (len(self.buffer) * prob[indices]) ** (-beta)
        weights = weights / weights.max()

        # 上でサンプルしたインデックスに基づいて経験をサンプルし, (obs, action, reward, next_obs, done)に分ける
        obs, action, reward, next_obs, done = zip(
            *[self.buffer[i] for i in indices])

        # あとで計算しやすいようにtorch.Tensorに変換して(obs, action, reward, next_obs, done, indices, weights)の7つ組を返す
        return (torch.stack(obs),
                torch.as_tensor(action),
                torch.as_tensor(reward, dtype=torch.float32),
                torch.stack(next_obs),
                torch.as_tensor(done, dtype=torch.uint8),
                indices,
                torch.as_tensor(weights, dtype=torch.float32))

    # 優先度を更新する. 優先度が極端に小さくなって経験が全く選ばれないということがないように, 微小値を加算しておく.
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities + 1e-4


#!Dueling Networkを用いたQ関数を実現するためのニューラルネットワークをクラスとして記述
class CNNQNetwork(nn.Module):
    def __init__(self, state_shape, n_action):
        super(CNNQNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_action = n_action

        # ? ネットワークの構成

        # Dueling Networkでも, 畳込み部分は共有する
        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_shape[0], 64, kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=2,
                      stride=1),
            nn.ReLU(),
        )

        cnn_out_size = self.check_cnn_size(state_shape)
        # Dueling Networkのための分岐した全結合層
        # 状態価値
        self.fc_state = nn.Sequential(
            nn.Linear(cnn_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # アドバンテージ
        self.fc_advantage = nn.Sequential(
            nn.Linear(cnn_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_action)
        )

    def check_cnn_size(self, shape):
        shape = torch.FloatTensor(1, shape[0], shape[1], shape[2])
        out = self.conv_layers(shape).size()
        out = np.prod(np.array(out))
        return out

    def forward(self, obs):
        feature = self.conv_layers(obs)
        feature = feature.view(feature.size(0), -1)  # Flatten. 64*11*11->7744

        state_values = self.fc_state(feature)
        advantage = self.fc_advantage(feature)

        # 状態価値 + アドバンテージ で行動価値を計算しますが、安定化のためアドバンテージの（行動間での）平均を引きます
        action_values = state_values + advantage - \
            torch.mean(advantage, dim=1, keepdim=True)
        return action_values

    # epsilon-greedy. 確率epsilonでランダムに行動し, それ以外はニューラルネットワークの予測結果に基づいてgreedyに行動します.
    def act(self, obs, epsilon):
        if random.random() < epsilon:
            action = random.randrange(self.n_action)
        else:
            # 行動を選択する時には勾配を追跡する必要がない
            with torch.no_grad():
                action = torch.argmax(self.forward(obs.unsqueeze(0))).item()
        return action


#!リプレイバッファの宣言
# ? パラメータ
buffer_size = 100000  # リプレイバッファに入る経験の最大数
initial_buffer_size = 1000  # 学習を開始する最低限の経験の数
replay_buffer = PrioritizedReplayBuffer(buffer_size)


#!ネットワークの宣言
net = CNNQNetwork(env.observation_space.shape,
                  n_action=env.action_space.n).to(device)
target_net = CNNQNetwork(env.observation_space.shape,
                         n_action=env.action_space.n).to(device)
target_update_interval = 500  # ? 学習安定化のために用いるターゲットネットワークの同期間隔(Episode)


#!オプティマイザとロス関数の宣言

optimizer = optim.Adam(net.parameters(), lr=1e-4)  # ? オプティマイザはAdam
# ? ロスはSmoothL1loss（別名Huber loss）
loss_func = nn.SmoothL1Loss(reduction='none')

# ? ハイパーパラメータ
gamma = 0.99  # 割引率
batch_size = 64
n_episodes = 50000  # 学習を行うエピソード数

# ? Prioritized Experience Replayのためのパラメータβ
beta_begin = 0.4
beta_end = 1.0
beta_decay = n_episodes - (n_episodes / 100)


def beta_func(episode):
    return min(beta_end, beta_begin +
               (beta_end - beta_begin) * (episode / beta_decay))


# ? 探索のためのパラメータε

epsilon_begin = 1.0
epsilon_end = 0.05
epsilon_decay = n_episodes - (n_episodes / 100)


def epsilon_func(episode):
    return max(epsilon_end, epsilon_begin -
               (epsilon_begin - epsilon_end) * (episode / epsilon_decay))


def update(batch_size, beta):
    obs, action, reward, next_obs, done, indices, weights = replay_buffer.sample(
        batch_size, beta)
    obs, action, reward, next_obs, done, weights \
        = obs.to(device), action.to(device), reward.to(device), next_obs.to(device), done.to(device), weights.to(device)

    # ニューラルネットワークによるQ関数の出力から, .gatherで実際に選択した行動に対応する価値を集めてきます.
    q_values = net(obs).gather(1, action.unsqueeze(1)).squeeze(1)

    # 目標値の計算なので勾配を追跡しない
    with torch.no_grad():
        # Double DQN.
        # ① 現在のQ関数でgreedyに行動を選択し,
        greedy_action_next = torch.argmax(net(next_obs), dim=1)
        # ② 対応する価値はターゲットネットワークのものを参照します.
        q_values_next = target_net(next_obs).gather(
            1, greedy_action_next.unsqueeze(1)).squeeze(1)

    # ベルマン方程式に基づき, 更新先の価値を計算します.
    # (1 - done)をかけているのは, ゲームが終わった後の価値は0とみなすためです.
    target_q_values = reward + gamma * q_values_next * (1 - done)

    # Prioritized Experience Replayのために, ロスに重み付けを行なって更新します.
    optimizer.zero_grad()
    loss = (weights * loss_func(q_values, target_q_values)).mean()
    loss.backward()
    optimizer.step()

    # TD誤差に基づいて, サンプルされた経験の優先度を更新します.
    replay_buffer.update_priorities(
        indices, (target_q_values - q_values).abs().detach().cpu().numpy())

    return loss.item()


#!TensorBoardを設定
writer = SummaryWriter(folder_path)
# netの可視化
obs = env.reset()
writer.add_graph(net, obs.float().to(device).unsqueeze(0))

#!学習
total_step = 0
total_reward = 0
win_num = 0
lose_num = 0

try:
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_step = 0

        while not done:
            # ε-greedyで行動を選択
            action = net.act(obs.to(device), epsilon_func(episode))
            # 環境中で実際に行動
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            total_reward += reward

            # リプレイバッファに経験を蓄積
            replay_buffer.push([obs, action, reward, next_obs, done])
            obs = next_obs

            # ネットワークを更新
            if len(replay_buffer) > initial_buffer_size:
                loss = update(batch_size, beta_func(episode))
                writer.add_scalar('Loss', loss, total_step)

            episode_step += 1
            total_step += 1

            # ターゲットネットワークを定期的に同期させる
            if (total_step) % target_update_interval == 0:
                target_net.load_state_dict(net.state_dict())

            writer.add_scalar('Changed_square',
                              info["changed_square"], total_step)

        if (info["isWon"]):
            win_num += 1
        elif (info["isLose"]):
            lose_num += 1

        if (episode + 1) % 100 == 0:
            print('Episode: {},  Episode-Step: {},  EpisodeReward: {}'.format(episode +
                                                                              1, episode_step, episode_reward))

        writer.add_scalar('Total-Reward', total_reward, episode)
        writer.add_scalar('Episode-Reward', episode_reward, episode)
        writer.add_scalar('Episode-Step', episode_step, episode)
        writer.add_scalar('Win-Rate', win_num / (episode + 1) * 100, episode)
        writer.add_scalar('Lose-Rate', lose_num / (episode + 1) * 100, episode)

        # 定期的に重みを保存
        if (episode + 1) % 2000 == 0:
            torch.save(net.state_dict(), folder_path +
                       "/checkpoints/weights_{}episodes.pth".format(episode + 1))

except KeyboardInterrupt:
    print("学習中断")

writer.close()
torch.save(net.state_dict(), folder_path + "/weights_final.pth")


#!学習結果の確認

env = gym.make(
    "floodit-v0",
    level=level,
    ignore_same_action=True,
    visualize=True,
    win_reward=10,
    lose_penalty=5,
    same_action_penalty=0.1,
    time_penalty=None,  # None=1 / (BOARDSIZE[self.level] * 2) / 10
    action_reward=None  # None=self.changed_square /(BOARDSIZE[self.level] * 2)
)
env = TorchFrame(env)  # ラッパーの追加
env.seed(SEED)
env.action_space.seed(SEED)


experiment_times = 5  # ?適宜変更

win_num = 0
lose_num = 0

for i in range(experiment_times):
    done = False
    obs = env.reset()
    while not done:
        env.render()
        action = net.act(obs.to(device), epsilon=0.0)
        obs, reward, done, info = env.step(action)
        print(reward, info)

    print(i, end=" : ")
    if (info["isWon"]):
        win_num += 1
    elif (info["isLose"]):
        lose_num += 1

print()
print("Result ({}times)".format(experiment_times))
print("-------------------")
win_rate = win_num / (experiment_times) * 100
lose_rate = lose_num / (experiment_times) * 100
print("win_rate:", win_rate)
print("lose_rate", lose_rate)
print("sum:", win_rate + lose_rate)
