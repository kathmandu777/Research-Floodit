

import gym
import gym_floodit
import datetime
import copy
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import sys

start = time.time()  # プログラムの実行時間を測定

# ログ（記録用変数）
episode_rewards = []
win_count = 0
lose_count = 0
win_rate = []
lose_rate = []

env = gym.make("floodit-v0")  # envの初期化（インスタンス作成）

num_episode = 10000  # 学習エピソード数

for episode in range(1, num_episode+1):
    observation = env.reset()
    episode_reward = 0
    done = False
    isQuit = False
    while(not done):
        action = env.action_space.sample()
        next_observation, reward, done, info = env.step(
            action)  # 色の変更(FloodIt)

        episode_reward += reward

        if (info["isWon"]):
            win_count += 1
        if (info["isLose"]):
            lose_count += 1
        if (info["isQuit"]):
            isQuit = True
            break

    episode_rewards.append(episode_reward)

    # 勝敗率の計算
    win_rate.append(win_count / episode * 100)
    lose_rate.append(lose_count / episode * 100)

    # グラフの描画(リアルタイム)
    if (episode == 1):
        fig, ax = plt.subplots()
        # scat = ax.scatter(np.arange(episode), episode_rewards, s=10,c="r", marker=".")  # scale,color
        ax.set_ylabel("Rate")
        lines, = ax.plot(np.arange(episode), win_rate, label="win rate", c="r")
        lines2, = ax.plot(np.arange(episode), lose_rate,
                          label="lose rate", c="b")
        ax.legend()
        # plt=折れ線グラフ,scatter=散布図
    else:
        #scat.set_offsets(np.c_[np.arange(episode), episode_rewards])
        lines.set_data(np.arange(episode), win_rate)
        lines2.set_data(np.arange(episode), lose_rate)
        ax.set_xlim((np.arange(episode).min(), np.arange(episode).max()))
        ax.set_ylim(-0.1, 100.1)
        ax.set_title("Win=" + str(win_count) + ", Lose=" + str(lose_count) +
                     ", WinRate="+str('{:.4f}'.format(win_rate[-1]))+" %, " +
                     "LoseRate="+str('{:.4f}'.format(lose_rate[-1]))+"%")
        ax.set_xlabel("Episode="+str(episode))
        plt.pause(.0001)

    if (isQuit):
        break

env.quit()

# 経過時間の計測
elapsed_time = time.time() - start
elapsed_time = datetime.timedelta(seconds=elapsed_time)

# 実行時の日付
now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

sys.path.append('../')
# 結果のプロット
plt.plot(np.arange(episode), win_rate, label="win rate", c="r")
plt.plot(np.arange(episode), lose_rate, label="lose rate", c="b")
plt.xlabel("episode")
plt.ylabel("rate")
plt.title("Final Result(Elapsed_time=" + str(elapsed_time) + ")" +
          "\nWin="+str(win_count)+", Lose="+str(lose_count)+", WinRate="+str(win_rate[-1])+"%, LoseRate="+str(lose_rate[-1])+"%")
plt.savefig("result/random/rate_plt/" + now_time + str("win_lose_rate.jpg"))
plt.show()

# 累積報酬の移動平均を表示
num_average_epidodes = 10
moving_average = np.convolve(episode_rewards, np.ones(
    num_average_epidodes)/num_average_epidodes, mode='valid')
plt.plot(np.arange(len(moving_average)), moving_average)
plt.title('Random: average rewards in %d episodes' %
          num_average_epidodes)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.savefig("result/random/ave_rewards_plt/" +
            now_time + str("ave_rewards.jpg"))
plt.show()
