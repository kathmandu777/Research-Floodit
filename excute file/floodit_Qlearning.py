# Q-learnig による実装

# 参考:https://www.tcom242242.net/entry/ai-2/%e5%bc%b7%e5%8c%96%e5%ad%a6%e7%bf%92/%e3%80%90%e5%bc%b7%e5%8c%96%e5%ad%a6%e7%bf%92%e3%80%81%e5%85%a5%e9%96%80%e3%80%91q%e5%ad%a6%e7%bf%92_%e8%bf%b7%e8%b7%af%e3%82%92%e4%be%8b%e3%81%ab/

import pickle
import datetime
import copy
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import gym
import gym_floodit

# 強化学習用


def ML_main():
    start = time.time()  # プログラムの実行時間を測定

    # 各種設定
    num_episode = 1200  # 学習エピソード数
    penalty = 0.5  # 途中でエピソードが終了したときのペナルティ

    # ログ（記録用変数）
    episode_rewards = []
    win_count = 0
    lose_count = 0
    win_rate = []
    lose_rate = []

    env = gym.make("floodit-v0")  # gameの初期化（インスタンス作成）
    agent = QLearningAgent(6*6, env.action_space.n, 6)

    for episode in range(1, num_episode+1):
        observation = env.reset()
        episode_reward = 0

        while(not done):
            action = agent.get_action(observation, episode)  # 行動を選択
            next_observation, reward, done, info = env.step(
                action)  # 色の変更(FloodIt)

            episode_reward += reward
            agent.update_qtable(observation, action, reward, next_observation)

            observation = next_observation

            if (info["isWon"]):
                win_count += 1
            if (info["isLose"]):
                lose_count += 1

        episode_rewards.append(episode_reward)

        # 勝敗率の計算
        win_rate.append(win_count / episode * 100)
        lose_rate.append(lose_count / episode * 100)

        # グラフの描画(リアルタイム)
        if (episode == 1):
            fig, ax = plt.subplots()
            # scat = ax.scatter(np.arange(episode), episode_rewards, s=10,c="r", marker=".")  # scale,color
            ax.set_ylabel("Rewards")
            lines, = ax.plot(np.arange(episode), win_rate)
            lines2, = ax.plot(np.arange(episode), lose_rate)
            # plt=折れ線グラフ,scatter=散布図
        else:
            #scat.set_offsets(np.c_[np.arange(episode), episode_rewards])
            lines.set_data(np.arange(episode), win_rate)
            lines2.set_data(np.arange(episode), lose_rate)
            ax.set_xlim((np.arange(episode).min(), np.arange(episode).max()))
            ax.set_ylim(min(min(lose_rate), min(win_rate)) - 0.1,
                        max(max(win_rate), max(lose_rate)) + 0.1)
            ax.set_title("Win=" + str(win_count) + ", Lose=" +
                         str(lose_count)+", WinRate="+str('{:.4f}'.format(win_rate[-1]))+"%")
            ax.set_xlabel("Episode="+str(episode))
            plt.pause(.0001)

    env.quit()

    # 経過時間の計測
    elapsed_time = time.time() - start
    elapsed_time = datetime.timedelta(seconds=elapsed_time)

    # 実行時の日付
    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 結果の書き出し(pickle)
    pickle_dump(agent.qtable, "../Q-learning/result_pickle/" + now_time +
                str("result.pickle"))

    # 結果の書き出し(txt)
    f = open("../Q-learning/result_txt/"+now_time + str("result.txt"), "w")
    f.write("Episode="+str(episode)+", Win=" +
            str(win_count)+", Lose="+str(lose_count) + ", WinRate="+str(win_rate[-1]) + "%" + ", Elapsed_time="+str(elapsed_time) + '\n')
    for k, v in agent.qtable.items():
        f.write(k + ' : ' + str(v) + '\n')
    f.close()

    # 結果のプロット
    plt.scatter(np.arange(episode-1), episode_rewards, s=10,
                c="r", marker=".")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("Final Result(Elapsed_time="+str(elapsed_time) +
              ")\nWin="+str(win_count)+", Lose="+str(lose_count)+", WinRate="+str(win_rate[-1])+"%")
    plt.savefig("../Q-learning/result_plt/"+now_time+str("result.jpg"))
    plt.show()

    # 累積報酬の移動平均を表示
    num_average_epidodes = 10
    moving_average = np.convolve(episode_rewards, np.ones(
        num_average_epidodes)/num_average_epidodes, mode='valid')
    plt.plot(np.arange(len(moving_average)), moving_average)
    plt.title('Q-Learning: average rewards in %d episodes' %
              num_average_epidodes)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.show()


def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)


class QLearningAgent:
    def __init__(self, num_state, num_action, num_discretize, gamma=0.99, alpha=0.5, max_initial_q=0.1):
        self.num_action = num_action
        self.gamma = gamma  # 割引率
        self.alpha = alpha  # 学習率
        # Qテーブルを作成し乱数で初期化
        self.qtable = np.random.uniform(
            low=-max_initial_q, high=max_initial_q, size=(num_discretize**num_state, num_action))

    # Qテーブルを更新
    def update_qtable(self, state, action, reward, next_state):
        self.qtable[state, action] \
            += self.alpha*(reward+self.gamma*self.qtable[next_state, np.argmax(self.qtable[next_state])] - self.qtable[state, action])

    # Q値が最大の行動を選択
    def get_greedy_action(self, state):
        action = np.argmax(self.qtable[state])
        return action

    # ε-greedyに行動を選択
    def get_action(self, state, episode):
        epsilon = 0.7 * (1/(episode+1))  # ここでは0.5から減衰していくようなεを設定
        if epsilon <= np.random.uniform(0, 1):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.num_action)
        return action


if __name__ == '__main__':
    ML_main()
