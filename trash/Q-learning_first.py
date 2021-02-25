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

    game = gym.make("floodit-v0")  # gameの初期化（インスタンス作成）
    game.render()
    board, life = game.reset()

    agent = QLearningAgent(epsilon=.1, alpha=0.5, actions=np.arange(
        6), observation=[board, True if(life != 0) else False],)  # Q学習エージェントの初期化（インスタンス作成）
    # nb_episode = 10000  # エピソード数

    rewards = []    # 評価用報酬の保存
    is_end_episode = False  # エージェントがゴールしてるかどうか？
    episode = 1
    win_count = 0
    lose_count = 0
    win_rate = []

    while(True):
        episode_reward = []  # 1エピソードの累積報酬
        pre_action = None
        game.reset()
        # board, life = game.get_info()
        # state = [board, life]
        # agent.observe(state)
        while(is_end_episode == False):    # 1game終わるまで続ける
            action = agent.act()  # 行動選択
            changed_square = game.step(action)  # 色の変更(FloodIt)
            board, life = game.get_info()
            life = True if(life != 0) else False  # ライフが0でなければTrue,0であればFalse
            state = [board, life]  # アクションを起こした結果の環境をobservさせる
            isLose, isWon = game.get_game_condition()  # ゲーム状況の取得（勝ちor負け）
            reward = 0.0
            if (pre_action == None):
                pass
            elif (pre_action != action):  # 前回と違う手を選んだら
                pass
            else:  # 前回と同じ手を選んだら
                reward = -0.5
                is_end_episode = True

            if (changed_square == 0):  # 色が変わったマスが0だったら
                reward = -0.2
            else:
                reward = changed_square*0.1

            if (isWon):  # 勝ったら
                reward = 5
                win_count += 1
                is_end_episode = True
            elif (isLose):  # ライフが0になったら
                reward = -1
                lose_count += 1
                is_end_episode = True
            agent.observe(state, reward)  # 状態と報酬の観測
            episode_reward.append(reward)
            pre_action = action
            print(str(reward))
        rewards.append(np.sum(episode_reward))  # このエピソードの平均報酬を与える
        print(str(episode)+" : "+str(np.sum(episode_reward))+"\n")
        agent.observe(state)    # エージェントを初期位置に
        is_end_episode = False

        # 勝率の計算
        win_rate.append(win_count / episode * 100)

        # グラフの描画(リアルタイム)
        if (episode == 1):
            fig, ax = plt.subplots()
            scat = ax.scatter(np.arange(episode), rewards, s=10,
                              c="r", marker=".")  # scale,color
            ax.set_ylabel("Rewards")
            lines, = ax.plot(np.arange(episode), win_rate)
            # plt=折れ線グラフ,scatter=散布図
        else:
            scat.set_offsets(np.c_[np.arange(episode), rewards])
            lines.set_data(np.arange(episode), win_rate)
            ax.set_xlim((np.arange(episode).min(), np.arange(episode).max()))
            ax.set_ylim(
                (min(rewards) - 0.2, max(max(rewards), max(win_rate)) + 0.2))
            ax.set_title("Win=" + str(win_count) + ", Lose=" +
                         str(lose_count)+", WinRate="+str('{:.4f}'.format(win_rate[-1]))+"%")
            ax.set_xlabel("Episode="+str(episode))
            plt.pause(.0001)

        episode += 1

        if (game.checkReset()):  # pygameの画面でResetボタンが押されていたら
            break

    game.quit()
    elapsed_time = time.time() - start
    elapsed_time = datetime.timedelta(seconds=elapsed_time)

    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 結果の書き出し
    pickle_dump(agent.q_values, "./Q-learning/result_pickle/" + now_time +
                str("result.pickle"))
    f = open("./Q-learning/result_txt/"+now_time + str("result.txt"), "w")
    f.write("Episode="+str(episode)+", Win=" +
            str(win_count)+", Lose="+str(lose_count) + ", WinRate="+str(win_rate[-1]) + "%" + ", Elapsed_time="+str(elapsed_time) + '\n')
    for k, v in agent.q_values.items():
        f.write(k + ' : ' + str(v) + '\n')
    f.close()

    # 結果のプロット
    plt.scatter(np.arange(episode-1), rewards, s=10,
                c="r", marker=".")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("Final Result(Elapsed_time="+str(elapsed_time) +
              ")\nWin="+str(win_count)+", Lose="+str(lose_count)+", WinRate="+str(win_rate[-1])+"%")
    plt.savefig("./Q-learning/result_plt/"+now_time+str("result.jpg"))
    plt.show()


def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)


class QLearningAgent:  # Q学習

    def __init__(self, alpha=.2, epsilon=.1, gamma=.99, actions=None, observation=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward_history = []
        self.actions = actions
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = None
        self.previous_action = None
        self.q_values = self._init_q_values()

    def _init_q_values(self):
        """
           Q テーブルの初期化
        """
        q_values = {}
        q_values[self.state] = np.repeat(0.0, len(self.actions))
        return q_values

    def init_state(self):
        """
            状態の初期化
        """
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def act(self):  # ε-greedy選択
        if np.random.uniform() < self.epsilon:
            # random行動(epsilon(探索率)=0.5,np.random.uniform()=0~1の乱数 -> 50%の確率でランダム行動)
            action = np.random.randint(0, len(self.q_values[self.state]))
        else:   # greedy 行動
            action = np.argmax(self.q_values[self.state])
            # argmax=指定された配列の中で最大値となっている要素のうち先頭のインデックスを返す

        self.previous_action = action
        return action

    def observe(self, next_state, reward=None):
        """
            次の状態と報酬の観測
        """
        next_state = str(next_state)
        if next_state not in self.q_values:  # 始めて訪れる状態であれば
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))
            # e.g. : np.repeat(1,5) -> [1,1,1,1,1]

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        if reward is not None:
            self.reward_history.append(reward)
            self.learn(reward)

    def learn(self, reward):
        """
            Q値の更新
        """
        q = self.q_values[self.previous_state][self.previous_action]  # Q(s, a)
        max_q = max(self.q_values[self.state])  # max Q(s')
        """
        Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))

        alpha=学習率=Q値の更新をどれだけ急激におこなうか
        cf. https://note.com/pumonmon/n/n04f9139ad826

        gamma=割引率=将来の価値をどれだけ割り引いて考えるかのパラメータ
        cf. http://neuro-educator.com/rl1/
        """
        self.q_values[self.previous_state][self.previous_action] = q + \
            (self.alpha * (reward + (self.gamma*max_q) - q))


if __name__ == '__main__':
    ML_main()
    # randomMain()


"""
# rondom用


def randomMain():
    game = fi.floodit()
    game.render()
    for j in range(2):
        print("\nNewGame:"+str(j+1))
        for i in range(30):
            board, life = game.get_info()
            changed_square = game.give_color(
                randomly_decide_color(board, life))
            print(str("{0:2d}".format(i+1)) +
                  ", changed-square:" + str(changed_square))
            isLose, isWon = game.get_game_condition()
            if (isWon):
                print("Win")
                break
            elif (isLose):
                print("Lose")
        game.funcResetGame()
    game.quit()


def randomly_decide_color(board, life):
    return random.randint(0, 5)  # 0~5のランダム値
"""
