# 使い方（備忘録）

## Anaconda
```
conda activate RL
cd C:\Users\kator\OneDrive\ドキュメント\ResearchFloodit\gym-floodit
python  C:\Users\kator\OneDrive\ドキュメント\ResearchFloodit\gym-floodit\floodit-learning.py 
tensorboard --logdir ../result/DQN                                                                          
```

```
conda activate RL-gpu
cd C:\Users\kator\OneDrive\ドキュメント\ResearchFloodit\excute file
python floodit-learning.py

cd C:\Users\kator\OneDrive\ドキュメント\ResearchFloodit\result
tensorboard --logdir DDQN or DQN

```

```
conda activate RL-torch
cd C:\Users\kator\OneDrive\ドキュメント\ResearchFloodit\excute file
python torch-learning.py

cd C:\Users\kator\OneDrive\ドキュメント\ResearchFloodit\data
tensorboard --logdir dir_name

```
## model
(全矢印-1)/2=層 と定義
activationが単体で書かれていない時: 全矢印-Input-Reshape