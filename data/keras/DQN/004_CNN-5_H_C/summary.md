# 実験概要

## モデル
- 軽量化したCNNネットワーク
```
# DQNのネットワーク定義
model = Sequential()
model.add(Reshape(env.observation_space.shape+(1,) , input_shape=(1,) + env.observation_space.shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
```

## 結果
- ひどい

## 備考
- 軽量化しすぎたか？
