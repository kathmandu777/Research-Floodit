# 実験概要

## モデル
- 少し厚めのCNNネットワーク
```
# DQNのネットワーク定義
model = Sequential()
model.add(Reshape(env.observation_space.shape+(1,) , input_shape=(1,) + env.observation_space.shape))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
```

## 結果
- だめ

## 備考
- 
