# 実験概要

## モデル
- CNNを実装
```
# DQNのネットワーク定義
model = Sequential()
model.add(Reshape(env.observation_space.shape+(1,) , input_shape=(1,) + env.observation_space.shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(env.action_space.n))
model.add(Activation('linear'))
```

## 結果
- 途中中断

## 備考
- パラメータが多すぎると判断した

## 重要
- DQN_model.pngとlog.txtのモデルが違う
- checkpointsも生成されていない
- 何かエラーが発生している可能性が高い
- 152919のほうにcheckpointsだけ生成されている　よくわからない