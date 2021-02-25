# FloodItのOpenAIGym環境

## 設計
- 環境状態(observation)=[self.mainBoard, self.life]

## 関数定義
```python
def reset(self):
    return observation

def step(self, action):
    return observation, reward, done, info
```