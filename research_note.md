&# 研究記録
[マークダウン記法チートシート](https://qiita.com/kamorits/items/6f342da395ad57468ae3)

### WinterVacation
- FloodItのコード理解
- FloodItが固まるissue  
    - pygame.event.get() を行わなかったため、windows側に動いていないと判定された
- FloodItにchanged_squareの実装
- Qlearningの実験  
    - 数式を理解しておらず、エージェントに与える環境状態が異なっていたため、当時の実験データはあまり意味がない。
- Colabが使用できるかの実験  
    - 映像出力が難しい。
    - new point : 松尾研のものを参考にするとできるかもしれない

### 2021/02/23
- OpenAIGymのAPI形式に合わせてFlooditを改良
- 松尾研作成のQ-learningを改良してFloodItに適合させた  
    - ValueError: Maximum allowed dimension exceeded で終わり  
    - ==> Qlearningでは解決できない(全通り試すにはとてつもない時間とメモリを要する)ことを確認
- randomで10,000エピソード実行した結果、win_rate=8%とlose_rate=18%程度に収束することが確認出来た。[参照:20210223_220844win_lose_rate.jpg]

### 2021/02/24
- Error:Keras requires TensorFlow 2.2 or higher. Install TensorFlow via `pip install tensorflow`  
    - https://note.com/break_break/n/n0e94e592211c を参考にanaconda環境で環境構築
- Error:len is not well defined for symbolic Tensors. (activation_4/Identity:0) Please call `x.shape` rather than `len(x)` for shape information.  
    - https://qiita.com/sky11fur/items/230c7898cfcd2948f608　(コマンドラインは厳格なので、==の前後に空白を入れると動かない)
- Error:'tensorflow' has no attribute 'get_default_graph'  
    - https://qiita.com/white1107/items/67ec0826a07a84442d31 解決策5
- Error:AttributeError: Tensor.op is meaningless when eager execution is enabled.  
    - 知らないうちに解決
- 従来は同じ手を選んだときにペナルティを与えていたが、それはつまるところ、変更されたマスが0ということなので、実装方法をそちらにシフトチェンジ
- 重みが保存されない  
    - pip install h5py ==>ダメ  
    - 拡張子を様々変更 ==>ダメ
    - overwrite=False ==>成功 (よくわからん)
&emsp;&emsp;===> 偶然見つけたが、全てUsers/katorに保存されていた (anacondaの仕様が分からない)
- 重みを親のフォルダ内に保存不可  
    - sys.path.append("../") ==> ダメ  
    - 直接 ../fname　で参照 ==> 成功 (conda環境ではイケルっぽい？)
- 1,000,000ステップ学習  
    - モデルの厚みが少なく学習出来なかった 報酬設計も見直す必要があるかもしれない
- plot_modelがうまくいかない  
    - https://www.servernote.net/article.cgi?id=failed-to-import-pydot
- anacondaの仕様を理解 pathはpython fname.pyを実行したところが基準となる
- logが保存できない  
    - pathの指定方法を .\\result\\DQN\\ のようにする [参考](https://github.com/ibab/tensorflow-wavenet/issues/255)
- 一手行うごとに軽微な罰を与える必要がある
- 軽量なモデルでも学習出来ていたことが分かった

### 2021/02/25
- git管理の開始
- google driveにも上げた

### 2021/02/26
- colab上でアニメーション化できるように試行錯誤
- matplotlibのカラーコード指定（16進数）が''にしか対応していない
- スカラーに対応した色が出力されない
    - [imshowが勝手に正規化している](http://hikuichi.hatenablog.com/entry/2015/12/26/225623)
- Keras symbolic inputs/outputs do not implement `__len__`. You may be trying to pass Keras symbolic inputs/outputs to a TF API that does not register dispatching, preventing Keras from automatically converting the API call to a lambda layer in the Functional Model. This error will also get raised if you try asserting a symbolic input/output directly.
    - 前回も見た。分からなかったため、にローカルでやることに決めた気がする。
    - バージョンをいじったところできた(keras==2.2.4, keras-rl==0.4.2, tensorflow==1.13.1, tensorflow-gpu==1.13.1)
- module 'tensorflow' has no attribute 'get_default_graph'
    - keras.something >>> tensorflow.python.keras.something https://github.com/keras-team/keras/issues/12379
- len is not well defined for symbolic Tensors. (DQNAgentの部分)
    - keras-rl2
- 'Adam' object has no attribute '_name'
    - from tensorflow.python.keras.optimizers import Adam >>> from tensorflow.keras.optimizers import Adam
- 'DQNAgent' object has no attribute '_get_distribution_strategy'
    - callbacks += [tf.keras.callbacks.TensorBoard(log_dir=logdir)] が原因
- Error when checking input: expected flatten_○○_input to have 4 dimensions, but got array with shape (1, 1, 2)
    - これはマジでわからない
- Model output "Tensor("activation_6/Identity:0", shape=(?, 6), dtype=float32)" has invalid shape. DQN expects a model that has one dimension for each action, in this case 6.
    - Noneが?と認識されている？ <==違うっぽい

### 2021/02/27
- 昨日のエラーが治った
```
!pip install tensorflow==2.3.0
!pip install gym
!pip install keras
!pip install keras-rl2
```
- resetの部分でobs以外にinfoも渡していたことが原因。しかし、(1,1,2)と認識されていたのは謎。(1,2)であるべき。
- gpuのほうが遅い
    - データ量の少なさが原因? https://teratail.com/questions/185545
- Reusing TensorBoard on port 6006 (pid 9412), started 0:10:57 ago. (Use '!kill 9412' to kill it.)
    - https://github.com/tensorflow/tensorboard/issues/3186
    - --port=6006 で解決
    - 時間を待てば何もしなくても出た
- 'TensorBoard' object has no attribute '_should_trace'
```
詳細な原因は分からないが、以下をインストールして再起動したところ治った。tensorflowが重要かもしれない。
!pip install tensorflow-gpu==2.3.0
!pip install tensorflow==2.3.0
!pip install gym
!pip install keras
!pip install keras-rl2
```
- reward-=0.01 # 軽微な罰 を追加
- kerasAPIの仕様でcheckpointが変に作成される==weights_final.h5f.data-00000-of-00001など
    - https://blog.shikoan.com/tensorflow-save-weights-hdf5/
        - 修正不可 keras-rlゴミ
- Optunaによるハイパーパラメータの最適化
    - https://qiita.com/pocokhc/items/8ed40be84a144b28180d
    - https://qiita.com/ryota717/items/28e2167ea69bee7e250d
- colab で1,000,000ステップ実行
    - 全く成長していなかった

### 2021/02/28
- GPU環境の構築
    - https://keita-blog.com/data_science/keras-tensorflow-gpu
    - https://qiita.com/nemutas/items/c7d9cca91a7e1bd404b6
    - https://keita-blog.com/data_science/keras-tensorflow-gpu
    - https://qiita.com/jonpili/items/e5444c31fbd16f30725a
    - https://qiita.com/pocokhc/items/-08d3fb3ef0b6fa427528#tensorflow-gpu%E7%89%88-%E3%82%92%E4%BD%BF%E3%81%86%E3%81%9F%E3%82%81%E3%81%AE%E6%BA%96%E5%82%99
    - https://qiita.com/chin_self_driving_car/items/f00af2dbd022b65c9068
- CNN-new-rewardうまくいかず
- フォルダ構成
- GPU NVIDIA関連
    - [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)
    - [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)

### 2021/03/01
- GPUで動作
```
Package                  Version
------------------------ --------------------
absl-py                  0.11.0
astor                    0.8.1
astunparse               1.6.3
atari-py                 0.2.6
cachetools               4.2.1
certifi                  2020.12.5
chardet                  4.0.0
click                    7.1.2
cloudpickle              1.6.0
configparser             5.0.1
cycler                   0.10.0
Cython                   0.29.22
docker-pycreds           0.4.0
flatbuffers              1.12
future                   0.18.2
gast                     0.3.3
gitdb                    4.0.5
GitPython                3.1.13
google-auth              1.27.0
google-auth-oauthlib     0.4.2
google-pasta             0.2.0
gql                      0.2.0
graphql-core             1.1
grpcio                   1.32.0
gym                      0.18.0
h5py                     2.10.0
idna                     2.10
importlib-metadata       3.5.0
joblib                   1.0.1
Keras                    2.3.1
Keras-Applications       1.0.8
Keras-Preprocessing      1.1.2
keras-rl                 0.4.2
keras-rl2                1.0.4
kiwisolver               1.3.1
Markdown                 3.3.3
matplotlib               3.3.4
numpy                    1.18.5
nvidia-ml-py3            7.352.0
oauthlib                 3.1.0
opencv-python            4.5.1.48
opt-einsum               3.3.0
pandas                   1.2.2
Pillow                   7.2.0
pip                      21.0.1
promise                  2.3
protobuf                 3.15.2
psutil                   5.8.0
pyasn1                   0.4.8
pyasn1-modules           0.2.8
pydot                    1.4.1
pydotplus                2.0.2
pygame                   2.0.1
pyglet                   1.5.0
pyparsing                2.4.7
python-dateutil          2.8.1
pytz                     2021.1
PyYAML                   5.4.1
requests                 2.25.1
requests-oauthlib        1.3.0
rsa                      4.7.2
scipy                    1.4.1
sentry-sdk               0.20.3
setuptools               52.0.0.post20210125
shortuuid                1.0.1
six                      1.15.0
smmap                    3.0.5
stable-baselines         2.10.1
subprocess32             3.5.4
tb-nightly               1.14.0a20190603
tensorboard              2.4.1
tensorboard-plugin-wit   1.8.0
tensorflow               2.3.0
tensorflow-estimator     2.3.0
tensorflow-gpu           2.3.0
tensorflow-gpu-estimator 2.3.0
tensorflow-hub           0.11.0
termcolor                1.1.0
tf-estimator-nightly     1.14.0.dev2019060501
torch                    1.7.1+cpu
torchaudio               0.7.2
torchvision              0.8.2+cpu
typing-extensions        3.7.4.3
urllib3                  1.26.3
wandb                    0.8.28
watchdog                 2.0.2
Werkzeug                 1.0.1
wheel                    0.36.2
wincertstore             0.2
wrapt                    1.12.1
zipp                     3.4.0

CUDA=10.1
cuDNN=7.6.5(for 10.1)
```

- demoリファクタリング
- DQNリファクタリング