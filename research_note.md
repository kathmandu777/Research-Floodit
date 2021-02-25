&# 研究記録
[マークダウン記法チートシート](https://qiita.com/kamorits/items/6f342da395ad57468ae3)

### WinterVacation
- FloodItのコード理解
- FloodItが固まるissue  
&emsp;==> pygame.event.get() を行わなかったため、windows側に動いていないと判定された
- FloodItにchanged_squareの実装
- Qlearningの実験  
&emsp;-> 数式を理解しておらず、エージェントに与える環境状態が異なっていたため、当時の実験データはあまり意味がない。
- Colabが使用できるかの実験  
&emsp;-> 映像出力が難しい。  
&emsp;new point : 松尾研のものを参考にするとできるかもしれない

### 2021/02/23
- OpenAIGymのAPI形式に合わせてFlooditを改良
- 松尾研作成のQ-learningを改良してFloodItに適合させた  
&emsp;-> ValueError: Maximum allowed dimension exceeded で終わり  
&emsp;&emsp;==> Qlearningでは解決できない(全通り試すにはとてつもない時間とメモリを要する)ことを確認
- randomで10,000エピソード実行した結果、win_rate=8%とlose_rate=18%程度に収束することが確認出来た。[参照:20210223_220844win_lose_rate.jpg]

### 2021/02/24
- Error:Keras requires TensorFlow 2.2 or higher. Install TensorFlow via `pip install tensorflow`  
&emsp;-> https://note.com/break_break/n/n0e94e592211c を参考にanaconda環境で環境構築
- Error:len is not well defined for symbolic Tensors. (activation_4/Identity:0) Please call `x.shape` rather than `len(x)` for shape information.  
&emsp;-> https://qiita.com/sky11fur/items/230c7898cfcd2948f608　(コマンドラインは厳格なので、==の前後に空白を入れると動かない)
- Error:'tensorflow' has no attribute 'get_default_graph'  
&emsp;-> https://qiita.com/white1107/items/67ec0826a07a84442d31 解決策5
- Error:AttributeError: Tensor.op is meaningless when eager execution is enabled.  
&emsp;-> 知らないうちに解決
- 従来は同じ手を選んだときにペナルティを与えていたが、それはつまるところ、変更されたマスが0ということなので、実装方法をそちらにシフトチェンジ
- 重みが保存されない  
&emsp;-> pip install h5py ==>ダメ  
&emsp;-> 拡張子を様々変更 ==>ダメ
&emsp;-> overwrite=False ==>成功 (よくわからん)
&emsp;&emsp;===> 偶然見つけたが、全てUsers/katorに保存されていた (anacondaの仕様が分からない)
- 重みを親のフォルダ内に保存不可  
&emsp;-> sys.path.append("../") ==> ダメ  
&emsp;-> 直接 ../fname　で参照 ==> 成功 (conda環境ではイケルっぽい？)
- 1,000,000ステップ学習  
&emsp;==> モデルの厚みが少なく学習出来なかった 報酬設計も見直す必要があるかもしれない
- plot_modelがうまくいかない  
&emsp;- >https://www.servernote.net/article.cgi?id=failed-to-import-pydot
- anacondaの仕様を理解 pathはpython fname.pyを実行したところが基準となる
- logが保存できない  
&emsp;-> pathの指定方法を .\\result\\DQN\\ のようにする [参考](https://github.com/ibab/tensorflow-wavenet/issues/255)
- 一手行うごとに軽微な罰を与える必要がある
- 軽量なモデルでも学習出来ていたことが分かった
