# FloodIt最適化について

## 参考論文・サイト
- [FloodIt最適化問題についての議論 stack over flow](https://stackoverflow.com/questions/1430962/how-to-optimally-solve-the-flood-fill-puzzle)
- [kaggleでコンペ開催を促されている](https://www.kaggle.com/general/7512)
- [最小ステップを見つける研究](https://github.com/raghadd/Flood-It)
- [floodit solver](https://www.youtube.com/watch?v=DLcdTck-SeQ)


## 分かったこと
- color**max_lifeの通りを全探索すると最短経路を導出可能  

|difficulty|max_life|P|≒|
| :---: | :---: | :---: | :---: |
|small|10|6.0466176e+7|6000万|
|medium|30|2.2107392e+23|2000垓|
|large|64|6.3340287e+49|60極|

cf.
|game|P|
| :---: | :---: | 
|将棋|10の220乗|
|囲碁|10の360乗以上|
|チェス|10の120乗|





