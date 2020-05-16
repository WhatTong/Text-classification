## Text-classification 电影评论情感分类
使用LSTM和LSTM+Attention来进行文本二分类  

## 环境
* python3
* python 1.4.0

## 数据集
  训练集：包含2W条左右中文电影评论，其中正负向评论各1W条左右；

  验证集：包含6K条左右中文电影评论，其中正负向评论各3K条左右；

  测试集：包含360条左右中文电影评论，其中正负向评论各180条左右。  

## 结果
 模型  | Acc  
 ---- | ----- 
 LSTM  | 0.847
 LSTM+Attention  | 0.856 

