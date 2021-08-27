# KGE
from NER to Embedding

文件说明：
  bert_cef_ner.py:利用训练好的BERT_CER模型，读取服务器上的数据集，标注其中的实体
  id.py:将数据集中涉及到的user、title、word、entity编号并保存，生成训练集和测试集
  kg_generate.py:将同一个user阅读的新闻title中涉及到的entity相连，构成kg
  preoare_for_transx.py:为transx嵌入方法作准备
  embedding.py:生成嵌入
