![Licence](https://img.shields.io/github/license/lawRossi/semantic_matching)
![Python](https://img.shields.io/badge/Python->=3.6-blue)

## 简介
   这是一个轻量的语义匹配库，主要包含文本编码器、索引两个组件。编码器实现了包括siamese-cbow、
   multihead attention、transformer和BERT等多种模型, 索引组件包括Annoy索引和Faiss索引。

## 编码器
语义匹配模型均采用下图所示的孪生网络结构，这种结构也叫双编码器(dual encoder)结构。句子首先进行
token级别的编码，然后通过池化操作得到句子的编码(句向量)，最后再计算句子向量的余弦相似度作为两
个句子的匹配程度。训练时支持同批负采样训练方式(in-batch negative)，并使用对比损失函数
(Contrastive Loss)。

![](resource/encoder.png)

### 困难负样本采样
研究表明训练时双编码器时，随机采样的负样本通常与正样本很容易区分，这样训练出来的编码器效果欠佳，
可以加入一些困难负样本(hard negative)进行训练来提高模型的效果。这里实现了一种利用索引的采样
方法，首先使用随机负采样训练一个初始模型，然后用该模型编码训练集的句子并构建索引。对一个正样本
句子对，借助索引找到与相似度高的一些句子当做负样本。

## 索引
编码器训练完后，即可对句子进行编码并构建索引，在线服务时对于一个query句子，先用编码器对它进行编码，
然后再通过索引检索最相似的句子，最后根据阈值判断是否匹配（或者把检索出来的句子当做候选，再过一个
排序模型找到最终匹配的句子）。Annoy和Faiss是两个常用的向量最近邻检索框架，这里基于这两个库实现了
两个索引组件，使用起来很方便。

## 安装

将代码克隆到本地，然后可通过下面的命令进行安装：

    python setup.py install
   
然后通过pip安装依赖：
    
    pip install -r requirements.txt

## 使用

### 训练模型
可通过修改train_encoder.sh脚本的参数来训练编码器模型，具体参数可参考train.py。训练数据格式为每一行一个同义句字对（即正样本），中间用'\t'分隔。

### 构建索引
编码器模型训练完后即可构建句子索引，示例如下：

```python
from semantic_matching.wrapper import EncoderWrapper
from semantic_matching.index import AnnoyIndex


encoder = EncoderWrapper("path/to/encoder-model")  # 指定编码器模型文件夹加载编码器
index_dir = "index" # 保存索引的路径
index = AnnoyIndex(encoder, index_dir)
# 每个document需要包含id和index_text字段
documents = [{"id": 1, "index_text": "sentence1"}, {"id": 2, "index_text": "sentence2"}]
index.build_index(documents)
```

### 检索
索引构建后则可以进行检索，示例如下：

```python
from semantic_matching.wrapper import EncoderWrapper
from semantic_matching.index import AnnoyIndex

encoder = EncoderWrapper("path/to/encoder-model")
index_dir = "index"
index = AnnoyIndex(encoder, index_dir)
index.load_index()  # 加载索引
query = "查询句子"
topk = 10
retrieved_ids = index.retrieve(query, topk)
print(retrieved_ids)
```

## 参考文献
1. Reimers, Nils, and Iryna Gurevych. "Sentence-bert: Sentence embeddings using siamese bert-networks." arXiv preprint arXiv:1908.10084 (2019).

2. Ding, Yingqi Qu Yuchen, et al. "RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering." arXiv preprint arXiv:2010.08191 (2020).

3. Gao, Tianyu, Xingcheng Yao, and Danqi Chen. "SimCSE: Simple Contrastive Learning of Sentence Embeddings." arXiv preprint arXiv:2104.08821 (2021).

4. Kenter, Tom, Alexey Borisov, and Maarten De Rijke. "Siamese cbow: Optimizing word embeddings for sentence representations." arXiv preprint arXiv:1606.04640 (2016).
