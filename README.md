# Pre-Prospectus-Classify
NLP往次募集说明书-字段分类

该项目是基于往次披露的募集说明书的nlp文本多分类问题。使用全链接、RNN、LSTM等算法实现。

提供了标注数据（上交所公开披露的募集说明书的标题与对应分类字段），使用tensorflow2.2 + keras 实现多文本分类问题。




一，各文件介绍


requirement.txt   需安装的环境软件

demo.py           实现预测初稿

data              往次募集标题数据（标注）

callbacks         表现最佳模型和tokenizer

badcase           错例的集合

backup            表现最佳模型备份 

script            纠错脚本、预测脚本



二，项目运行方式

python v1_full_link_classify.py


三，总结

项目结果：

本项目采用了5676多条标注数据，全链接，RNN，LSTM模型训练

最佳效果达到准确率:98%，F1-Score:99%


不足之处:

1，标记数据还有较多的错例

2，模型的选择还可使用attention和Bert

3，存在过拟合现象
