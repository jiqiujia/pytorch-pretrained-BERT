# -*- encoding: utf-8 -*-
'''
 Created on 2019/2/20.
 @author: eddielin
'''
import time
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

logging.basicConfig(level=logging.INFO)

model_name = 'E:\\dlprojects\\data\\bert\\chinese_L-12_H-768_A-12'

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained(model_name)
model.eval()
# If you have a GPU, put everything on cuda
# tokens_tensor = tokens_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# model.to('cuda')

# Tokenized input
# text = "冬天就应该是热乎乎的 [MASK] 孤独的灵魂也应该得到安慰" # ,
# text = "孤独的灵魂也应该得到安慰 [MASK] 你愿意来吗" # .
# text = "你愿意来吗 [MASK] B姐为你整理了几家居酒屋" # ?
# text = "B姐为你整理了几家居酒屋 [MASK] 在这座城市找到让自己安心的所在" # ,

text = '''板栗吐司
卖相在整个柜子里是最朴素的
实际却有着一股高级感的气息
里面是用板栗做的夹心
咬起来特别有存在感
表面起酥 均匀细腻的气孔
让整个吐司都有较高的含水量
满是弹润的口感
如果你以为Gentle Marble的吐司
受欢迎只是爆表的颜值和81层起酥
那你就错了
作为精致的网红吐司
连包装袋也是不会放过的
由韩国著名眼镜品牌的
设计师亲手设计的包装袋
选用质量超好的卡纸加上烫金工艺
最特别的是随意搭配出镜都能出色
无雷区
当然 这个夏天你还可以
选择一杯装满彩虹的单品降温
相比一般饮品复杂的工序
这杯“彩虹”冰饮
更注重的是简单而美好
层层叠叠的彩色
让人真的想自拍一百张
清新到心扉的气泡水
几种口感的搭配一起迸发出新奇味道
像是把彩虹都装进了杯子里
只想一个人独占一大杯
除此之外
它还是一个可以交朋友的茶
尤其适合单身的小可爱们
在卡片上勾选高、帅、有钱
鹿晗、张艺兴或是彭于晏
就会得到一杯印有二维码答案的茶
像这样出现在盖子上
相信很多人已经打过
大理石先生卡啦
绝对是品尝一次又想吃的吐司
去了现场的小可爱别忘记
祝大家都吃到美味的吐司~'''

# text = '''话说大广东这天气
# 真的不宜吃辣
# 但是随着菜系的发展和不断改良
# 好多广东人都不可自拔地爱上那一抹重口味
# 这次小编我要“辣”一下你们的嘴巴
# 精选广州几家吃辣圣地
# 爱吃辣的你一定不能错过'''
#
# text = '''用料
# 贝果一个
# 鸡蛋1个
# 安佳有盐黄油适量
# 黄瓜一根
# 植物油适量
# 做法步骤'''

# text = '''在中国
# 有这么一个老太太
# 她用她的辣椒
# 征服了全国
# 噢不
# 征服了全世界
# 并且被全世界人民
# 公认为“下饭神器”
# 没错
# 就是老干妈'''

# text = '''随着老干妈的出名
# # 贵州的辣
# # 也开始走入大家的视线
# # 娱乐圈最出名的吃货
# # “华妃娘娘”蒋欣
# # 当然也不会错过好吃的贵州辣椒
# # 然而她的推荐却并不是老干妈
# # 那是什么呢
# # 快来跟小编一起看看吧'''

# text = '''蒋欣在拍《欢乐颂》时
# 被集体吐槽太爱吃
# 她还有一套理论
# “我不吃饱了，哪有力气减肥啊！”
# 关关说
# 蒋欣给她推荐了一款辣椒
# 特别好吃'''

threshold = 0.01  # 是否加标点
texts = text.split('\n')
start = time.time()
concat = []
for i in range(len(texts) - 1):
    text1 = texts[i]
    text2 = texts[i + 1]
    text = '[CLS] ' + text1 + ' [MASK] ' + text2 + ' [SEP]'

    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = tokenized_text.index('[MASK]')
    assert tokenized_text[masked_index] == '[MASK]'

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])

    # Predict all tokens
    softmax = torch.nn.Softmax(dim=-1)
    with torch.no_grad():
        predictions = model(tokens_tensor)
        predictions = softmax(predictions)

    print("Text: " + text1 + "\t" + text2)
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token, predictions[0, masked_index, predicted_index].item())
    # max_prob = 0.0
    # sel_ch = ''
    # for ch in [':', '，', '。', '？', '!', '；']:
    #     idx = tokenizer.convert_tokens_to_ids([ch])[0]
    #     predicted_prob = predictions[0, masked_index, idx].item()
    #     print(ch, predicted_prob)
    #     if predicted_prob > max_prob:
    #         max_prob = predicted_prob
    #         sel_ch = ch
    #
    # concat.append(text1)
    # if max_prob > threshold:
    #     concat.append(sel_ch)
    # if i == len(texts) - 2:
    #     concat.append(text2)

end = time.time()
print(''.join(concat))
print('Cost: ' + str((end - start) * 1.0 / len(texts)))
