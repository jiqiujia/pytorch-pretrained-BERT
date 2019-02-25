import time
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
import io
import re

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

ILLEGAL_STS_CHARS = "[^\u4e00-\u9fff0-9a-zA-Z\\- ”“\"，。*/！…；：、?+\\.:;!,~\\[\\]\\)\\(【】\n\r\t（） ]"

texts = []
with io.open('E:\\projects\\AiProductDescWriter\\local_data\\cloth\\rawdata\\wxf\\test.txt', 'r', encoding='utf-8') as fin:
    lines = fin.readlines()
    for line in lines:
        line = line.replace("]]", "").replace("➕", "+").replace("❓", "？").replace("‼", "！").replace("❗", "！").replace("❓", "？")\
            .replace("[⃣ ️]", "").replace(" ⃣", "").replace("\uD83C\uDDEE\uD83C\uDDF9", "").replace("\uD83C\uDE36️", "有")\
            .replace("适合?\uD83C\uDE34️", "适合").replace("\uD83C\uDE1A️", "无").replace("\uD83C\uDE37️", "月")\
            .replace("\uD83C\uDE35足", "满足")
        line = re.sub(r'【[^】]*】', " ", line)
        line = re.sub('&#x[0-9a-z]{2,5};', " ", line)
        line = re.sub(ILLEGAL_STS_CHARS, "，", line)
        stses = re.split("[。！!,，;；:：]", line)
        texts.append(stses)

for stses in texts:
    for i in range(len(stses) - 1):
        text1 = stses[i]
        text2 = stses[i + 1]
        text = '[CLS] ' + text1 + ' [MASK] ' + text2 + ' [SEP]'

        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        # tokens_tensor = tokens_tensor.to('cuda')
        # segments_tensors = segments_tensors.to('cuda')
        # model.to('cuda')

        # Predict  tokens
        softmax = torch.nn.Softmax(dim=-1)
        with torch.no_grad():
            predictions = model(tokens_tensor)
            predictions = softmax(predictions)

        predicted_index = torch.argmax(predictions[0, 1]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print(predicted_token, predictions[0, 1, predicted_index].item())

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