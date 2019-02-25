import time
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
import io

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


stses = set()
with io.open('E:\\projects\\AiProductDescWriter\\local_data\\cloth\\rawdata\\wxf\\wxfriend_filtered3.txt', 'r', encoding='utf-8') as fin:
    lines = fin.readlines()
    for line in lines:
        line = line.split('\t')[0].strip()
        for sts in line.split('，'):
            stses.add(sts)
print('sts number: ', len(stses))

threshold = 0.01
stses = list(stses)[:1000]
res = []
for sts in stses:
    sts1 = '[CLS] [MASK] ' + sts + '[SEP]'
    #sts2 = '[CLS]' + sts + '[MASK][SEP]'
    print(sts1)

    tokenized_text = tokenizer.tokenize(sts1)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])

    # If you have a GPU, put everything on cuda
    # tokens_tensor = tokens_tensor.to('cuda')
    # segments_tensors = segments_tensors.to('cuda')
    # model.to('cuda')

    # Predict all tokens
    softmax = torch.nn.Softmax(dim=-1)
    with torch.no_grad():
        predictions = model(tokens_tensor)
        predictions = softmax(predictions)

    masked_index = 1
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token, predictions[0, masked_index, predicted_index].item())

    max_prob = 0.0
    sel_ch = ''
    for ch in [':', '，', '。', '？', '!', '；']:
        idx = tokenizer.convert_tokens_to_ids([ch])[0]
        predicted_prob = predictions[0, masked_index, idx].item()
        #print(ch, predicted_prob)
        if predicted_prob > max_prob:
            max_prob = predicted_prob
            sel_ch = ch
    if max_prob<threshold:
        res.append([sts, max_prob])

res.sort(key = lambda x: x[1])

for sts in res:
    print(sts)