# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import copy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from .modeling import BertModel, BertPreTrainedModel

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1 (unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state)

        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        logits = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

        return logits


class BertForLSTMPointer(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForLSTMPointer, self).__init__(config)
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.attn = Attention(self.hidden_size)
        self.apply(self.init_bert_weights)
        self.num_lstm_layers = 2
        self.rnn = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=self.num_lstm_layers,
                           batch_first=True, bidirectional=False)
        self.tgt_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.tgt_embeddings.weight = copy.deepcopy(
            self.bert.embeddings.word_embeddings.weight
        )


    def forward(self, input_ids, input_mask, segment_ids, tgt_ids, tgt_mask, labels, train=True):
        sequence_output, _ = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        # sequence_output = self.dropout(sequence_output)

        batch_size = input_ids.shape[0]
        max_seq_len = input_ids.shape[1]

        # Lets use zeros as an intial input for sorting example
        # decoder_hidden = (torch.zeros((self.num_lstm_layers, batch_size, self.hidden_size)),
        #                   torch.zeros((self.num_lstm_layers, batch_size, self.hidden_size)))
        encoder_hidden = sequence_output[:, 0].unsqueeze(0).expand((self.num_lstm_layers, batch_size, self.hidden_size)).detach()
        decoder_hidden = (copy.deepcopy(encoder_hidden), copy.deepcopy(encoder_hidden))

        seq_logits = []
        seq_argmaxs = []

        decoder_input = sequence_output[:, 0].unsqueeze(1)
        for i in range(max_seq_len):
            # We will simply mask out when calculating attention or max (and loss later)
            # not all input and hiddens, just for simplicity

            # h, c: (batch_size, hidden_size)
            decoder_output, (h_i, ci) = self.rnn(decoder_input, decoder_hidden)
            decoder_hidden = (h_i, ci)

            # Get a pointer distribution over the encoder outputs using attention
            # (batch_size, max_seq_len)
            logits = self.attn(decoder_output, sequence_output)
            seq_logits.append(logits)

            masked_logits = logits.masked_fill((1 - input_mask).bool(), -1e7)
            masked_argmax = torch.argmax(masked_logits, dim=1)

            seq_argmaxs.append(masked_argmax)

            # (batch_size, hidden_size)
            if train:
                decoder_input = self.tgt_embeddings(tgt_ids[:, i].unsqueeze(1))
            else:
                decoder_input = torch.gather(input_ids, dim=1, index=masked_argmax.unsqueeze(1))
                decoder_input = self.tgt_embeddings(decoder_input)

        seq_logits = torch.stack(seq_logits, 1)
        seq_argmaxs = torch.stack(seq_argmaxs, 1)

        if tgt_ids is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # if tgt_mask is not None:
            #     active_loss = tgt_mask.view(-1) == 1
            #     active_logits = seq_logits.view(-1, self.num_labels)[active_loss]
            #     active_labels = tgt_ids.view(-1)[active_loss]
            #     loss = loss_fct(active_logits, active_labels)
            # else:
            #     loss = loss_fct(seq_logits.view(-1, self.num_labels), tgt_ids.view(-1))
            loss = loss_fct(seq_logits, labels)
            return loss, seq_logits
        else:
            return seq_logits
