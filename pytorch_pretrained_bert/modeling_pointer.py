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
        # encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1 (unsqueezed), hidden_size)
        # decoder_transform = self.W2(decoder_state)

        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        # logits = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

        logits = torch.bmm(decoder_state, self.W1(encoder_outputs).transpose(1,2)).squeeze()
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
        self.num_lstm_layers = 1
        self.encoder_rnn = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                                   num_layers=self.num_lstm_layers,
                                   batch_first=True, bidirectional=False)
        self.decoder_rnn = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                                   num_layers=self.num_lstm_layers,
                                   batch_first=True, bidirectional=False)
        self.tgt_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.tgt_embeddings.weight = copy.deepcopy(self.bert.embeddings.word_embeddings.weight)
        self.softmax = nn.Softmax(dim=-1)
        self.W1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.W2 = nn.Linear(config.hidden_size, config.hidden_size)


        for p in self.encoder_rnn.parameters():
            if p.dim() > 1:
                torch.nn.init.uniform_(p, -0.1, 0.1)

        for p in self.decoder_rnn.parameters():
            if p.dim() > 1:
                torch.nn.init.uniform_(p, -0.1, 0.1)
        #
        # for p in self.attn.parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)

        # self.decoder_hidden = nn.Parameter(torch.randn((self.num_lstm_layers, 1, self.hidden_size)))
        # self.decoder_context = nn.Parameter(torch.randn((self.num_lstm_layers, 1, self.hidden_size)))

    def forward(self, input_ids, input_mask, segment_ids, tgt_ids, tgt_mask, ext_input_ids, ext_tgt_ids,
                vocab_size, device, train=True):
        # with torch.no_grad():
        #     sequence_output, _ = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        # print(torch.mean(sequence_output), torch.std(sequence_output))
        # sequence_output = self.dropout(sequence_output)

        sequence_output = self.tgt_embeddings(input_ids)
        sequence_output, _ = self.encoder_rnn(sequence_output)

        batch_size = input_ids.shape[0]
        max_src_length = input_ids.shape[1]
        max_tgt_len = tgt_ids.shape[1]

        # Lets use zeros as an intial input for sorting example
        decoder_hidden = (torch.zeros((self.num_lstm_layers, batch_size, self.hidden_size)).to(device),
                          torch.zeros((self.num_lstm_layers, batch_size, self.hidden_size)).to(device))
        # encoder_hidden = sequence_output[:, 0].unsqueeze(0).expand((self.num_lstm_layers, batch_size, self.hidden_size)).detach()
        # decoder_hidden = (copy.deepcopy(encoder_hidden), copy.deepcopy(encoder_hidden))

        seq_logits = []
        seq_probs = []
        seq_argmaxs = []

        decoder_input = torch.mean(sequence_output, dim=1).unsqueeze(1)
        # self.decoder_hidden = self.decoder_hidden.repeat((self.num_lstm_layers, batch_size, self.hidden_size))
        # self.decoder_context = self.decoder_context.repeat((self.num_lstm_layers, batch_size, self.hidden_size))
        # decoder_hidden = (self.decoder_hidden, self.decoder_context)

        for i in range(max_tgt_len):
            # We will simply mask out when calculating attention or max (and loss later)
            # not all input and hiddens, just for simplicity

            # h, c: (num_layers, batch_size, hidden_size)
            decoder_output, (h_i, ci) = self.decoder_rnn(decoder_input, decoder_hidden)
            # print(i, torch.std_mean(decoder_output))
            decoder_hidden = (h_i, ci)

            # Get a pointer distribution over the encoder outputs using attention
            # (batch_size, max_seq_len)
            logits = self.attn(decoder_output, sequence_output)
            if torch.__version__ >= '1.2':
                masked_logits = logits.masked_fill((1 - input_mask).bool(), -1e7)
            else:
                masked_logits = logits.masked_fill((1 - input_mask).byte(), -1e7)
            prob_a = self.softmax(masked_logits)
            context = torch.bmm(prob_a.unsqueeze(1), sequence_output).squeeze()

            seq_logits.append(masked_logits)
            seq_probs.append(prob_a)

            masked_argmax = torch.argmax(prob_a, dim=1)

            seq_argmaxs.append(masked_argmax)

            # (batch_size, hidden_size)
            if train:
                decoder_input = self.tgt_embeddings(tgt_ids[:, i].unsqueeze(1))
                # decoder_input = self.W1(decoder_input) + self.W2(context.unsqueeze(1))
            else:
                decoder_input = torch.gather(input_ids, dim=1, index=masked_argmax.unsqueeze(1))
                decoder_input = self.tgt_embeddings(decoder_input)
                # decoder_input = self.W1(decoder_input) + self.W2(context.unsqueeze(1))

        seq_logits = torch.stack(seq_logits, 1)
        seq_probs = torch.stack(seq_probs, 1)
        seq_argmaxs = torch.stack(seq_argmaxs, 1)

        coverage_losses = [torch.zeros(batch_size)]
        for i in range(1, max_tgt_len):
            coverage_vector = torch.sum(seq_probs[:, :i], dim=1)
            coverage_losses.append(torch.sum(torch.min(seq_probs[:, i], coverage_vector), dim=1))
        coverage_losses = torch.stack(coverage_losses, 1)

        vocab_dist = torch.zeros((batch_size, max_tgt_len, vocab_size), requires_grad=False).to(device)
        vocab_dist = torch.scatter_add(vocab_dist, 2, ext_input_ids.unsqueeze(1).expand_as(seq_probs), seq_probs)

        coverage_loss = torch.mean(torch.sum(coverage_losses * tgt_mask.float(), dim=1) / torch.sum(tgt_mask))
        print(coverage_loss)
        if tgt_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if tgt_mask is not None:
                active_loss = tgt_mask.view(-1) == 1
                active_logits = vocab_dist.view(-1, vocab_size)[active_loss]
                active_labels = ext_tgt_ids.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels) #+ 1 * coverage_loss
            else:
                loss = loss_fct(vocab_dist.view(-1, vocab_size), ext_tgt_ids.view(-1)) #+ 1 * coverage_loss
            return loss, vocab_dist
        else:
            return seq_logits
