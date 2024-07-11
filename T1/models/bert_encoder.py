import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from models.base_model import base_model
from transformers import BertModel, BertConfig


class Bert_Encoder(base_model):

    def __init__(self, config):
        super(Bert_Encoder, self).__init__()

        # load model
        self.encoder = BertModel.from_pretrained(config.bert_path).to(config.device)
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # the dimension for the final outputs
        self.output_size = config.encoder_output_size
        self.drop = nn.Dropout(config.drop_out)


        self.pattern = config.pattern
        # find which encoding is used
        # if config.pattern in ['standard', 'entity_marker', 'prompt']:
        #     self.pattern = config.pattern
        # else:
        #     raise Exception('Wrong encoding.')
        config.hidden_size = self.bert_config.hidden_size
        config.output_size = config.encoder_output_size
        # self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size)
        self.encoder.resize_token_embeddings(config.vocab_size + 20)
        if self.pattern == 'entity_marker':
            self.linear_transform = nn.Linear(self.bert_config.hidden_size * 2, self.output_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)
        self.layer_normalization = nn.LayerNorm([self.output_size])

    def forward(self, inputs):
        '''
        :param inputs: of dimension [B, N]
        :return: a result of size [B, H*2] or [B, H], according to different strategy
        '''
        # generate representation under a certain encoding strategy
        if self.pattern == 'standard':
            # in the standard mode, the representation is generated according to
            # the representation of[CLS] mark.

            output = self.encoder(inputs)[1]
        elif self.pattern == 'entity_marker':
            # for each sample in the batch, acquire the positions of its [E11] and [E21]
            e11 = (inputs == 30522).nonzero(as_tuple=True)[1] # [B, 1]
            e21 = (inputs == 30524).nonzero(as_tuple=True)[1] # [B, 1]

            # input the sample to BERT
            tokens_output = self.encoder(inputs)[0] # [B,N] --> [B,N,H]

            # for each sample in the batch, acquire its representations for [E11] and [E21]
            indices = torch.stack([e11, e21], dim=1) # [B, 2]
            output = torch.gather(tokens_output, 1, indices.unsqueeze(2).expand(-1, -1, tokens_output.size(-1)))  # [B, 2, H]

            # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
            output = output.view(output.size()[0], -1)  # [B, H*2]

            # the output dimension is [B, H*2], B: batchsize, H: hiddensize
            # output = self.drop(output)
            # output = self.linear_transform(output)
            # output = F.gelu(output)
            # output = self.layer_normalization(output)
        else:
            mask_pos = inputs.eq(103).nonzero() # [B, 1]
            tokens_output = self.encoder(inputs)[0]
            output = tokens_output[mask_pos[:, 0], mask_pos[:, 1], :] # [B,N,H] --> [B,H]
        return output