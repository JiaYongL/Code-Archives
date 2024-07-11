import torch.nn as nn
import torch
import numpy as np
from transformers import BertModel, BertConfig

class Bert_Encoder(nn.Module):

    def __init__(self, config, out_token=False):
        super(Bert_Encoder, self).__init__()

        # load model
        self.encoder = BertModel.from_pretrained(config.bert_path).cuda()
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # the dimension for the final outputs
        self.output_size = config.encoder_output_size
        self.out_dim = self.output_size

        # # find which encoding is used
        # if config.pattern in ['standard', 'entity_marker']:
        #     self.pattern = config.pattern
        # else:
        #     raise Exception('Wrong encoding.')
        self.pattern = config.pattern
        if config.use_unused == 0:
            self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size + config.prompt_size)
        
        if config.pattern == 'entity_marker':
            # self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size*2, self.output_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)

        self.layer_normalization = nn.LayerNorm([self.output_size])


    def get_output_size(self):
        return self.output_size

    def forward(self, inputs, attention_mask=None):
        # generate representation under a certain encoding strategy
        if self.pattern == 'standard':
            # in the standard mode, the representation is generated according to
            #  the representation of[CLS] mark.
            output = self.encoder(input_ids=inputs, attention_mask=attention_mask)[1]
        elif self.pattern.find('prompt') != -1:
            tokens_output = self.encoder(input_ids=inputs, attention_mask=attention_mask)[0]
            mask_pos = torch.tensor([np.argwhere(inputs[i].cpu().numpy() == 103)[0][0] for i in range(inputs.size()[0])])
            output = tokens_output[torch.arange(tokens_output.size(0)), mask_pos]
            output = self.linear_transform(output)
            
        else:
            # in the entity_marker mode, the representation is generated from the representations of
            #  marks [E11] and [E21] of the head and tail entities.
            e11 = []
            e21 = []
            # for each sample in the batch, acquire the positions of its [E11] and [E21]
            head_id = 30522
            tail_id = 30524
            if self.config.use_unused == 1:
                head_id = 1
                tail_id = 3
            for i in range(inputs.size()[0]):
                tokens = inputs[i].cpu().numpy()
                e11.append(np.argwhere(tokens == head_id)[0][0])
                e21.append(np.argwhere(tokens == tail_id)[0][0])

            # input the sample to BERT
            tokens_output = self.encoder(input_ids=inputs, attention_mask=attention_mask)[0] # [B,N] --> [B,N,H]
            output = []
            # for each sample in the batch, acquire its representations for [E11] and [E21]
            for i in range(len(e11)):
                if inputs.device.type in ['cuda']:
                    instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
                    instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
                else:
                    instance_output = torch.index_select(tokens_output, 0, torch.tensor(i))
                    instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]))
                output.append(instance_output) # [B,N] --> [B,2,H]
            # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
            output = torch.cat(output, dim=0)
            output = output.view(output.size()[0], -1) # [B,N] --> [B,H*2]
            
            output = self.linear_transform(output)


        return output