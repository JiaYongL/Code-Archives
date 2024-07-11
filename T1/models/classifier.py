from torch import nn, optim
from models.base_model import base_model
import torch
import torch.nn.functional as F
import math


class Softmax_Layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, input_size, num_class, prev=None):
        """
        Args:
            num_class: number of classes
        """
        super(Softmax_Layer, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.fc = nn.Linear(self.input_size, self.num_class, bias=False)
        self.prev_fc = prev

    # def updata_fc(self, softmax_layer):
    #     prev_len = softmax_layer.fc.weight.data.shape[0]
    #     self.fc.weight.data[:prev_len] = softmax_layer.fc.weight.data.clone()
    
    def forward(self, input):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """

        logits = self.fc(input)
        if self.prev_fc is not None:
            prev_logits = self.prev_fc(input)
            logits = torch.cat([logits, prev_logits], dim=1)
        
        return logits


class Proto_Softmax_Layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """
    def __init__(self, config):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super(Proto_Softmax_Layer, self).__init__()
        self.config = config
        self.prototypes = None

    def set_prototypes(self, protos):
        self.prototypes = protos.to(self.config.device)

    def update_prototypes(self, protos, idx):
        self.prototypes[idx] = protos.to(self.config.device)
    
    def expand_prototypes(self, protos):
        if self.prototypes is None:
            self.prototypes = protos.detach().to(self.config.device)
        else:
            self.prototypes = torch.cat([self.prototypes, protos.detach().to(self.config.device)], dim=0)

    def forward(self, rep):

        dis_mem = self.__distance__(rep, self.prototypes)
        return dis_mem

    def __distance__(self, rep, rel):
        '''
        rep_ = rep.view(rep.shape[0], 1, rep.shape[-1])
        rel_ = rel.view(1, -1, rel.shape[-1])
        dis = (rep_ * rel_).sum(-1)
        return dis
        '''
        rep_norm = rep / rep.norm(dim=1)[:, None]
        rel_norm = rel / rel.norm(dim=1)[:, None]

        res = torch.mm(rep_norm, rel_norm.transpose(0, 1))
        return res


class MYNET(nn.Module):

    def __init__(self, args, mode='cos'):
        super().__init__()
        self.mode = mode
        self.args = args
        self.num_features = 768
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

    def forward_metric(self, x):
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
            # self.fc.data.weight = new_fc
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def soft_calibration(self, args, session):
        base_protos = self.fc.weight.data[:args.base_class].detach().cpu().data
        base_protos = F.normalize(base_protos, p=2, dim=-1)
        
        cur_protos = self.fc.weight.data[args.base_class + (session-1) * args.way : args.base_class + session * args.way].detach().cpu().data
        cur_protos = F.normalize(cur_protos, p=2, dim=-1)
        
        weights = torch.mm(cur_protos, base_protos.T) * args.softmax_t
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)

        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        
        updated_protos = (1-args.shift_weight) * cur_protos + args.shift_weight * delta_protos

        self.fc.weight.data[args.base_class + (session-1) * args.way : args.base_class + session * args.way] = updated_protos