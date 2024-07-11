from collections import defaultdict
import torch
from torch.utils.data import Dataset
import numpy as np
import os


class FewRel(Dataset):
    
    def __init__(self, root='', n_way='5', k_shot='10'):
        self.root = root
        self.n_way = n_way
        self.k_shot = k_shot
        self.relation = self._read_relation(self.root)
        self.data_path = os.path.join(self.root, 'CFRLdata_10_100_' + self.n_way+'_'+self.k_shot)
        self.training_data = self.load_data(os.path.join(self.data_path, 'train_0.txt'))
        self.training_data = self.load_data(os.path.join(self.data_path, 'test_0.txt'))
        self.training_data = self.load_data(os.path.join(self.data_path, 'valid_0.txt'))
        
     
    def _read_relations(self, file):
        '''
        :param file: input relation file
        :return:  a list of relations, and a mapping from relations to their ids.
        '''

        id2rel = []
        rel2id = {}
        with open(file) as file_in:
            for line in file_in:
                id2rel.append(line.strip())
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id
    
    def load_data(self, file):
        samples = []
        with open(file) as file_in:
            for line in file_in:
                items = line.strip().split('\t')
                if (len(items[0]) > 0):
                    relation_ix = int(items[0])
                    if items[1] != 'noNegativeAnswer':
                        candidate_ixs = [int(ix) for ix in items[1].split()]
                        sentence = items[2].split('\n')[0]
                        headent = items[3]
                        headidx = [int(ix) for ix in items[4].split()]
                        tailent = items[5]
                        tailidx = [int(ix) for ix in items[6].split()]
                        headid = items[7]
                        tailid = items[8]
                        samples.append(
                            [relation_ix, candidate_ixs, sentence, headent, headidx, tailent, tailidx,
                             headid, tailid])
        read_data = defaultdict(list)
        for sample in samples:
            text = sample[2]
            split_text = text.split(" ")
            new_headent = ' [E11] ' + sample[3] + ' [E12] '
            new_tailent = ' [E21] ' + sample[5] + ' [E22] '
            if sample[4][0] < sample[6][0]:
                new_text = " ".join(split_text[0:sample[4][0]]) + new_headent + " ".join(
                    split_text[sample[4][-1] + 1:sample[6][0]]) \
                           + new_tailent + " ".join(split_text[sample[6][-1] + 1:len(split_text)])
            else:
                new_text = " ".join(split_text[0:sample[6][0]]) + new_tailent + " ".join(
                    split_text[sample[6][-1] + 1:sample[4][0]]) \
                           + new_headent + " ".join(split_text[sample[4][-1] + 1:len(split_text)])

            tokenized_sample = {}
            tokenized_sample['relation'] = sample[0] - 1
            tokenized_sample['neg_labels'] = [can_idx - 1 for can_idx in sample[1]]
            tokenized_sample['tokens'] = self.tokenizer.encode(new_text,
                                                               padding='max_length',
                                                               truncation=True,
                                                               max_length=self.config.max_length)
            self.id2sent[len(self.id2sent)] = tokenized_sample['tokens']
            read_data[tokenized_sample['relation']].append(tokenized_sample)
        return read_data
    
    
if __name__== '__main__':
    root = '../../data/'
    dataset = FewRel(root=root, n_way='5', k_shot='10')
    