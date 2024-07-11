import numpy as np
import json
import random
from transformers import BertTokenizer
import logging
from builtins import list

class data_sampler(object):

    def __init__(self, config=None, seed=None):

        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.id2rel, self.rel2id = self._read_relations(config.relation_file)
        self.id2sent = {}
        self.cur_ids = 0
        
        self.ent11 = None
        self.ent12 = None
        self.ent21 = None
        self.ent22 = None
        self.left = None
        self.right = None
        self.left1 = None
        self.left2 = None
        self.right1 = None
        self.right2 = None
        self.special_tokens = []
        
        if config.use_unused:
            self.fake_token = 'unused'
        else:
            self.fake_token = 'E'
        self.training_data = self.load_data(config.training_file)
        self.valid_data = self.load_data(config.valid_file)
        self.test_data = self.valid_data
        
        self.task_length = config.task_length
        rel_index = np.load(config.rel_index)
        rel_cluster_label = np.load(config.rel_cluster_label)
        self.cluster_to_labels = {}
        for index, i in enumerate(rel_index):
            if rel_cluster_label[index] in self.cluster_to_labels.keys():
                self.cluster_to_labels[rel_cluster_label[index]].append(i-1)
            else:
                self.cluster_to_labels[rel_cluster_label[index]] = [i-1]

        self.seed = seed
        if self.seed != None:
            self.set_seed(self.seed)

        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)

        self.batch = 0

        # record relations
        self.seen_relations = []
        self.history_test_data = {}
        
        

    def read_text(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip()
        return lines

    def generate_fake_tokens(self, num=1):
        tokens = ['[' + self.fake_token + str(i) + ']' for i in range(self.cur_ids, self.cur_ids + num)]
        self.cur_ids += num
        # self.tokenizer.add_tokens(tokens)
        self.special_tokens.extend(tokens)
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.special_tokens})
        return ' '.join(tokens)
    
    def process_sample(self, sample):
        text = sample[2]
        split_text = text.split(" ")
        new_headent = sample[3]
        new_tailent = sample[5]
        if self.config.use_entity_marker:
            if self.ent11 == None:
                self.ent11 = self.generate_fake_tokens()
                self.ent12 = self.generate_fake_tokens()
                self.ent21 = self.generate_fake_tokens()
                self.ent22 = self.generate_fake_tokens()
            new_headent = ' '.join([self.ent11, sample[3], self.ent12])
            new_tailent = ' '.join([self.ent21, sample[5], self.ent22])

        head_start, head_end = sample[4][0], sample[4][-1] + 1
        tail_start, tail_end = sample[6][0], sample[6][-1] + 1

        if head_start < tail_start:
            parts = [split_text[:head_start], new_headent, split_text[head_end:tail_start], new_tailent, split_text[tail_end:]]
        else:
            parts = [split_text[:tail_start], new_tailent, split_text[tail_end:head_start], new_headent, split_text[head_end:]]

        merged_parts = []
        for part in parts:
            if isinstance(part, list):
                merged_parts.extend(part)
            else:
                merged_parts.append(part)
        new_text = " ".join(merged_parts)
        
        prompt = ''
        if self.config.pattern == 'softprompt':
            if self.left == None:
                self.left = self.generate_fake_tokens(self.config.prompt_length)
                self.right = self.generate_fake_tokens(self.config.prompt_length)
            prompt = ' '.join([self.left,'[MASK]', self.right])
        elif self.config.pattern == 'hardprompt':
            template = self.config.template.replace('[E1]', sample[3])
            prompt = template.replace('[E2]', sample[5])
        elif self.config.pattern == 'hybridprompt':
            if self.left1 == None:
                self.left1 = self.generate_fake_tokens(self.config.prompt_length)
                self.left2 = self.generate_fake_tokens(self.config.prompt_length)
                self.right1 = self.generate_fake_tokens(self.config.prompt_length)
                self.right2 = self.generate_fake_tokens(self.config.prompt_length)
            prompt = ' '.join([self.left1, sample[3], self.left2,'[MASK]', self.right1, sample[5], self.right2])
        
        # tokenized_sample = {
        #     'relation': sample[0] - 1,
        #     'neg_labels': [can_idx - 1 for can_idx in sample[1]],
        #     'tokens': self.tokenizer.encode(new_text, padding='max_length', truncation=True, max_length=self.config.max_length)
        # }
        if prompt != '':
            tokenized_sample = self.tokenizer.encode_plus(new_text, prompt, padding='max_length', truncation=True, max_length=self.config.max_length)
        else:
            tokenized_sample = self.tokenizer.encode_plus(new_text, padding='max_length', truncation=True, max_length=self.config.max_length)
            
        tokenized_sample['relation'] = sample[0] - 1
        
        if self.config.pattern == 'entity_marker':
            tokenized_sample['pos'] = []
            tokenized_sample['pos'].append(tokenized_sample['input_ids'].index(self.tokenizer.convert_tokens_to_ids(self.ent11)))
            tokenized_sample['pos'].append(tokenized_sample['input_ids'].index(self.tokenizer.convert_tokens_to_ids(self.ent21)))
        elif self.config.pattern.find('prompt') != -1:
            tokenized_sample['pos'] = []
            tokenized_sample['pos'].append(tokenized_sample['input_ids'].index(self.tokenizer.convert_tokens_to_ids('[MASK]')))
        else:
            tokenized_sample['pos'] = [0]
        
        return tokenized_sample
    
    def load_data(self, file):
        samples = []
        for line in self.read_text(file):
            items = line.strip().split('\t')
            if (len(items[0]) > 0):
                relation_ix = int(items[0])
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
        read_data = [[] for i in range(self.config.num_of_relation)]

        for sample in samples:
            tokenized_sample = self.process_sample(sample)
            # tokenized_sample['ind'] = len(self.id2sent)
            self.id2sent[len(self.id2sent)] = tokenized_sample['input_ids']
            read_data[tokenized_sample['relation']].append(tokenized_sample)
        return read_data

    def set_seed(self, seed):
        print('set seed:', seed)
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)

        indices = list(range(self.task_length - 1))
        random.shuffle(indices)

        sorted_indices = np.argsort(indices)
        self.shuffle_index = np.insert(sorted_indices, 0, self.task_length - 1)

        print(f"Shuffle index: {self.shuffle_index}")


    def _read_relations(self, file):
        '''
        :param file: input relation file
        :return:  a list of relations, and a mapping from relations to their ids.
        '''

        id2rel = []
        rel2id = {}
        id2rel = self.read_text(file)
        for i, rel in enumerate(id2rel):
            rel2id[rel] = i
        return id2rel, rel2id

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch == self.task_length:
            self.batch = 0
            raise StopIteration()

        indexs = self.cluster_to_labels[self.shuffle_index[self.batch]]  # 每个任务出现的id
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indexs:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])

            cur_training_data[self.id2rel[index]] = self.training_data[index]
            cur_valid_data[self.id2rel[index]] = self.valid_data[index]
            cur_test_data[self.id2rel[index]] = self.test_data[index]
            self.history_test_data[self.id2rel[index]] = self.test_data[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations

    def get_id2sent(self):
        return self.id2sent
    