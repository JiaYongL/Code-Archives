import pickle
import random
import json, os
from transformers import BertTokenizer
import numpy as np
def get_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    return tokenizer


class data_sampler(object):

    def __init__(self, args, seed=None):
        self.set_path(args)
        self.args = args
        temp_name = [args.dataname, args.seed]
        file_name = "{}.pkl".format(
                    "-".join([str(x) for x in temp_name])
                )
        mid_dir = "datasets/"
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)
        for temp_p in ["_process_path"]:
            mid_dir = os.path.join(mid_dir, temp_p)
            if not os.path.exists(mid_dir):
                os.mkdir(mid_dir)
        self.save_data_path = os.path.join(mid_dir, file_name)

        self.tokenizer = get_tokenizer(args)

        # read relation data
        self.id2rel, self.rel2id = self._read_relations(args.relation_file)

        # random sampling
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

        # regenerate data
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.args.data_file)

        # generate the task number
        self.batch = 0
        self.task_length = len(self.id2rel) // self.args.rel_per_task

        # record relations
        self.seen_relations = []
        self.history_test_data = {}
    def set_path(self, args):
        use_marker = ""
        if args.dataname in ['FewRel']:
            args.data_file = os.path.join(args.data_path,"data_with{}_marker.json".format(use_marker))
            args.relation_file = os.path.join(args.data_path, "id2rel.json")
            args.num_of_relation = 80
            args.num_of_train = 420
            args.num_of_val = 140
            args.num_of_test = 140
        elif args.dataname in ['TACRED']:
            args.data_file = os.path.join(args.data_path,"data_with{}_marker_tacred.json".format(use_marker))
            args.relation_file = os.path.join(args.data_path, "id2rel_tacred.json")
            args.num_of_relation = 40
            args.num_of_train = 420
            args.num_of_val = 140
            args.num_of_test = 140

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch == self.task_length:
            raise StopIteration()

        indexs = self.shuffle_index[self.args.rel_per_task*self.batch: self.args.rel_per_task*(self.batch+1)]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}
        
        for index in indexs:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])
            cur_training_data[self.id2rel[index]] = self.training_dataset[index]
            cur_valid_data[self.id2rel[index]] = self.valid_dataset[index]
            cur_test_data[self.id2rel[index]] = self.test_dataset[index]
            self.history_test_data[self.id2rel[index]] = self.test_dataset[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations

    def _read_data(self, file):
        if os.path.isfile(self.save_data_path):
            with open(self.save_data_path, 'rb') as f:
                datas = pickle.load(f)
            train_dataset, val_dataset, test_dataset = datas
            return train_dataset, val_dataset, test_dataset
        else:
            data = json.load(open(file, 'r', encoding='utf-8'))
            train_dataset = [[] for i in range(self.args.num_of_relation)]
            val_dataset = [[] for i in range(self.args.num_of_relation)]
            test_dataset = [[] for i in range(self.args.num_of_relation)]
            for relation in data.keys():
                rel_samples = data[relation]
                if self.seed != None:
                    random.seed(self.seed)
                random.shuffle(rel_samples)
                count = 0
                count1 = 0
                for i, sample in enumerate(rel_samples):
                    tokenized_sample = {}
                    tokenized_sample['relation'] = self.rel2id[sample['relation']]
                    tokenized_sample['tokens'] = self.tokenizer.encode(' '.join(sample['tokens']),
                                                                    padding='max_length',
                                                                    truncation=True,
                                                                    max_length=self.args.max_length)
                    if self.args.task_name == 'FewRel':
                        if i < self.args.num_of_train:
                            train_dataset[self.rel2id[relation]].append(tokenized_sample)
                        elif i < self.args.num_of_train + self.args.num_of_val:
                            val_dataset[self.rel2id[relation]].append(tokenized_sample)
                        else:
                            test_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        if i < len(rel_samples) // 5 and count <= 40:
                            count += 1
                            test_dataset[self.rel2id[relation]].append(tokenized_sample)
                        else:
                            count1 += 1
                            train_dataset[self.rel2id[relation]].append(tokenized_sample)  
                            if count1 >= 320:
                                break
            with open(self.save_data_path, 'wb') as f:
                pickle.dump((train_dataset, val_dataset, test_dataset), f)
            return train_dataset, val_dataset, test_dataset

    def _read_relations(self, file):
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        rel2id = {}
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id


class data_sampler_CFRL(object):
    def __init__(self, config=None, seed=None):
        self.config = config
        # self.qid2desc = json.load(open('qid2desc.json'))
        
        self.max_length = self.config.max_length
        self.task_length = self.config.task_length
        # self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_tokens = ['[E0]', '[E1]', '[E2]', '[E3]']
        if self.config.use_unused == 1:
            self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_prompt_tokens = []
        if self.config.pattern == 'softprompt' or self.config.pattern == 'hybridprompt':
            self.unused_prompt_tokens = ['[V0]', '[V1]', '[V2]', '[V3]']
            if self.config.use_unused == 1:
                self.unused_prompt_tokens = ['[unused4]', '[unused5]', '[unused6]', '[unused7]']
        self.unused_token = '[unused0]'
        if self.config.model == 'bert':
            self.mask_token = '[MASK]' 
            model_path = self.config.bert_path
            tokenizer_from_pretrained = BertTokenizer.from_pretrained
        elif self.config.model == 'roberta':
            self.mask_token = '<mask>' 
            model_path = self.config.roberta_path
            tokenizer_from_pretrained = RobertaTokenizer.from_pretrained

        # tokenizer
        if config.pattern == 'marker' or config.pattern == 'entity_marker':
            self.tokenizer = tokenizer_from_pretrained(model_path, \
            additional_special_tokens=self.unused_tokens)
            self.config.h_ids = self.tokenizer.get_vocab()[self.unused_tokens[0]]
            self.config.t_ids = self.tokenizer.get_vocab()[self.unused_tokens[2]]
        elif config.pattern == 'hardprompt' or config.pattern == 'cls':
            self.tokenizer = tokenizer_from_pretrained(model_path, \
            additional_special_tokens=self.unused_tokens)
            # self.tokenizer = tokenizer_from_pretrained(model_path)
        elif config.pattern == 'softprompt' or config.pattern == 'hybridprompt':
            self.tokenizer =tokenizer_from_pretrained(model_path, \
            additional_special_tokens=self.unused_tokens + self.unused_prompt_tokens)
            self.config.prompt_token_ids = self.tokenizer.get_vocab()[self.unused_token]
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.unused_tokens + self.unused_prompt_tokens})
        self.config.vocab_size = len(self.tokenizer)
        self.config.sep_token_ids = self.tokenizer.get_vocab()[self.tokenizer.sep_token]
        self.config.mask_token_ids = self.tokenizer.get_vocab()[self.tokenizer.mask_token]
        self.sep_token_ids, self.mask_token_ids =  self.config.sep_token_ids, self.config.mask_token_ids

        # read relations
        self.id2rel, self.rel2id = self._read_relations(self.config.relation_name)
        self.config.num_of_relation = len(self.id2rel)

        # if self.config.do_pre_train == 1:
        #     if self.config.task_name == 'FewRel':
        #         self.all_data = self._read_all_data('data/CFRLFewRel/train_wiki.json')
        #     else:
        #         self.all_data = self._read_all_data('data/CFRLFewRel/train_wiki.json')
        # read data
        self.training_data = self._read_data(self.config.training_data, self._temp_datapath('train'))
        self.valid_data = self._read_data(self.config.valid_data, self._temp_datapath('valid'))
        self.test_data = self._read_data(self.config.test_data, self._temp_datapath('test'))

        # read relation order
        rel_index = np.load(self.config.rel_index)
        rel_cluster_label = np.load(self.config.rel_cluster_label)
        self.cluster_to_labels = {}
        for index, i in enumerate(rel_index):
            if rel_cluster_label[index] in self.cluster_to_labels.keys():
                self.cluster_to_labels[rel_cluster_label[index]].append(i-1)
            else:
                self.cluster_to_labels[rel_cluster_label[index]] = [i-1]

        # shuffle task order
        self.seed = seed
        self.set_seed(self.seed)

        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)

        self.batch = 0

        # record relations
        self.seen_relations = []
        self.history_test_data = {}

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)
        
        print(f"Shuffle index: {self.shuffle_index}")

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch == self.task_length:
            raise StopIteration()
        
        indexs = self.cluster_to_labels[self.shuffle_index[self.batch]]
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

        return cur_training_data, cur_valid_data, cur_test_data, current_relations,\
            self.history_test_data, self.seen_relations
    
    
    def _read_all_data(self, file):
        path = f'{file}_{self.config.pretrain_mode}_{self.config.pretrain_add_entity_marker}_save.pkl'
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                all_data = pickle.load(f)
        else:
            all_data = []
            data = json.load(open(file, 'r', encoding='utf-8'))
            for relation in data.keys():
                rel_samples = data[relation]
                for i, sample in enumerate(rel_samples):
                    tokenized_sample = {}
                    tokenized_sample['relation'] = 0
                    tokenized_sample['tokens'] = ' '.join(sample['tokens'])
                    tokenized_sample['h'] = sample['h']
                    tokenized_sample['t'] = sample['t']
                    tokenized_sample['index'] = []
                    if self.config.pretrain_add_entity_marker == 1 and self.config.pretrain_mode.find('marker') == -1:
                        tokenized_sample['tokens'] = self.add_entity_marker(tokenized_sample)
                    all_data.append(self.tokenize_pre(tokenized_sample))

            with open(path, 'wb') as f:
                pickle.dump(all_data, f)
        return all_data

    def _temp_datapath(self, data_type):
        '''
            data_type = 'train'/'valid'/'test'
        '''
        temp_name = [data_type]
        file_name = '{}.pkl'.format('-'.join([str(x) for x in temp_name]))
        prompt_len = self.config.prompt_len * self.config.prompt_num
        if self.config.model == 'bert':
            tp1 = '_process_BERT_'
        elif self.config.model == 'roberta':
            tp1 = '_process_Roberta_'
        if self.config.task_name == 'FewRel':
            tp2 = 'CFRLFewRel/CFRLdata_10_100_10_'
        else:
            tp2 = 'CFRLTacred/CFRLdata_6_100_5_'
        if self.config.pattern == 'hardprompt':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1  + self.config.pattern)
        elif self.config.pattern == 'softprompt' or self.config.pattern == 'hybridprompt':                
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1 + self.config.pattern + '_' + str(prompt_len) + 'token')
        elif self.config.pattern == 'cls':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1 + self.config.pattern)            
        elif self.config.pattern == 'marker' or self.config.pattern == 'entity_marker':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k),  \
            tp1 + self.config.pattern)      
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)
        save_data_path = os.path.join(mid_dir, file_name)   
        return save_data_path     

    def _read_data(self, file, save_data_path):
        # if os.path.isfile(save_data_path):
        #     with open(save_data_path, 'rb') as f:
        #         datas = pickle.load(f)
        #         print(save_data_path)
        #     return datas
        # else:
            samples = []
            with open(file) as f:
                for i, line in enumerate(f):
                    sample = {}
                    items = line.strip().split('\t')
                    if (len(items[0]) > 0):
                        sample['relation'] = int(items[0]) - 1
                        sample['index'] = i
                        if items[1] != 'noNegativeAnswer':
                            candidate_ixs = [int(ix) for ix in items[1].split()]
                            sample['tokens'] = items[2].split()
                            headent = items[3]
                            headidx = [[int(ix) for ix in items[4].split()]]
                            tailent = items[5]
                            tailidx = [[int(ix) for ix in items[6].split()]]
                            headid = items[7]
                            tailid = items[8]
                            sample['h'] = [headent, headid, headidx]
                            sample['t'] = [tailent, tailid, tailidx]
                            if self.config.add_entity_marker == 1 and self.config.pattern.find('marker') == -1:
                                sample['tokens'] = self.add_entity_marker(sample)
                            samples.append(sample)

            read_data = [[] for i in range(self.config.num_of_relation)]
            for sample in samples:
                tokenized_sample = self.tokenize(sample)
                read_data[tokenized_sample['relation']].append(tokenized_sample)
            # with open(save_data_path, 'wb') as f:
            #     pickle.dump(read_data, f)
            #     print(save_data_path)
            return read_data
    
    def add_entity_marker(self, sample):
        new_headent = self.unused_tokens[0] + ' ' + sample['h'][0] + ' ' + self.unused_tokens[1]
        new_tailent = self.unused_tokens[2] + ' ' + sample['t'][0] + ' ' + self.unused_tokens[3]
        
        head_start, head_end = sample['h'][2][0][0], sample['h'][2][0][-1] + 1
        tail_start, tail_end = sample['t'][2][0][0], sample['t'][2][0][-1] + 1
        
        if head_start < tail_start:
            parts = [sample['tokens'][:head_start], new_headent, sample['tokens'][head_end:tail_start], new_tailent, sample['tokens'][tail_end:]]
        else:
            parts = [sample['tokens'][:tail_start], new_tailent, sample['tokens'][tail_end:head_start], new_headent, sample['tokens'][head_end:]]
        merged_parts = []
        for part in parts:
            if isinstance(part, list):
                merged_parts.extend(part)
            else:
                merged_parts.append(part)
        return " ".join(merged_parts)
    
    def tokenize(self, sample):
        tokenized_sample = {}
        tokenized_sample['relation'] = sample['relation']
        tokenized_sample['index'] = sample['index']
        # tokenized_sample['h_qid'] = sample['h'][1]
        # tokenized_sample['t_qid'] = sample['t'][1]
        # head_desc = self.qid2desc[sample['h'][1]] if sample['h'][1] in self.qid2desc else ''
        # tail_desc = self.qid2desc[sample['t'][1]] if sample['t'][1] in self.qid2desc else ''
        
        prompt = ''
        if self.config.pattern == 'hardprompt':
            prompt = self._tokenize_hardprompt(sample)
        elif self.config.pattern == 'softprompt':
            prompt = self._tokenize_softprompt(sample)   
        elif self.config.pattern == 'hybridprompt':
            prompt = self._tokenize_hybridprompt(sample)
            # head_pos = prompt.index(self.unused_prompt_tokens[1])
            # txt = prompt[:head_pos] +'(' + head_desc + ') ' + prompt[head_pos:]
            # tail_pos = txt.index(self.unused_prompt_tokens[3])
            # txt = txt[:tail_pos] +'(' + tail_desc + ') ' + txt[tail_pos:]
            # prompt = txt
        elif self.config.pattern == 'marker' or self.config.pattern == 'entity_marker':
            sample['tokens'] = self.add_entity_marker(sample)
        elif self.config.pattern == 'cls':
            sample['tokens'] = self.add_entity_marker(sample)
        
        # head_pos = sample['tokens'].index(self.unused_tokens[1])
        # txt = sample['tokens'][:head_pos] +'(' + head_desc + ') ' + sample['tokens'][head_pos:]
        # tail_pos = txt.index(self.unused_tokens[3])
        # txt = txt[:tail_pos] +'(' + tail_desc + ') ' + txt[tail_pos:]
        # sample['tokens'] = txt
        
        if prompt != '':
            out = self.tokenizer(prompt, sample['tokens'], padding='max_length', truncation=True, max_length=512)
        else:
            out = self.tokenizer(sample['tokens'], padding='max_length', truncation=True, max_length=self.max_length)
        tokenized_sample['ids'] = out['input_ids']
        tokenized_sample['mask'] = out['attention_mask']
        
        return tokenized_sample
    
    def tokenize_pre(self, sample):
        tokenized_sample = {}
        tokenized_sample['relation'] = sample['relation']
        tokenized_sample['index'] = sample['index']
        prompt = ''
        if self.config.pretrain_mode == 'hardprompt':
            prompt = self._tokenize_hardprompt(sample)
        elif self.config.pretrain_mode == 'softprompt':
            prompt = self._tokenize_softprompt(sample)   
        elif self.config.pretrain_mode == 'hybridprompt':
            prompt = self._tokenize_hybridprompt(sample)                     
        elif self.config.pretrain_mode == 'marker' or self.config.pretrain_mode == 'entity_marker':
            prompt = self._tokenize_marker(sample)
        elif self.config.pretrain_mode == 'cls':
            prompt = self._tokenize_cls(sample)
        if prompt != '':
            out = self.tokenizer(prompt, sample['tokens'], padding='max_length', truncation=True, max_length=self.max_length)   
        else:
            out = self.tokenizer(sample['tokens'], padding='max_length', truncation=True, max_length=self.max_length)
        tokenized_sample['ids'] = out['input_ids']
        tokenized_sample['mask'] = out['attention_mask']
        
        return tokenized_sample


    def _read_relations(self, file):
        id2rel, rel2id = {}, {}
        with open(file) as f:
            for index, line in enumerate(f):
                rel = line.strip()
                id2rel[index] = rel
                rel2id[rel] = index
        return id2rel, rel2id

    def _tokenize_softprompt(self, sample):
        '''
        X [v] [v] [v] [v]
        [v] = [unused0] * prompt_len
        '''
        prompt = [self.unused_prompt_tokens[0]] + [self.mask_token] + [self.unused_prompt_tokens[1]]
        return ' '.join(prompt)

    def _tokenize_hybridprompt(self, sample):
        '''
        [v0] e1 [v1] [MASK] [v2] e2 [v3] 
        '''
        h, t = sample['h'][0].split(' '),  sample['t'][0].split(' ')
        prompt = [self.unused_prompt_tokens[0]] + h + [self.unused_prompt_tokens[1]] + [self.mask_token] + [self.unused_prompt_tokens[2]] + t + [self.unused_prompt_tokens[3]]

        return ' '.join(prompt)

    def _tokenize_hardprompt(self, sample):
        '''
        X e1 [MASK] e2 
        '''
        h, t = sample['h'][0].split(' '),  sample['t'][0].split(' ')
        prompt = h + [self.mask_token] + t
        return ' '.join(prompt)

    def _tokenize_marker(self, sample):
        '''
        [unused]e[unused]
        '''
        raw_tokens = sample['tokens']
        h1, h2, t1, t2 =  sample['h'][2][0][0], sample['h'][2][0][-1], sample['t'][2][0][0], sample['t'][2][0][-1]
        new_tokens = []

        # add entities marker        
        for index, token in enumerate(raw_tokens):
            if index == h1:
                new_tokens.append(self.unused_tokens[0])
                new_tokens.append(token)
                if index == h2:
                    new_tokens.append(self.unused_tokens[1])
            elif index == h2:
                new_tokens.append(token)
                new_tokens.append(self.unused_tokens[1])
            elif index == t1:
                new_tokens.append(self.unused_tokens[2])
                new_tokens.append(token)
                if index == t2:
                    new_tokens.append(self.unused_tokens[3])
            elif index == t2:
                new_tokens.append(token)
                new_tokens.append(self.unused_tokens[3])
            else:
                new_tokens.append(token)
            
            ids = self.tokenizer.encode(' '.join(new_tokens),
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.max_length)
            
            # mask
            mask = np.zeros(self.max_length, dtype=np.int32)
            end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
            mask[:end_index + 1] = 1

        return ids, mask

    def _tokenize_cls(self, sample):
        '''
        [CLS] X
        '''
        raw_tokens = sample['tokens']
        ids = self.tokenizer.encode(' '.join(raw_tokens),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)
        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1

        return ids, mask

