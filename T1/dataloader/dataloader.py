from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

class data_set(Dataset):

    def __init__(self, data,config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        label = torch.tensor([item['relation'] for item in data])
        neg_labels = [torch.tensor(item['neg_labels']) for item in data]
        tokens = [torch.tensor(item['tokens']) for item in data]
        return (
            label,
            neg_labels,
            tokens
        )

def get_data_loader(config, data, shuffle = False, drop_last = False, batch_size = None):
    dataset = p_data_set(data, tokenizer, config)
        
    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader

# class prompt_data_set(Dataset):

#     def __init__(self, data, tokenizer, config=None):
#         self.data = data
#         self.config = config
#         self.template = config.template
#         self.tokenizer = tokenizer
#         self.tokenizer.add_special_tokens({'additional_special_tokens': ["[E11]", "[E12]", "[E21]", "[E22]",'[E1]', '[E2]']})
#         self.tokenized_template = self.tokenizer.encode(self.template)
#         self.head_pos = self.tokenized_template.index(self.tokenizer.convert_tokens_to_ids('[E1]'))
#         self.tail_pos = self.tokenized_template.index(self.tokenizer.convert_tokens_to_ids('[E2]'))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

#     def collate_fn(self, data):

#         label = torch.tensor([item['relation'] for item in data])
#         neg_labels = [torch.tensor(item['neg_labels']) for item in data]
#         tokens = []
#         for item in data:
#             head_start = item['tokens'].index(self.tokenizer.convert_tokens_to_ids('[E11]')) + 1
#             head_end = item['tokens'].index(self.tokenizer.convert_tokens_to_ids('[E12]'))
#             tail_start = item['tokens'].index(self.tokenizer.convert_tokens_to_ids('[E21]')) + 1
#             tail_end = item['tokens'].index(self.tokenizer.convert_tokens_to_ids('[E22]'))
#             prompt = (self.tokenized_template[ :self.head_pos] + item['tokens'][head_start:head_end] 
#           + self.tokenized_template[self.head_pos+1:self.tail_pos] 
#           + item['tokens'][tail_start:tail_end] + self.tokenized_template[self.tail_pos+1:])
#             new_tokens = prompt + item['tokens'][1:]
#             new_tokens = new_tokens[:self.config.max_length]
#             tokens.append(torch.tensor(new_tokens))
#         # tokens = [torch.tensor(item['tokens']) for item in data]
#         return (
#             label,
#             neg_labels,
#             tokens
#         )

class prompt_data_set(Dataset):

    def __init__(self, data, tokenizer, config=None):
        self.data = data
        self.config = config
        self.template = config.template
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [
                    "[E11]", "[E12]", "[E13]", "[E14]", "[E15]", "[E16]", "[E17]", "[E18]",
                    "[E21]", "[E22]", "[E23]", "[E24]", "[E25]", "[E26]", "[E27]", "[E28]",
                    '[E1]', '[E2]'
                ]
            })
        self.tokenized_template = self.tokenizer.encode(self.template)
        self.head_pos = self.tokenized_template.index(self.tokenizer.convert_tokens_to_ids('[E1]'))
        self.tail_pos = self.tokenized_template.index(self.tokenizer.convert_tokens_to_ids('[E2]'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):

        label = torch.tensor([item['relation'] for item in data])
        neg_labels = [torch.tensor(item['neg_labels']) for item in data]
        tokens = []
        for item in data:
            head_start = item['tokens'].index(self.tokenizer.convert_tokens_to_ids('[E11]')) + 1
            head_end = item['tokens'].index(self.tokenizer.convert_tokens_to_ids('[E12]'))
            tail_start = item['tokens'].index(self.tokenizer.convert_tokens_to_ids('[E21]')) + 1
            tail_end = item['tokens'].index(self.tokenizer.convert_tokens_to_ids('[E22]'))
            if self.config.pattern == 'soft_prompt':
                prompt = self.tokenizer.encode("[E11] [E12] [E13] [E14] [E15] [E16] [MASK] [E21] [E22] [E23] [E24] [E25] [E26]")
            elif self.config.pattern == 'hard_prompt':
                prompt = (self.tokenized_template[ :self.head_pos] + item['tokens'][head_start:head_end] 
                    + self.tokenized_template[self.head_pos+1:self.tail_pos] 
                    + item['tokens'][tail_start:tail_end] + self.tokenized_template[self.tail_pos+1:])
            elif self.config.pattern == 'hybrid_prompt':
                prompt = (self.tokenized_template[ :self.head_pos] 
                        + self.tokenizer.encode(" [E11] [E12] [E13] ", add_special_tokens=False) + item['tokens'][head_start:head_end] + self.tokenizer.encode(" [E14] [E15] [E16] ", add_special_tokens=False)
                        + self.tokenized_template[self.head_pos+1:self.tail_pos]
                        + self.tokenizer.encode(" [E21] [E22] [E23] ", add_special_tokens=False)
                        + item['tokens'][tail_start:tail_end] + self.tokenizer.encode(" [E24] [E25] [E26] ", add_special_tokens=False)
                        + self.tokenized_template[self.tail_pos+1:])
            else:
                raise Exception('Wrong prompt type')
            new_tokens = prompt + item['tokens'][1:]
            new_tokens = new_tokens[:self.config.max_length]
            tokens.append(torch.tensor(new_tokens))
        # tokens = [torch.tensor(item['tokens']) for item in data]
        return (
            label,
            neg_labels,
            tokens
        )


class p_data_set(Dataset):

    def __init__(self, data, tokenizer, config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):

        label = torch.tensor([item['relation'] for item in data])
        # ind = torch.tensor([item['relation'] for item in data])
        instance = defaultdict(list)
        for item in data:
            instance['input_ids'].append(torch.tensor(item['input_ids']))
            instance['attention_mask'].append(torch.tensor(item['attention_mask']))
            instance['pos'].append(torch.tensor(item['pos']))
        # tokens = [torch.tensor(item['tokens']) for item in data]
        instance['input_ids'] = torch.stack(instance['input_ids'])
        instance['attention_mask'] = torch.stack(instance['attention_mask'])
        instance['pos'] = torch.stack(instance['pos'])
        return (
            label,
            instance
        )