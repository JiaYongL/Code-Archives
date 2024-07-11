import unittest
import os
print(os.getcwd())
from dataloader.sampler import data_sampler


class TestDataSampler(unittest.TestCase):

    def setUp(self):
        self.config = {
            'bert_path': 'bert-base-uncased',
            'relation_file': 'data/fewrel/relation_name.txt',
            'training_file': 'data/fewrel/CFRLdata_10_100_10_5/train_0.txt',
            'valid_file': 'data/fewrel/CFRLdata_10_100_10_5/valid_0.txt',
            'task_length': 8,
            'rel_index': 'data/fewrel/rel_index.npy',
            'rel_cluster_label': 'data/fewrel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy',
            'num_of_relation': 80,
            'max_length': 128
        }
        from types import SimpleNamespace
        self.config = SimpleNamespace(**self.config)
        self.sampler = data_sampler(config=self.config, seed=123)

    def test_sampler(self):
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(self.sampler):
            print(steps)
            print(training_data)
            print(valid_data)
            print(test_data)
            print(current_relations)
            

if __name__ == '__main__':
    unittest.main()