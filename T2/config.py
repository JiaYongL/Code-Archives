import argparse
import os
"""
Detailed hyper-parameter configurations.
"""
class Param:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()
        self.args = all_args

    def all_param(self, parser):

        ##################################common parameters####################################
        parser.add_argument("--gpu", default=0, type=int)

        parser.add_argument("--dataname", default='FewRel', type=str, help="Use TACRED or FewRel datasets.")

        parser.add_argument("--task_name", default='FewRel', type=str)

        parser.add_argument("--max_length", default=256, type=int)

        parser.add_argument("--this_name", default="continual", type=str)

        parser.add_argument("--device", default="cuda", type=str)

        ###############################   training ################################################

        parser.add_argument("--batch_size", default=8, type=int)

        parser.add_argument("--learning_rate", default=5e-6, type=float)
        
        parser.add_argument("--total_round", default=6, type=int)
        
        parser.add_argument("--rel_per_task", default=4)

        parser.add_argument("--pattern", default="entity_marker") 

        parser.add_argument("--encoder_output_size", default=768, type=int)

        parser.add_argument("--vocab_size", default=30522, type =int)

        parser.add_argument("--marker_size", default=4, type=int)

        # Temperature parameter in CL and CR
        parser.add_argument("--temp", default=0.1, type=float)

        # The projection head outputs dimensions
        parser.add_argument("--feat_dim", default=768, type=int)

        # Temperature parameter in KL
        parser.add_argument("--kl_temp", default=10, type=float)

        parser.add_argument("--num_workers", default=0, type=int)

        # epoch1
        parser.add_argument("--step1_epochs", default=10, type=int) 

        # epoch2
        parser.add_argument("--step2_epochs", default=10, type=int) 

        parser.add_argument("--seed", default=100, type=int) 

        parser.add_argument("--max_grad_norm", default=10, type=float) 

        # Memory size
        parser.add_argument("--num_protos", default=1, type=int)

        parser.add_argument("--optim", default='adam', type=str)

        # dataset path
        parser.add_argument("--data_path", default='datasets/', type=str)

        # bert-base-uncased weights path
        parser.add_argument("--bert_path", default="datasets/bert-base-uncased", type=str)
        
        
        parser.add_argument("--num_k", default=5, type=int)
        parser.add_argument("--use_ce", default=1, type=int)
        parser.add_argument("--use_unused", default=0, type=int)
        parser.add_argument("--task_length", default=8, type=int)
        parser.add_argument("--prompt_len", default=1, type=int)
        parser.add_argument("--prompt_num", default=1, type=int)
        parser.add_argument("--add_entity_marker", default=1, type=int)
        parser.add_argument("--prompt_size", default=4, type=int)
        
        parser.add_argument("--rel_cluster_label", default='./data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy', type=str)
        parser.add_argument("--training_data", default='./data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt', type=str)
        parser.add_argument("--valid_data", default='./data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt', type=str)
        parser.add_argument("--test_data", default='./data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt', type=str)
        parser.add_argument("--rel_index", default='./data/CFRLFewRel/rel_index.npy', type=str)
        parser.add_argument("--model", default='bert', type=str)
        parser.add_argument("--relation_name", default='./data/CFRLFewRel/relation_name.txt', type=str)
        parser.add_argument("--relation_description", default='./data/CFRLFewRel/relation_description.txt', type=str)
        
        
        return parser
