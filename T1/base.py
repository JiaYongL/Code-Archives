import abc
from utils import AverageMeter
import numpy as np

class Trainer(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self):
        pass
    def __init__(self, config):
        self.config = config
        self.ave = AverageMeter()
        self.init_log()
        
    def init_log(self):
        # train statistics
        self.trlog = {}
        self.trlog['current_acc'] = [] 
        self.trlog['history_acc'] = [] 
        self.trlog['forward_acc'] = [] 
        self.trlog['all_current_acc'] = [] 
        self.trlog['all_history_acc'] = []
    
    def show_list(self, lists):
        for list in lists:
            print(list)
    
    def show_log(self):
        print('>>>>>>>>>>>>>>>current_acc<<<<<<<<<<<<<<<\n\n')
        print(self.trlog['current_acc'])
        print('\n\n----------------------------------------\n\n')
        
        print('>>>>>>>>>>>>>>>history_acc<<<<<<<<<<<<<<<\n\n')
        print(self.trlog['history_acc'])
        print('\n\n----------------------------------------\n\n')
        
        print('>>>>>>>>>>>>>>>all_current_acc<<<<<<<<<<<<<<<\n\n')
        self.show_list(self.trlog['all_current_acc'])
        print('\n\n----------------------------------------\n\n')
        
        print('>>>>>>>>>>>>>>>all_current_acc_mean<<<<<<<<<<<<<<<\n\n')
        print(np.mean(np.array(self.trlog['all_current_acc']), axis=0).tolist())
        print('\n\n----------------------------------------\n\n')
        
        print('>>>>>>>>>>>>>>>all_current_acc_std<<<<<<<<<<<<<<<\n\n')
        print('all_current_acc_std: ', np.std(np.array(self.trlog['all_current_acc']), axis=0).tolist())
        print('\n\n----------------------------------------\n\n')
        
        print('>>>>>>>>>>>>>>>all_history_acc<<<<<<<<<<<<<<<\n\n')
        self.show_list(self.trlog['all_history_acc'])
        print('\n\n----------------------------------------\n\n')
        
        print('>>>>>>>>>>>>>>>all_history_acc_mean<<<<<<<<<<<<<<<\n\n')
        print('all_history_acc: ', np.mean(self.trlog['all_history_acc'], axis=0).tolist())
        print('\n\n----------------------------------------\n\n')
        
        print('>>>>>>>>>>>>>>>all_history_acc_std<<<<<<<<<<<<<<<\n\n')
        print('all_history_acc_std: ', np.std(self.trlog['all_history_acc'], axis=0).tolist())
        print('\n\n----------------------------------------\n\n')
    
    def save_result_after_training(self):
        self.trlog['all_current_acc'].append(self.trlog['current_acc'])
        self.trlog['all_history_acc'].append(self.trlog['history_acc'])
    
    def reset_trlog_for_next_round(self):
        self.ave = AverageMeter()
        self.trlog['current_acc'] = []
        self.trlog['history_acc'] = []
        self.trlog['forward_acc'] = []