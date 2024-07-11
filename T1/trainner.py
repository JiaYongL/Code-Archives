import argparse
import collections
from utils import AverageMeter, set_seed
import numpy as np
from base import Trainer
from models.bert_encoder import Bert_Encoder
from models.dropout_layer import Dropout_Layer
from models.classifier import Softmax_Layer, MYNET
from dataloader.sampler2 import data_sampler
from dataloader.dataloader import get_data_loader
from tqdm import tqdm
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict
import json
import torch.optim as optim
from sklearn.cluster import KMeans
from config import Config

class RelTrainner(Trainer):
    
    def __init__(self, config):
        super().__init__(config)
    
    def set_up_model(self):
        self.encoder = Bert_Encoder(self.config)
        self.dropout_layer = Dropout_Layer(self.config)
        self.classifier = None
        self.prev_encoder = None 
        self.prev_dropout_layer = None
        self.prev_classifier = None
        torch.cuda.empty_cache()
        
    def get_proto(self, data):
        data_loader = get_data_loader(config, data, shuffle=False, drop_last=False, batch_size=16)
        features = []
        self.encoder.eval()
        self.dropout_layer.eval()
        for step, batch_data in enumerate(data_loader):
            _, _, tokens = batch_data
            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            with torch.no_grad():
                feature = self.dropout_layer(self.encoder(tokens))[1]
            features.append(feature)
        features = torch.cat(features, dim=0)
        proto = torch.mean(features, dim=0, keepdim=True).cpu()
        standard = torch.sqrt(torch.var(features, dim=0)).cpu()
        return proto, standard
    
    def select_data(self, relation_dataset):
        config = self.config
        data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=16)
        features = []
        self.encoder.eval()
        self.dropout_layer.eval()
        for step, batch_data in enumerate(data_loader):
            labels, _, tokens = batch_data
            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            with torch.no_grad():
                feature = self.dropout_layer(self.encoder(tokens))[1].cpu()
            features.append(feature)

        features = np.concatenate(features)
        num_clusters = min(config.num_protos, len(relation_dataset))
        distances = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit_transform(features)

        memory = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            instance = relation_dataset[sel_index]
            memory.append(instance)
        return memory
    
    def generate_relation_data(self, protos, relation_standard):
        relation_data = {}
        relation_sample_nums = 10
        for id in protos.keys():
            relation_data[id] = []
            difference = np.random.normal(loc=0, scale=1, size=relation_sample_nums)
            for diff in difference:
                relation_data[id].append(protos[id] + diff * relation_standard[id])
        return relation_data
    
    def generate_current_relation_data(self, relation_dataset):
        config = self.config
        data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=16)
        relation_data = []
        self.encoder.eval()
        self.dropout_layer.eval()
        for step, batch_data in enumerate(data_loader):
            _, _, tokens = batch_data
            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            with torch.no_grad():
                feature = self.dropout_layer(self.encoder(tokens))[1].cpu()
            relation_data.extend([x.unsqueeze(0) for x in torch.unbind(feature, dim=0)])
        return relation_data
    
    def construct_hard_triplets(self, output, labels, relation_data):
        positive = []
        negative = []
        pdist = nn.PairwiseDistance(p=2)
        for rep, label in zip(output, labels):
            positive_relation_data = relation_data[label.item()]
            negative_relation_data = []
            for key in relation_data.keys():
                if key != label.item():
                    negative_relation_data.extend(relation_data[key])
            positive_distance = torch.stack([pdist(rep.cpu(), p) for p in positive_relation_data])
            negative_distance = torch.stack([pdist(rep.cpu(), n) for n in negative_relation_data])
            positive_index = torch.argmax(positive_distance)
            negative_index = torch.argmin(negative_distance)
            positive.append(positive_relation_data[positive_index.item()])
            negative.append(negative_relation_data[negative_index.item()])

        return positive, negative
    
    def contrastive_loss(self, hidden, labels):
        logsoftmax = nn.LogSoftmax(dim=-1)
        return -(logsoftmax(hidden) * labels).sum() / labels.sum()
    
    def compute_jsd_loss(self, m_input):
        # m_input: the result of m times dropout after the classifier.
        # size: m*B*C
        m = m_input.shape[0]
        mean = torch.mean(m_input, dim=0)
        jsd = 0
        for i in range(m):
            loss = F.kl_div(F.log_softmax(mean, dim=-1), F.softmax(m_input[i], dim=-1), reduction='none')
            loss = loss.sum()
            jsd += loss / m
        return jsd
    
    def evaluate_strict_model(self, test_data, num_of_seen_relations):
        config = self.config
        data_loader = get_data_loader(config, test_data, batch_size=16)
        self.encoder.eval()
        self.dropout_layer.eval()
        self.classifier.eval()
        n = len(test_data)

        correct = 0
        seen_relation_ids = [i for i in range(num_of_seen_relations)]
        for step, batch_data in enumerate(data_loader):
            labels, _, tokens = batch_data
            labels = labels.to(config.device)
            labels = [self.map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)

            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            reps = self.encoder(tokens)wrqwrq
            reps, _ = self.dropout_layer(reps)
            logits = self.classifier(reps)
            
            seen_sim = torch.gather(logits, 1, torch.tensor(seen_relation_ids, device=config.device).unsqueeze(0).expand(logits.size(0), -1))
            max_smi, _ = torch.max(seen_sim, dim=1)
            label_smi = torch.gather(logits, 1, labels.unsqueeze(1)).squeeze(1)
            correct += torch.sum(label_smi >= max_smi).item()

        return correct / n
    
    def train_simple_model(self, data):
        # print('start training simple model')
        config = self.config
        dataloader = get_data_loader(self.config, data, shuffle=True, batch_size=config.batch_size)
        
        self.encoder.train()
        self.dropout_layer.train()
        self.classifier.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'lr': 0.00001},
            {'params': self.dropout_layer.parameters(), 'lr': 0.00001},
            {'params': self.classifier.parameters(), 'lr': 0.001}
        ])
        for epoch_i in range(config.step1_epochs):
            losses = []
            for step, batch_data in enumerate(tqdm(dataloader)):
                optimizer.zero_grad()
                labels, _, tokens = batch_data
                labels = [self.map_relid2tempid[x.item()] for x in labels]
                labels = torch.tensor(labels).to(config.device)
                tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
                
                reps = self.encoder(tokens)
                reps, _ = self.dropout_layer(reps)
                logits = self.classifier(reps)
                loss = criterion(logits, labels)

                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            # print('epoch: {}, loss: {}'.format(epoch_i, np.mean(losses)))
    
    def train_mem_model(self, data, new_relation_data, num_of_epochs=10):
        # print('start training memory model')
        config = self.config
        data_loader = get_data_loader(config, data, shuffle=True, batch_size=config.batch_size)
        self.encoder.train()
        self.dropout_layer.train()
        self.classifier.train()

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam([
        #     {'params': self.encoder.parameters(), 'lr': 0.00001},
        #     {'params': self.dropout_layer.parameters(), 'lr': 0.00001},
        #     {'params': self.classifier.parameters(), 'lr': 0.001}
        # ])
        
        if self.prev_encoder is not None:
            optimizer = optim.Adam([
                {'params': self.encoder.parameters(), 'lr': 0.00001},
                {'params': self.dropout_layer.parameters(), 'lr': 0.00001},
                {'params': self.classifier.fc.parameters(), 'lr': 0.001},
                {'params': self.classifier.prev_fc.parameters(), 'lr': 0.0001}
            ])
        else:
            optimizer = optim.Adam([
                {'params': self.encoder.parameters(), 'lr': 0.00001},
                {'params': self.dropout_layer.parameters(), 'lr': 0.00001},
                {'params': self.classifier.fc.parameters(), 'lr': 0.001},
            ])
        
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        distill_criterion = nn.CosineEmbeddingLoss()
        T = config.kl_temp
        for epoch_i in range(num_of_epochs):
            losses = []
            for step, (labels, _, tokens) in enumerate(tqdm(data_loader)):
                optimizer.zero_grad()
                logits_all = []
                tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
                labels = labels.to(config.device)
                origin_labels = labels[:]
                labels = [self.map_relid2tempid[x.item()] for x in labels]
                labels = torch.tensor(labels).to(config.device)
                reps = self.encoder(tokens)
                normalized_reps_emb = F.normalize(reps.view(-1, reps.size()[1]), p=2, dim=1)
                outputs,_ = self.dropout_layer(reps)
                if self.prev_dropout_layer is not None:
                    prev_outputs, _ = self.prev_dropout_layer(reps)
                    positives,negatives = self.construct_hard_triplets(prev_outputs, origin_labels, new_relation_data)
                else:
                    positives, negatives = self.construct_hard_triplets(outputs, origin_labels, new_relation_data)

                for _ in range(config.f_pass):
                    output, output_embedding = self.dropout_layer(reps)
                    logits = self.classifier(output)
                    logits_all.append(logits)

                positives = torch.cat(positives, 0).to(config.device)
                negatives = torch.cat(negatives, 0).to(config.device)
                anchors = outputs
                logits_all = torch.stack(logits_all)
                m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B
                loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
                loss2 = self.compute_jsd_loss(logits_all)
                tri_loss = triplet_loss(anchors, positives, negatives)
                loss = loss1 + loss2 + tri_loss

                if self.prev_encoder is not None:
                    prev_reps = self.prev_encoder(tokens).detach()
                    normalized_prev_reps_emb = F.normalize(prev_reps.view(-1, prev_reps.size()[1]), p=2, dim=1)

                    feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb,
                                                            torch.ones(tokens.size(0)).to(
                                                                config.device))
                    loss += feature_distill_loss

                if self.prev_dropout_layer is not None and self.prev_classifier is not None:
                    prediction_distill_loss = None
                    dropout_output_all = []
                    prev_dropout_output_all = []
                    for i in range(config.f_pass):
                        output, _ = self.dropout_layer(reps)
                        prev_output, _ = self.prev_dropout_layer(reps)
                        dropout_output_all.append(output)
                        prev_dropout_output_all.append(output)
                        pre_logits = self.prev_classifier(output).detach()

                        pre_logits = F.softmax(pre_logits[: , :len(self.prev_relations)] / T, dim=1)

                        log_logits = F.log_softmax(logits_all[i][: , :len(self.prev_relations)] / T, dim=1)
                        if i == 0:
                            prediction_distill_loss = -torch.mean(torch.sum(pre_logits * log_logits, dim=1))
                        else:
                            prediction_distill_loss += -torch.mean(torch.sum(pre_logits * log_logits, dim=1))

                    prediction_distill_loss /= config.f_pass
                    loss += prediction_distill_loss
                    dropout_output_all = torch.stack(dropout_output_all)
                    prev_dropout_output_all = torch.stack(prev_dropout_output_all)
                    mean_dropout_output_all = torch.mean(dropout_output_all, dim=0)
                    mean_prev_dropout_output_all = torch.mean(prev_dropout_output_all,dim=0)
                    normalized_output = F.normalize(mean_dropout_output_all.view(-1, mean_dropout_output_all.size()[1]), p=2, dim=1)
                    normalized_prev_output = F.normalize(mean_prev_dropout_output_all.view(-1, mean_prev_dropout_output_all.size()[1]), p=2, dim=1)
                    hidden_distill_loss = distill_criterion(normalized_output, normalized_prev_output,
                                                            torch.ones(tokens.size(0)).to(
                                                                config.device))
                    loss += hidden_distill_loss

                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            # print(f"epoch {epoch_i}, loss is {np.array(losses).mean()}")
    
    def train_base(self, data):
        data_loader = get_data_loader(self.config, data, shuffle=True, batch_size=self.config.batch_size)
        self.encoder.train()
        self.dropout_layer.train()
        self.classifier.train()
        optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'lr': 0.00001},
            {'params': self.dropout_layer.parameters(), 'lr': 0.00001},
            {'params': self.classifier.parameters(), 'lr': 0.001}
        ])
        
        for epoch_i in range(self.config.step1_epochs):
            losses = 0
            for step, batch_data in enumerate(tqdm(data_loader)):
                labels, _, tokens = batch_data
                labels = [self.map_relid2tempid[x.item()] for x in labels]
                labels = torch.tensor(labels).to(self.config.device)
                tokens = torch.stack([x.to(self.config.device) for x in tokens],dim=0)
                reps = self.encoder(tokens)
                reps, _ = self.dropout_layer(reps)
                logits = self.classifier(reps)
                loss = F.cross_entropy(logits, labels)
                losses += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # print('epoch: {}, loss: {}'.format(epoch_i, losses))
            
    def train_soft(self, data, session):
        self.encoder.eval()
        self.dropout_layer.eval()
        self.classifier.eval()
        data_loader = get_data_loader(self.config, data, shuffle=True, batch_size=self.config.batch_size)
        features = []
        la = []
        classlist = set()
        for step, batch_data in enumerate(tqdm(data_loader)):
            labels, _, tokens = batch_data
            labels = [self.map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(self.config.device)
            tokens = torch.stack([x.to(self.config.device) for x in tokens],dim=0)
            reps = self.encoder(tokens).detach()
            reps, _ = self.dropout_layer(reps)
            reps = reps.detach()
            for id, rep in zip(labels, reps):
                features.append(rep)
                la.append(id)
                classlist.add(id.item())
        features = torch.stack(features)
        la = torch.stack(la)
        self.classifier.update_fc_avg(features, la, classlist)
        self.classifier.soft_calibration(self.config, session=session)
        
    
    def train_proto(self, round):
        sampler = data_sampler(config=self.config, seed= self.config.seed + 100 * round)
        set_seed(self.config.seed + 100 * round)
        self.set_up_model()
        self.reset_trlog_for_next_round()
        self.prev_relations = []
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.map_relid2tempid = {}
        self.memorized_samples = {}
        self.relation_standard = {}
        self.memory = collections.defaultdict(list)
        self.classifier = MYNET(self.config).to(self.config.device)
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in tqdm(enumerate(sampler)):
            # print('round: {}, step: {}'.format(round, steps))
            training_data_for_init = []
            for rel in current_relations:
                training_data_for_init.extend(training_data[rel])
            current_test_data = []
            for rel in current_relations:
                current_test_data.extend(test_data[rel])
            for i, rel in enumerate(current_relations):
                self.map_relid2tempid[self.rel2id[rel]] = len(self.map_relid2tempid)
            
            if steps == 0:
                self.train_base(training_data_for_init)
            else:
                self.train_soft(training_data_for_init, steps)
            
            if steps != 0:
                forward = self.evaluate_strict_model(current_test_data, len(seen_relations))
            
            history_test_data = []
            for rel in seen_relations:
                history_test_data.extend(historic_test_data[rel])
            
            cur_acc = self.evaluate_strict_model(current_test_data, len(seen_relations))
            history_acc = self.evaluate_strict_model(history_test_data, len(seen_relations))
            
            self.trlog['current_acc'].append(cur_acc)
            self.trlog['history_acc'].append(history_acc)

            self.prev_relations.extend(current_relations)
            
        self.save_result_after_training()
        print('round: {}, current_acc: {}, history_acc: {}'.format(round, self.trlog['current_acc'], self.trlog['history_acc']))
    
    
    def train_one_round(self, round):
        sampler = data_sampler(config=self.config, seed= self.config.seed + 100 * round)
        set_seed(self.config.seed + 100 * round)
        self.set_up_model()
        self.reset_trlog_for_next_round()
        self.prev_relations = []
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.map_relid2tempid = {}
        self.memorized_samples = {}
        self.relation_standard = {}
        self.prev_classifier = None
        self.memory = collections.defaultdict(list)
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in tqdm(enumerate(sampler)):
            # print('round: {}, step: {}'.format(round, steps))
            training_data_for_init = []
            for rel in current_relations:
                training_data_for_init.extend(training_data[rel])
            current_test_data = []
            for rel in current_relations:
                current_test_data.extend(test_data[rel])
            for i, rel in enumerate(current_relations):
                self.map_relid2tempid[self.rel2id[rel]] = len(self.map_relid2tempid)
            
            if self.config.retain_prev:
                prev_fc = None
                if self.prev_classifier is not None:
                    if self.prev_classifier.prev_fc is not None:
                        dim1 = self.prev_classifier.prev_fc.weight.data.shape[0] + self.prev_classifier.fc.weight.data.shape[0]
                        prev_fc = nn.Linear(self.config.encoder_output_size, dim1).to(self.config.device)
                        prev_fc.weight.data[:self.prev_classifier.prev_fc.weight.data.shape[0]] = self.prev_classifier.prev_fc.weight.data.clone()
                        prev_fc.weight.data[self.prev_classifier.prev_fc.weight.data.shape[0]:] = self.prev_classifier.fc.weight.data.clone()
                    else:
                        prev_fc = self.prev_classifier.fc
                self.classifier = Softmax_Layer(self.config.encoder_output_size, len(current_relations), prev=prev_fc).to(self.config.device)
            else:
                self.classifier = Softmax_Layer(self.config.encoder_output_size, len(seen_relations)).to(self.config.device)
            
            self.train_simple_model(training_data_for_init)
            
            if steps != 0:
                forward = self.evaluate_strict_model(current_test_data, len(seen_relations))
            temp_protos = {}
            
            for relation in current_relations:
                proto, standard = self.get_proto(training_data[relation])
                temp_protos[self.rel2id[relation]] = proto
                self.relation_standard[self.rel2id[relation]] = standard

            for relation in self.prev_relations:
                proto, _ = self.get_proto(self.memorized_samples[relation])
                temp_protos[self.rel2id[relation]] = proto

            new_relation_data = self.generate_relation_data(temp_protos, self.relation_standard)
            for rel in current_relations:
                new_relation_data[self.rel2id[rel]].extend(self.generate_current_relation_data(training_data[rel]))
            
            torch.cuda.empty_cache()
            
            self.train_mem_model(training_data_for_init, new_relation_data, self.config.step2_epochs)
            
            for relation in current_relations:
                selected_data = self.select_data(training_data[relation])
                self.memorized_samples[relation] = selected_data
                self.memory[self.rel2id[relation]] = selected_data
            
            train_data_for_memory = []
            for rel in seen_relations:
                train_data_for_memory.extend(self.memorized_samples[rel])

            temp_protos = {}
            
            for relation in seen_relations:
                proto, standard = self.get_proto(self.memorized_samples[relation])
                temp_protos[self.rel2id[relation]] = proto
                
            self.train_mem_model(train_data_for_memory, new_relation_data, self.config.step3_epochs)
            
            
            history_test_data = []
            for rel in seen_relations:
                history_test_data.extend(historic_test_data[rel])
            
            cur_acc = self.evaluate_strict_model(current_test_data, len(seen_relations))
            history_acc = self.evaluate_strict_model(history_test_data, len(seen_relations))
            
            self.trlog['current_acc'].append(cur_acc)
            self.trlog['history_acc'].append(history_acc)
            
            self.prev_relations.extend(current_relations)
            self.save_cur2prev()
        self.save_result_after_training()
        print('round: {}, current_acc: {}, history_acc: {}'.format(round, self.trlog['current_acc'], self.trlog['history_acc']))
        

    def save_cur2prev(self):
        self.prev_encoder = deepcopy(self.encoder)
        self.prev_dropout_layer = deepcopy(self.dropout_layer)
        self.prev_classifier = deepcopy(self.classifier)
        torch.cuda.empty_cache()
    
    def train(self, round):
        for i in range(round):
            self.train_one_round(i)
            # self.train_proto(i)
        self.show_log()
        

# import signal
# import sys
# import time

# def signal_handler(sig, frame):
    
#     print('You pressed Ctrl+C!')
#     for i in range(10):
#         print('Wait for {} seconds'.format(10 - i))
#         time.sleep(1)
#     sys.exit(0)

# signal.signal(signal.SIGINT, signal_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", type=str)
    parser.add_argument("--pattern", default="soft_prompt", type=str)
    parser.add_argument("--retain_prev", default=False, type=bool)
    args = parser.parse_args()
    config = Config(args.config)
    config.pattern = args.pattern
    config.retain_prev = args.retain_prev
    
    print(config.retain_prev)
    print('config: ')
    print(json.dumps(config.__dict__, indent=4))
    print('----------------------------------------')
    trainer = RelTrainner(config)
    trainer.train(round=config.total_round)
    # for i in range(10):
    #     trainer = RelTrainner(config)
    #     config.shift_weight = i / 10
    #     print('config: ')
    #     print(json.dumps(config.__dict__, indent=4))
    #     print('----------------------------------------')
    #     trainer.train(round=config.total_round)