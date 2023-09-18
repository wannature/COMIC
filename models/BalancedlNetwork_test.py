import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from logger import Logger
import time
import numpy as np
import warnings
import pdb
import json
from loss.xERM_loss import xERMLoss
from loss.SpccLoss import SPLC
from sklearn.decomposition import IncrementalPCA
from utils.helper_functions import CutoutPIL, ModelEma, add_weight_decay,shot_mAP
from datetime import datetime
import sklearn.metrics as mt
from typing import Tuple, List
from utils.util import *
import sklearn.metrics as mt
class MLTModel ():
    def __init__(self, config, train_loader, val_dataloader, test=False):
        self.config = config
        self.training_opt = self.config['training_opt']
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config['shuffle'] if 'shuffle' in config else False
        self.train_loader=train_loader
        
        self.val_dataloader=val_dataloader
        # Setup logger
        self.logger = Logger(self.training_opt['log_dir'])
        self.total_mAP=0
        # init moving average
        self.balanced_embed_mean = torch.zeros(int(self.training_opt['feature_dim'])).numpy()
        self.head_embed_mean = torch.zeros(int(self.training_opt['feature_dim'])).numpy()
        self.tail_embed_mean = torch.zeros(int(self.training_opt['feature_dim'])).numpy()
        self.distribution_path=self.training_opt['distribution_path']
        self.co_occurrence_matrix=self.training_opt['co_occurrence_matrix']
        self.mu = 0.9
        
        # Initialize model
        self.init_models()

        # apply incremental pca
        self.apply_pca = ('apply_ipca' in self.config) and self.config['apply_ipca']
        if self.apply_pca:
            print('==========> Apply Incremental PCA <=======')
            self.pca = IncrementalPCA(n_components=self.config['num_components'], batch_size=self.training_opt['batch_size'])

        # Load pre-trained model parameters
        if 'model_dir' in self.config and self.config['model_dir'] is not None:
            self.load_model(self.config['model_dir'])

        # Under training mode, initialize training steps, optimizers, schedulers, criterions
        if not self.test_mode:
            print('Using steps for training.')
            self.training_data_num = self.train_loader.dataset.number
            self.epoch_steps = int(self.training_data_num  / self.training_opt['batch_size'])
            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            self.init_optimizers(self.balanced_optim_params_dict, self.head_optim_params_dict, self.tail_optim_params_dict)
            self.init_criterions()
            
            # Set up log file
            self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
            self.logger.log_cfg(self.config)
        else:
            self.log_file = None
        
    def init_models(self, optimizer=True):
        print("Using", torch.cuda.device_count(), "GPUs.")
        # head_networks
        head_networks_defs = self.config['head_networks']
        self.head_networks = {}
        self.head_optim_params_dict = {}
        self.head_optim_named_params = {}
        for key, val in head_networks_defs.items():
            
            def_file = val['def_file']
            head_args = val['params']
            head_args.update({'test': self.test_mode})

            self.head_networks[key] = source_import(def_file).create_model(**head_args)
            self.head_networks[key] = nn.DataParallel(self.head_networks[key]).cuda()

            # if 'fix' in val and val['fix']:
            #     print('Freezing weights of module {}'.format(key))
            #     for param_name, param in self.head_networks[key].named_parameters():
            #         # Freeze all parameters except final fc layer
            #         if 'fc' not in param_name:
            #             param.requires_grad = False
            #     print('=====> Freezing: {} | False'.format(key))
            
            # if 'fix_set' in val:
            #     for fix_layer in val['fix_set']:
            #         for param_name, param in self.head_networks[key].named_parameters():
            #             if fix_layer == param_name:
            #                 param.requires_grad = False
            #                 print('=====> Freezing: {} | {}'.format(param_name, param.requires_grad))
            #                 continue

            # Optimizer list
            optim_params = val['optim_params']
            self.head_optim_named_params.update(dict(self.head_networks[key].named_parameters()))
            self.head_optim_params_dict[key] = {'params': self.head_networks[key].parameters(),
                                                'lr': optim_params['lr'],
                                                'momentum': optim_params['momentum'],
                                                'weight_decay': optim_params['weight_decay']}

        # tail_networks
        tail_networks_defs = self.config['tail_networks']
        self.tail_networks = {}
        self.tail_optim_params_dict = {}
        self.tail_optim_named_params = {}
        for key, val in tail_networks_defs.items():
            
            def_file = val['def_file']
            tail_args = val['params']
            tail_args.update({'test': self.test_mode})

            self.tail_networks[key] = source_import(def_file).create_model(**tail_args)
            self.tail_networks[key] = nn.DataParallel(self.tail_networks[key]).cuda()


            # Optimizer list
            optim_params = val['optim_params']
            self.tail_optim_named_params.update(dict(self.tail_networks[key].named_parameters()))
            self.tail_optim_params_dict[key] = {'params': self.tail_networks[key].parameters(),
                                                'lr': optim_params['lr'],
                                                'momentum': optim_params['momentum'],
                                                'weight_decay': optim_params['weight_decay']}


        # balanced_networks
        balanced_networks_defs = self.config['banlaced_networks']    

        self.balanced_networks = {}
        self.balanced_optim_params_dict = {}
        self.balanced_optim_named_params = {}
        
        for key, val in balanced_networks_defs.items():
            # Networks
            def_file = val['def_file']
            model_args = val['params']
            model_args.update({'test': False})
            self.balanced_networks[key] = source_import(def_file).create_model(**model_args)
            self.balanced_networks[key] = nn.DataParallel(self.balanced_networks[key]).cuda()
            # Optimizer list
            optim_params = val['optim_params']
            self.balanced_optim_named_params.update(dict(self.balanced_networks[key].named_parameters()))
            self.balanced_optim_params_dict[key] = {'params': self.balanced_networks[key].parameters(),
                                                'lr': optim_params['lr'],
                                                'momentum': optim_params['momentum'],
                                                'weight_decay': optim_params['weight_decay']}

    

            
    def init_criterions(self):
        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = list(val['loss_params'].values())

            self.criterions[key] = source_import(def_file).create_loss(*loss_args).cuda()
            self.criterion_weights[key] = val['weight']
          
            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                'lr': optim_params['lr'],
                                'momentum': optim_params['momentum'],
                                'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

        

    def init_optimizers(self, balanced_optim_params_dict, head_optim_params_dict, tail_optim_params_dict):
        '''
        seperate backbone optimizer and classifier optimizer
        '''
        #balanced optimizer
        balanced_networks_defs = self.config['banlaced_networks']
        self.balanced_optimizer_dict = {}
        self.balanced_scheduler_dict = {}

        for key, val in balanced_networks_defs.items():
            # optimizer
            if 'optimizer' in self.training_opt and self.training_opt['optimizer'] == 'adam':
                print('=====> Using Adam optimizer')
                optimizer = optim.Adam([balanced_optim_params_dict[key]])
            else:
                print('=====> Using SGD optimizer')
                optimizer = optim.SGD([balanced_optim_params_dict[key]])
            self.balanced_optimizer_dict[key] = optimizer

            # scheduler
            scheduler_params = val['scheduler_params']
            print(f'===> Module {key}: Using stepLR')
            self.balanced_scheduler_dict[key] = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_params['step_size'], gamma=scheduler_params['gamma'])

                         
        #head optimizer
        head_networks_defs = self.config['head_networks']
        self.head_optimizer_dict = {}
        self.head_scheduler_dict = {}

        for key, val in head_networks_defs.items():
            # optimizer
            if 'optimizer' in self.training_opt and self.training_opt['optimizer'] == 'adam':
                print('=====> Using Adam optimizer')
                optimizer = optim.Adam([head_optim_params_dict[key]])
            else:
                print('=====> Using SGD optimizer')
                optimizer = optim.SGD([head_optim_params_dict[key]])
            self.head_optimizer_dict[key] = optimizer

            # scheduler
            scheduler_params = val['scheduler_params']
            print(f'===> Module {key}: Using stepLR')
            self.head_scheduler_dict[key] = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_params['step_size'], gamma=scheduler_params['gamma'])

        #tail optimizer
        tail_networks_defs = self.config['tail_networks']
        self.tail_optimizer_dict = {}
        self.tail_scheduler_dict = {}

        for key, val in tail_networks_defs.items():
            # optimizer
            if 'optimizer' in self.training_opt and self.training_opt['optimizer'] == 'adam':
                print('=====> Using Adam optimizer')
                optimizer = optim.Adam([tail_optim_params_dict[key]])
            else:
                print('=====> Using SGD optimizer')
                optimizer = optim.SGD([tail_optim_params_dict[key]])
            self.tail_optimizer_dict[key] = optimizer

            # scheduler
            scheduler_params = val['scheduler_params']
            print(f'===> Module {key}: Using stepLR')
            self.tail_scheduler_dict[key] = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_params['step_size'], gamma=scheduler_params['gamma'])                                                                          

            # self.balanced_scheduler_dict[key] = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=scheduler_params['step_size'],
            #                                                                       epochs=self.training_opt['num_epochs'], pct_start=0.2)     
            # if scheduler_params['coslr']:
            #     print("===> Module {} : Using coslr eta_min={}".format(key, scheduler_params['endlr']))
            #     self.model_scheduler_dict[key] = torch.optim.lr_scheduler.CosineAnnealingLR(
            #                         optimizer, self.training_opt['num_epochs'], eta_min=scheduler_params['endlr'])
            # elif scheduler_params['warmup']:
            #     print("===> Module {} : Using warmup".format(key))
            #     self.model_scheduler_dict[key] = WarmupMultiStepLR(optimizer, scheduler_params['lr_step'], 
            #                                         gamma=scheduler_params['lr_factor'], warmup_epochs=scheduler_params['warm_epoch'])
            # elif scheduler_params['steplr']:
            #     print(f'===> Module {key}: Using stepLR')
            #     self.model_scheduler_dict[key] = optim.lr_scheduler.StepLR(optimizer,
            #                                                                         step_size=scheduler_params['step_size'],
            #                                                                       gamma=scheduler_params['gamma'])
            # else: 
           
        return

    def show_current_lr(self):
        max_lr = 0.0
        for key, val in self.balanced_optimizer_dict.items():
            lr_set = list(set([para['lr'] for para in val.param_groups]))
            if max(lr_set) > max_lr:
                max_lr = max(lr_set)
            lr_set = ','.join([str(i) for i in lr_set])
            print_str = ['=====> Current Learning Rate of model {} : {}'.format(key, str(lr_set))]
            print_write(print_str, self.log_file)
            
        return max_lr


    def batch_forward(self, inputs, labels=None, feature_ext=False, phase='train'):
        '''
        This is a general single batch running function. 
        '''

        # Calculate Features of balanced model
        self.balanced_features = self.balanced_networks['feat_model'](inputs)
        if phase == 'train':
            self.balanced_embed_mean = self.mu * self.balanced_embed_mean + self.balanced_features.detach().mean(0).view(-1).cpu().numpy()

        # Calculate Features of head model
        self.head_features = self.head_networks['feat_model'](inputs)
        # update moving average
        if phase == 'train':
            self.head_embed_mean = self.mu * self.head_embed_mean + self.head_features.detach().mean(0).view(-1).cpu().numpy()
        
        # Calculate Features of tail model
        self.tail_features = self.tail_networks['feat_model'](inputs)
        # update moving average
        if phase == 'train':
            self.tail_embed_mean = self.mu * self.tail_embed_mean + self.tail_features.detach().mean(0).view(-1).cpu().numpy()
        
        # temp_head_features=self.head_features.clone()
        # temp_tail_features=self.tail_features.clone()
        # self.keys=torch.cat([self.balanced_features.unsqueeze(1), temp_tail_features.unsqueeze(1)],1)
        # self.head_features=temp_head_features + self.balanced_networks['additive_attention'](self.head_features.unsqueeze(1),self.keys,self.keys).squeeze(1)

        # self.keys=torch.cat([self.balanced_features.unsqueeze(1), temp_head_features.unsqueeze(1)],1)
        # self.tail_features= temp_tail_features+ self.balanced_networks['additive_attention'](self.tail_features.unsqueeze(1),self.keys,self.keys).squeeze(1)

        # addtive attention
        self.keys=torch.cat([self.head_features.unsqueeze(1), self.tail_features.unsqueeze(1)],1)
        # self.balanced_features=(self.balanced_features+ 0.1*self.balanced_networks['additive_attention'](self.balanced_features.unsqueeze(1),self.keys,self.keys).squeeze(1))


        # self.head_features=temp_tail_features/(self.balanced_features+temp_tail_features) * temp_tail_features+self.balanced_features/(self.balanced_features+self.tail_features) * self.balanced_features+ temp_head_features
        # self.tail_features=temp_head_features/(temp_head_features+self.balanced_features) * temp_head_features+self.balanced_features/(temp_head_features+self.balanced_features) * self.balanced_features+ temp_tail_features
        # self.balanced_features=temp_tail_features/(temp_head_features+temp_tail_features) * temp_tail_features+temp_head_features/(temp_head_features+temp_tail_features) * temp_head_features+ self.balanced_features
            
        
        # # If not just extracting features, calculate logits
        if not feature_ext:
            # self.logits=self.features
            cont_eval = 'continue_eval' in self.training_opt and self.training_opt['continue_eval'] and phase != 'train'
            self.logits_balanced,_= self.balanced_networks['classifier'](self.balanced_features, labels, self.balanced_embed_mean)

        if not feature_ext:
            # self.logits=self.features
            cont_eval = 'continue_eval' in self.training_opt and self.training_opt['continue_eval'] and phase != 'train'
            self.logits_head, self.logits_head_moving= self.head_networks['classifier'](self.head_features, labels, self.head_embed_mean)

       # If not just extracting features, calculate logits
        if not feature_ext:
            # self.logits=self.features
            cont_eval = 'continue_eval' in self.training_opt and self.training_opt['continue_eval'] and phase != 'train'
            self.logits_tail, self.logits_tail_moving= self.tail_networks['classifier'](self.tail_features, labels, self.tail_embed_mean)
    
    
    def batch_backward(self, print_grad=False):
        # Zero out head optimizer gradients
        for key, optimizer in self.head_optimizer_dict.items():
            optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from head loss outputs
        losses=self.head_loss
        losses.backward(retain_graph=True)

        # Zero out tail optimizer gradients
        for key, optimizer in self.tail_optimizer_dict.items():
            optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from tail loss outputs
        losses=self.tail_loss
        losses.backward(retain_graph=True)

        # Zero out balanced optimizer gradients
        for key, optimizer in self.balanced_optimizer_dict.items():
            optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        losses=self.balanced_loss
        losses.backward()
        # display gradient
        if self.training_opt['display_grad']:
            print_grad_norm(self.balanced_optim_named_params, print_write, self.log_file, verbose=print_grad)
        # Step optimizers
        for key, optimizer in self.head_optimizer_dict.items():
            optimizer.step()
        for key, optimizer in self.tail_optimizer_dict.items():
            optimizer.step()
        for key, optimizer in self.balanced_optimizer_dict.items():
            optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def get_recall_precision(self, targets, preds):
        # temp_targets=targets[0]
        # temp_preds=(preds[0]>0.5).int()
        # for i in range(temp_targets.shape[0]):
        #     print(temp_targets[i])
        #     print(temp_preds[i])

        # for i in range(1, len(preds)):
        #     temp_targets=torch.cat((temp_targets,targets[i]),0)
        #     temp_preds=torch.cat((temp_preds,(preds[i]>0.5).int()),0)
        # # print((temp_targets.reshape(1,-1)==1).cpu().sum())
        # # temp1=np.where(temp_targets.reshape(1,-1)==1)
        # # temp2=np.where(temp_preds.reshape(1,-1)==True)
        # # sum=0
        # # for i in range(len(temp2[1])):
        # #     if(temp2[1][i] in temp1[1] ):
        # #         sum+=1
        # # print(sum)
        # temp_recall1=[]
        # temp_prec1=[]
        # temp_recall2=[]
        # temp_prec2=[]
        # temp_F1=[]
        # for i in range(temp_preds.shape[1]):

        #     this_tp = (temp_targets[:,i].reshape(1,-1) + temp_preds[:,i].reshape(1,-1)).eq(2).sum()
        #     this_fp = (temp_preds[:,i].reshape(1,-1) - temp_targets[:,i].reshape(1,-1)).eq(1).sum()
        #     this_fn = (temp_preds[:,i].reshape(1,-1) - temp_targets[:,i].reshape(1,-1) ).eq(-1).sum()
        #     this_tn = (temp_targets[:,i].reshape(1,-1) + temp_preds[:,i].reshape(1,-1)).eq(0).sum()

        #     this_prec = this_tp.float() / (
        #     this_tp + this_fp).float() * 100.0 if (this_tp + this_fp) != 0 else 0.0
        #     this_rec = this_tp.float() / (
        #     this_tp + this_fn).float() * 100.0 if (this_tp + this_fn) != 0 else 0.0
        
        #     recall=mt.recall_score(temp_targets[:,i],temp_preds[:,i]) * 100.
        #     F1 = mt.f1_score(temp_targets[:,i],temp_preds[:,i]) * 100.
        #     precision=mt.precision_score(temp_targets[:,i],temp_preds[:,i]) * 100.

        #     temp_recall1.append(this_rec)
        #     temp_prec1.append(this_prec)
        #     temp_recall2.append(recall)
        #     temp_prec2.append(precision)
        #     temp_F1.append(F1)

        # print("recall one is %f" % np.mean(np.array(temp_recall1)))
        # print("recall two is %f" % np.mean(np.array(temp_recall2)))
        # print("prec one is %f" % np.mean(np.array(temp_prec1)))
        # print("prec two is %f" % np.mean(np.array(temp_prec2)))
        # print("F1 is %f" % np.mean(np.array(temp_F1)))
        # for i in range(len(preds)):
        #     preds[i]=(preds[i]>0.5).int()
        target=targets[0]
        pred=preds[0].data.gt(0.5).long()
        for i in range(1, len(preds)):
            target=torch.cat((target,targets[i]),0)
            pred=torch.cat((pred,(preds[i].data.gt(0.5).long()>0.5).int()),0)

        tp = (pred + target).eq(2).sum(dim=0)
        fp = (pred - target).eq(1).sum(dim=0)
        fn = (pred - target).eq(-1).sum(dim=0)
        tn = (pred + target).eq(0).sum(dim=0)
        # count = input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0


        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
               i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        
        print(
                  'Precision {:.2f}\t Recall {:.2f}'.format(
                this_prec,this_rec))
        print(
                'P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                    .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))




        return this_rec, r_o, this_prec

    def batch_loss(self, labels):
        for key, scheduler in self.head_scheduler_dict.items():
                scheduler.optimizer.zero_grad()
        for key, scheduler in self.tail_scheduler_dict.items():
                scheduler.optimizer.zero_grad()
        for key, scheduler in self.balanced_scheduler_dict.items():
                scheduler.optimizer.zero_grad()
        # head loss
        self.head_loss = 0
        self.head_loss_perf = self.criterions['PerformanceLoss'](self.logits_head, labels)
        self.head_loss_perf *=  self.criterion_weights['PerformanceLoss']
        self.head_loss += self.head_loss_perf
        # self.head_loss +=self.splc_head


        # tail loss
        self.tail_loss = 0
        self.tail_loss_perf = self.criterions['PerformanceLoss'](self.logits_tail, labels)
        self.tail_loss_perf *=  self.criterion_weights['PerformanceLoss']
        self.tail_loss += self.tail_loss_perf
        # self.tail_loss +=self.splc_tail

        # balanced loss
        self.balanced_loss=0
        # self.balanced_loss += self.erm_loss
        self.balanced_loss_perf = self.criterions['PerformanceLoss'](self.logits_balanced, labels)
        self.balanced_loss_perf *=  self.criterion_weights['PerformanceLoss']        
        self.balanced_loss+=self.splc_balanced
        self.balanced_output_loss=self.balanced_loss
        self.balanced_loss += self.balanced_loss_perf






    
    def shuffle_batch(self, x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y

    def train(self):
        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.balanced_networks['feat_model'].state_dict())
        # best_model_weights['additive_attention'] = copy.deepcopy(self.balanced_networks['additive_attention'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.balanced_networks['classifier'].state_dict())
        end_epoch = self.training_opt['num_epochs']
        best_mAP=0
        self.criterions["PerformanceLoss"].distribution_path=self.distribution_path
        self.criterions["PerformanceLoss"].create_co_occurrence_matrix(self.co_occurrence_matrix)
        self.criterions["PerformanceLoss"].create_weight(self.distribution_path)
        xERM_loss = xERMLoss(gamma=self.training_opt['gamma'])
        xERM_loss../loss/PriorFocalModifierLoss.py=self.criterions["PerformanceLoss"]
        SPLC_loss_balanced=SPLC(distribution_path=self.distribution_path)
        SPLC_loss_head=SPLC(distribution_path=self.distribution_path)
        SPLC_loss_tail=SPLC(distribution_path=self.distribution_path)
        # Loop over epochs
        for epoch in range(1, end_epoch + 1):
            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            for key, model in self.balanced_networks.items():
                model.train()
            torch.cuda.empty_cache()
            for key, scheduler in self.balanced_scheduler_dict.items():
                scheduler.step() 

            for key, model in self.head_networks.items():
                model.train()
            torch.cuda.empty_cache()
            for key, scheduler in self.head_scheduler_dict.items():
                scheduler.step() 

            for key, model in self.tail_networks.items():
                model.train()
            torch.cuda.empty_cache()
            for key, scheduler in self.tail_scheduler_dict.items():
                scheduler.step() 

            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()

            # indicate current path
            print_write([self.training_opt['log_dir']], self.log_file)
            # print learning rate
            current_lr = self.show_current_lr()
            current_lr = min(current_lr * 50, 1.0)
            preds=[]
            targets=[]
            head_loss_file="/home//project/NLT-multi-label-classification/log/head_loss_3.txt"
            tail_loss_file="/home//project/NLT-multi-label-classification/log/tail_loss_3.txt"
            balanced_loss_file="/home//project/NLT-multi-label-classification/log/balanced_loss_3.txt"
            losses = MiscMeter()
            with open(self.config["training_opt"]["no_missing_path"]) as f:
                json_file=json.load(f)
            for step, (inputs, labels, indexes) in enumerate(self.train_loader):
                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break
                
                if self.do_shuffle:
                    inputs, labels = self.shuffle_batch(inputs, labels)
                inputs, labels = inputs.cuda(), labels.cuda()          
                paths_image=self.train_loader.dataset.get_image_path(np.array(indexes).reshape(1,-1)[0])

                ground_truth=torch.zeros((self.config["training_opt"]["batch_size"], self.config["training_opt"]["num_classes"]))

                for i in range(len(paths_image)):
                    ground_truth[i,:]= torch.tensor(json_file[paths_image[i].split("/")[-1]])
                # print (ground_truth)
                # print (labels)
                # print(ground_truth)
                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):    
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(inputs, labels, phase='train')  
                    #loss
                    # self.erm_loss = 0.5*xERM_loss(self.logits_head_moving, self.logits_tail_moving, self.logits_balanced, labels)
        
                    # self.splc_head=0.2*SPLC_loss_head(self.logits_head_moving, labels, epoch, ground_truth)
                    # self.splc_tail=0.2*SPLC_loss_tail(self.logits_tail_moving, labels, epoch, ground_truth)
                    self.splc_balanced=0.2*SPLC_loss_balanced(self.logits_balanced, labels, epoch, ground_truth)
                    self.batch_loss(labels)
                     #loss bp
                    self.batch_backward(print_grad=(step % self.training_opt['display_grad_step'] == 0)) 
                    losses.update(self.balanced_loss.item())
                    
                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 1:
                        minibatch_loss_perf = self.balanced_loss_perf.item() \
                            if 'PerformanceLoss' in self.criterions else None
                        minibatch_loss_total = self.balanced_loss.item()

                        print_str = ['Epoch: [%d/%d]' 
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d' 
                                     % (step),
                                     'Minibatch_loss_performance: %.3f'
                                     % (minibatch_loss_perf) if minibatch_loss_perf else ''
                                      ]
                        with open(head_loss_file, 'a') as f:
                             print(str(self.head_loss.item()), file=f)
                        with open(tail_loss_file, 'a') as f:
                             print(str(self.tail_loss.item()), file=f)
                        with open(balanced_loss_file, 'a') as f:
                             print(str(self.balanced_output_loss.item()), file=f)
                        print_write(print_str, self.log_file)
                        loss_info = {
                            'Epoch': epoch,
                            'Step': step,
                            'Total': minibatch_loss_total,
                            'CE': minibatch_loss_perf
                        }
                        self.logger.log_loss(loss_info)
            # print("epoch %d : Tp is %f, FP is %f" %(epoch, SPLC_loss_balanced.total_TP, SPLC_loss_balanced.total_FP))
            # print ("estimated distribtuion is"+ str(SPLC_loss_balanced.distribution))
            # print ("Training performance")
            # mAP_regular_total, mAP_regular_many_shot, mAP_regular_median_shot,mAP_regular_low_shot=\
            #     shot_mAP(self.train_loader.dataset.per_class_labels,torch.cat(targets).numpy(), torch.cat(preds).numpy())
            # print (mAP_regular_total, mAP_regular_many_shot, mAP_regular_median_shot,mAP_regular_low_shot)
            # # epoch-level: reset sampler weight
            if hasattr(self.train_loader.sampler, 'get_weights'):
                self.logger.log_ws(epoch, self.train_loader.sampler.get_weights())
            if hasattr(self.train_loader.sampler, 'reset_weights'):
                self.train_loader.sampler.reset_weights(epoch)

            # After every epoch, validation
            mAP_total, mAP_many_shot, mAP_median_shot,mAP_low_shot = self.eval(epoch, phase='val')
            # self.average_mAP=(mAP_many_shot+ mAP_median_shot+mAP_low_shot)/3

            # Reset class weights for sampling if pri_mode is valid
            results= {'Epoch': epoch, 'Total Shot':mAP_total, 'Many Shot':mAP_many_shot, 'Median Shot':mAP_median_shot, 'Low Shot': mAP_low_shot}
            # Log results
            self.logger.log_acc(results)

            # Under validation, the best model need to be updated
            if self.total_mAP > best_mAP:
                best_epoch = epoch
                best_mAP = self.mAP_total
                best_model_weights['feat_model'] = copy.deepcopy(self.balanced_networks['feat_model'].state_dict())
                # best_model_weights['additive_attention'] = copy.deepcopy(self.balanced_networks['additive_attention'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.balanced_networks['classifier'].state_dict())
            print('===> Saving checkpoint')
            self.save_latest(epoch)
        print('Training Complete.')
        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_mAP, best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model
        self.save_model(epoch, best_epoch, best_model_weights, best_mAP)
        print('Done')
    


    def save_mean_embedding(self):
        # Iterate over training data, 
        # save the mean features for each class
        # save the mean features for all class

        self.saving_feature_with_label_init()

        for inputs, labels, _ in tqdm(self.data['train_labeled']):
            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                self.batch_forward(inputs, labels, phase='test')
                self.saving_feature_with_label_update(self.features, self.logits, labels)

        self.saving_feature_with_label_export(save_name='train_statistics.pth')


    def eval(self,  epoch, phase='val', save_feat=False):
        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)
 
        torch.cuda.empty_cache()
        Sig = torch.nn.Sigmoid()
        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.balanced_networks.values():
            model.eval()

        

        # feature saving initialization
        if save_feat:
            self.saving_feature_with_label_init()
        head_preds= []
        balanced_preds= []
        tail_preds= []
        targets = []
        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.val_dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):
                # In validation or testing
                self.head_features = self.head_networks['feat_model'](inputs)
                self.tail_features = self.tail_networks['feat_model'](inputs)
                self.balanced_features = self.balanced_networks['feat_model'](inputs)
                self.keys=torch.cat([self.head_features.unsqueeze(1), self.tail_features.unsqueeze(1)],1)
                # self.balanced_features=(self.balanced_features+ 0.1*self.balanced_networks['additive_attention'](self.balanced_features.unsqueeze(1),self.keys,self.keys).squeeze(1))
                self.logits_balanced,_= self.balanced_networks['classifier'](self.balanced_features, labels, self.balanced_embed_mean)
                # self.logits_head, self.logits_head_moving= self.head_networks['classifier'](self.head_features, labels, self.head_embed_mean)
                # self.logits_tail, self.logits_tail_moving= self.tail_networks['classifier'](self.tail_features, labels, self.tail_embed_mean)

                # output = Sig(self.logits_head).cpu()
                # head_preds.append(output.cpu().detach())

                output = Sig(self.logits_balanced).cpu()
                balanced_preds.append(output.cpu().detach())

                # output = Sig(self.logits_tail).cpu()
                # tail_preds.append(output.cpu().detach())

                targets.append(labels.cpu().detach())

        # feature saving export
        if save_feat:
            self.saving_feature_with_label_export(save_name='test_statistics.pth')

        # mAP_regular_total, mAP_regular_many_shot, mAP_regular_median_shot,mAP_regular_low_shot\
            # = shot_mAP(self.train_loader.dataset.per_class_labels,torch.cat(targets).numpy(), torch.cat(head_preds).numpy(), self.training_opt["head_class_number"], self.training_opt["tail_class_number"])
        #
        # print ("Head classifier performance.")
        # average_map=(mAP_regular_many_shot+ mAP_regular_median_shot+mAP_regular_low_shot)/3
        # recall,F1,precision=self.get_recall_precision(targets, balanced_preds)
        # print_str = ['\n Head classifier  Epoch {}, mAP score Total Shot:{:.2f}'.format(epoch + 1, mAP_regular_total),
        #              'Many Shot :{:.2f}'.format(mAP_regular_many_shot),
        #              'Median Shot :{:.2f}'.format(mAP_regular_median_shot),
        #              'Low Shot :{:.2f}'.format(mAP_regular_low_shot),
        #              'Average mAP :{:.2f}'.format(average_map),
        #              'Recall :{:.2f}'.format(recall),
        #              'Precision :{:.2f}'.format(precision),
        #              'F1 :{:.2f}'.format(F1),
        #              '\n']  
        # print_write(print_str, self.log_file)

        print ("balanced classifier performance.")
        mAP_regular_total, mAP_regular_many_shot, mAP_regular_median_shot,mAP_regular_low_shot\
            = shot_mAP(self.train_loader.dataset.per_class_labels,torch.cat(targets).numpy(), torch.cat(balanced_preds).numpy(), self.training_opt["head_class_number"], self.training_opt["tail_class_number"])
        average_map=(mAP_regular_many_shot+ mAP_regular_median_shot+mAP_regular_low_shot)/3
        recall,F1,precision=self.get_recall_precision(targets, balanced_preds)
        print_str = ['\n Balanced classifier Epoch {}, mAP score Total Shot:{:.2f}'.format(epoch, mAP_regular_total),
                     'Many Shot :{:.2f}'.format(mAP_regular_many_shot),
                     'Median Shot :{:.2f}'.format(mAP_regular_median_shot),
                     'Low Shot :{:.2f}'.format(mAP_regular_low_shot),
                     'Average mAP :{:.2f}'.format(average_map),
                     'Recall :{:.2f}'.format(recall),
                     'Precision :{:.2f}'.format(precision),
                     'F1 :{:.2f}'.format(F1),
                     '\n']  
        print_write(print_str, self.log_file)

        # print ("Tail classifier performance.")
        # mAP_regular_total, mAP_regular_many_shot, mAP_regular_median_shot,mAP_regular_low_shot\
        #     = shot_mAP(self.train_loader.dataset.per_class_labels,torch.cat(targets).numpy(), torch.cat(tail_preds).numpy(), self.training_opt["head_class_number"], self.training_opt["tail_class_number"])
        # average_map=(mAP_regular_many_shot+ mAP_regular_median_shot+mAP_regular_low_shot)/3
        # recall,F1,precision=self.get_recall_precision(targets, balanced_preds)
        # print_str = ['\n Tail classifier Epoch {}, mAP score Total Shot:{:.2f}'.format(epoch + 1, mAP_regular_total),
        #              'Many Shot :{:.2f}'.format(mAP_regular_many_shot),
        #              'Median Shot :{:.2f}'.format(mAP_regular_median_shot),
        #              'Low Shot :{:.2f}'.format(mAP_regular_low_shot),
        #              'Average mAP :{:.2f}'.format(average_map),
        #              'Recall :{:.2f}'.format(recall),
        #              'Precision :{:.2f}'.format(precision),
        #              'F1 :{:.2f}'.format(F1),
        #              '\n']  
        # print_write(print_str, self.log_file)
        return mAP_regular_total, mAP_regular_many_shot, mAP_regular_median_shot,mAP_regular_low_shot



    def reset_model(self, model_state):
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)

    def load_model(self, model_dir=None):
        model_dir = self.training_opt['log_dir'] if model_dir is None else model_dir
        print('Validation on the best model.')
        if not model_dir.endswith('.pth'):
            model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')
        print('Loading model from %s' % (model_dir))
        checkpoint = torch.load(model_dir)          
        model_state = checkpoint['state_dict_best']
        
        for key, model in self.networks.items():
            ##########################################
            # if loading classifier in training:
            #     1. only tuning memory embedding
            #     2. retrain the entire classifier
            ##########################################
            if 'embed' in checkpoint:
                print('============> Load Moving Average <===========')
                self.embed_mean = checkpoint['embed']
                self.embed_back = checkpoint['embed']

            if not self.test_mode and 'Classifier' in self.config['networks'][key]['def_file']:
                if 'tuning_memory' in self.config and self.config['tuning_memory']:
                    print('=============== WARNING! WARNING! ===============')
                    print('========> Only Tuning Memory Embedding  <========')
                    for param_name, param in self.networks[key].named_parameters():
                        # frezing all params only tuning memory_embeding
                        if 'embed' in param_name:
                            param.requires_grad = True
                            print('=====> Abandon Weight {} in {} from the checkpoints.'.format(param_name, key))
                            if param_name in model_state[key]:
                                del model_state[key][param_name]
                        else:
                            param.requires_grad = False
                        print('=====> Tuning: {} | {}'.format(str(param.requires_grad).ljust(5, ' '), param_name))
                    print('=================================================')
                else:
                    # Skip classifier initialization 
                    #print('================ WARNING! WARNING! ================')
                    print('=======> Load classifier from checkpoint <=======')
                    #print('===================================================')
                    #continue
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            if all([weights[k].sum().item() == x[k].sum().item() for k in weights if k in x]):
                print('=====> All keys in weights have been loaded to the module {}'.format(key))
            else:
                print('=====> Error! Error! Error! Error! Loading failure in module {}'.format(key))
            model.load_state_dict(x)
    
    def save_latest(self, epoch):
        model_weights = {}
        model_weights['feat_model'] = copy.deepcopy(self.balanced_networks['feat_model'].state_dict())
        # model_weights['additive_attention'] = copy.deepcopy(self.balanced_networks['additive_attention'].state_dict())
        model_weights['classifier'] = copy.deepcopy(self.balanced_networks['classifier'].state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights,
            'embed': self.balanced_embed_mean,
        }

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'latest_model_checkpoint.pth')
        torch.save(model_states, model_dir)
        
    def save_model(self, epoch, best_epoch, best_model_weights, best_acc):
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'embed': self.balanced_embed_mean,}

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)
            
    def output_logits(self):
        filename = os.path.join(self.training_opt['log_dir'], 'logits')
        print("Saving total logits to: %s.npz" % filename)
        np.savez(filename, 
                 logits=self.total_logits.detach().cpu().numpy(), 
                 labels=self.total_labels.detach().cpu().numpy(),
                 paths=self.total_paths)

    def saving_feature_with_label_init(self):
        self.saving_feature_container = []
        self.saving_logit_container = []
        self.saving_label_container = []
        # record number of instances for each class
        if isinstance(self.data['train'], np.ndarray):
            training_labels = np.array(self.data['train']).astype(int)
        else:
            training_labels = np.array(self.data['train'].dataset.labels).astype(int)

        train_class_count = []
        for l in np.unique(training_labels):
            train_class_count.append(len(training_labels[training_labels == l]))
        self.train_class_count = np.array(train_class_count)

    def saving_feature_with_label_update(self, features, logits, labels):
        self.saving_feature_container.append(features.detach().cpu())
        self.saving_logit_container.append(logits.detach().cpu())
        self.saving_label_container.append(labels.detach().cpu())

    
    def saving_feature_with_label_export(self, save_name='eval_features_with_labels.pth'):
        eval_features = {'features': torch.cat(self.saving_feature_container, dim=0).numpy(),
                    'labels': torch.cat(self.saving_label_container, dim=0).numpy(),
                    'logits': torch.cat(self.saving_logit_container, dim=0).numpy(),
                    'class_count': self.train_class_count}

        eval_features_dir = os.path.join(self.training_opt['log_dir'], save_name)
        torch.save(eval_features, eval_features_dir)
        print_write(['=====> Features with labels are saved as {}'.format(eval_features_dir)], self.log_file)



