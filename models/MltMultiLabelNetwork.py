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
from torch.cuda.amp import GradScaler, autocast
from loss.xERM_loss import xERMLoss
from sklearn.decomposition import IncrementalPCA
from utils.helper_functions import CutoutPIL, ModelEma, add_weight_decay,shot_mAP
from datetime import datetime
import sklearn.metrics as mt
from typing import Tuple, List
# from loss.SpccLoss import SPLC
from utils.util import *
scaler = GradScaler()
class MLTModel ():
    def __init__(self, config, train_loader, val_dataloader, test=False):
        self.config = config
        self.training_opt = self.config['training_opt']
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config['shuffle'] if 'shuffle' in config else False
        self.train_loader=train_loader
        self.val_dataloader=val_dataloader
        self.distribution_path=self.training_opt['distribution_path']
        self.co_occurrence_matrix=self.training_opt['co_occurrence_matrix']
        # Setup logger
        self.logger = Logger(self.training_opt['log_dir'])
        self.average_mAP=0
        # init moving average
        self.embed_mean = torch.zeros(int(self.training_opt['feature_dim'])).numpy()
        self.mu = 0.9
        self.Sig = torch.nn.Sigmoid()
        
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
            self.init_optimizers(self.model_optim_params_dict)
            self.init_criterions()
            
            # Set up log file
            self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
            self.logger.log_cfg(self.config)
        else:
            self.log_file = None
        
    def init_models(self, optimizer=True):
        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_dict = {}
        self.model_optim_named_params = {}

        print("Using", torch.cuda.device_count(), "GPUs.")
        
        for key, val in networks_defs.items():
            # Networks
            def_file = val['def_file']
            model_args = val['params']
            model_args.update({'test': self.test_mode})

            self.networks[key] = source_import(def_file).create_model(**model_args)
            self.networks[key] = nn.DataParallel(self.networks[key]).cuda()

            if 'fix' in val and val['fix']:
                print('Freezing weights of module {}'.format(key))
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except final fc layer
                    if 'fc' not in param_name:
                        param.requires_grad = False
                print('=====> Freezing: {} | False'.format(key))
            
            if 'fix_set' in val:
                for fix_layer in val['fix_set']:
                    for param_name, param in self.networks[key].named_parameters():
                        if fix_layer == param_name:
                            param.requires_grad = False
                            print('=====> Freezing: {} | {}'.format(param_name, param.requires_grad))
                            continue


            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_named_params.update(dict(self.networks[key].named_parameters()))
            self.model_optim_params_dict[key] = {'params': self.networks[key].parameters(),
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

    def init_optimizers(self, optim_params_dict):
        '''
        seperate backbone optimizer and classifier optimizer
        by Kaihua
        '''
        networks_defs = self.config['networks']
        self.model_optimizer_dict = {}
        self.model_scheduler_dict = {}

        for key, val in networks_defs.items():
            # optimizer
            if 'optimizer' in self.training_opt and self.training_opt['optimizer'] == 'adam':
                print('=====> Using Adam optimizer')
                optimizer = optim.Adam([optim_params_dict[key],])
                
            else:
                print('=====> Using SGD optimizer')
                optimizer = optim.SGD([optim_params_dict[key],])
            self.model_optimizer_dict[key] = optimizer

            # scheduler
            scheduler_params = val['scheduler_params']
            if scheduler_params['coslr']:
                print("===> Module {} : Using coslr eta_min={}".format(key, scheduler_params['endlr']))
                self.model_scheduler_dict[key] = torch.optim.lr_scheduler.CosineAnnealingLR(
                                    optimizer, self.training_opt['num_epochs'], eta_min=scheduler_params['endlr'])
            elif scheduler_params['warmup']:
                print("===> Module {} : Using warmup".format(key))
                self.model_scheduler_dict[key] = WarmupMultiStepLR(optimizer, scheduler_params['lr_step'], 
                                                    gamma=scheduler_params['lr_factor'], warmup_epochs=scheduler_params['warm_epoch'])
            else:
                print(f'===> Module {key}: Using stepLR')
                self.model_scheduler_dict[key] = optim.lr_scheduler.StepLR(optimizer,
                                                                                    step_size=scheduler_params['step_size'],
                                                                                  gamma=scheduler_params['gamma'])
            # else: 
            #     print(f'===> Module {key}: Using OneCycleLR')
            #     self.model_scheduler_dict[key] = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-1, steps_per_epoch=scheduler_params['step_size'],
            #                                                                       epochs=self.training_opt['num_epochs'], pct_start=0.2)
                  
        return

    def show_current_lr(self):
        max_lr = 0.0
        for key, val in self.model_optimizer_dict.items():
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

        # Calculate Features

        self.features = self.networks['feat_model'](inputs)

        if self.apply_pca:
            if phase=='train' and self.features.shape[0] > 0:
                self.pca.partial_fit(self.features.cpu().numpy())
            else:
                pca_feat = self.pca.transform(self.features.cpu().numpy())
                pca_feat[:, 0] = 0.0
                new_feat = self.pca.inverse_transform(pca_feat)
                self.features = torch.from_numpy(new_feat).float().to(self.features.device)

        # with torch.no_grad():
        #     temp_prediction = self.Sig(self.features).cpu()       
        #     print (temp_prediction)

        # update moving average
        if phase == 'train':
            self.embed_mean = self.mu * self.embed_mean + self.features.detach().mean(0).view(-1).cpu().numpy()

        # If not just extracting features, calculate logits
        if not feature_ext:
            # self.logits=self.features
            cont_eval = 'continue_eval' in self.training_opt and self.training_opt['continue_eval'] and phase != 'train'
            self.logits, self.logits_moving= self.networks['classifier'](self.features, labels, self.embed_mean)

    def batch_backward(self, print_grad=False):
        # Zero out optimizer gradients
        for key, optimizer in self.model_optimizer_dict.items():
            optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        
        self.loss.backward()
        
        

        # display gradient
        if self.training_opt['display_grad']:
            print_grad_norm(self.model_optim_named_params, print_write, self.log_file, verbose=print_grad)
        # Step optimizers
        for key, optimizer in self.model_optimizer_dict.items():
            optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels, epoch):
        self.loss = 0
        
        # First, apply performance loss
        if 'PerformanceLoss' in self.criterions.keys():
            self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels)
            self.loss_perf *=  self.criterion_weights['PerformanceLoss']
            self.loss += self.loss_perf
            # self.loss +=self.splc

    
    def shuffle_batch(self, x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y

    def train(self):
        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        end_epoch = self.training_opt['num_epochs']
        best_mAP=0
        self.criterions["PerformanceLoss"].distribution_path=self.distribution_path
        self.criterions["PerformanceLoss"].create_co_occurrence_matrix(self.co_occurrence_matrix)
        self.criterions["PerformanceLoss"].create_weight(self.distribution_path)
        # SPLC_loss=SPLC(distribution_path=self.distribution_path)
        # Loop over epochs
        for epoch in range(1, end_epoch + 1):
            for key, model in self.networks.items():
                model.train()

            torch.cuda.empty_cache()
            
            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            for key, scheduler in self.model_scheduler_dict.items():
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
            # with open(self.config["training_opt"]["train_annatation_path"]) as f:
                # json_file=json.load(f)
            for step, (inputs, labels, indexes) in enumerate(self.train_loader):
                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break
                if self.do_shuffle:
                    inputs, labels = self.shuffle_batch(inputs, labels)
                inputs, labels = inputs.cuda(), labels.cuda()
                
                # paths_image=self.train_loader.dataset.get_image_path(np.array(indexes).reshape(1,-1)[0])
                # ground_truth=torch.zeros((self.config["training_opt"]["batch_size"], self.config["training_opt"]["num_classes"]))

                # for i in range(len(paths_image)):
                    # ground_truth[i,:]= torch.tensor(json_file[paths_image[i].split("/")[-1]])
                    
                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):    
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(inputs, labels, phase='train')
                    # self.splc=0.2*SPLC_loss(self.logits, labels, epoch)
                    self.batch_loss(labels, epoch)
                    self.batch_backward(print_grad=(step % self.training_opt['display_grad_step'] == 0))

                    
                    
                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:

                        minibatch_loss_route = self.loss_route.item() \
                            if 'RouteWeightLoss' in self.criterions.keys() else None
                        minibatch_loss_perf = self.loss_perf.item() \
                            if 'PerformanceLoss' in self.criterions else None
                        minibatch_loss_total = self.loss.item()

                        print_str = ['Epoch: [%d/%d]' 
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d' 
                                     % (step),
                                     'Minibatch_loss_route: %.3f' 
                                     % (minibatch_loss_route) if minibatch_loss_route else '',
                                     'Minibatch_loss_performance: %.3f'
                                     % (minibatch_loss_perf) if minibatch_loss_perf else ''
                                      ]
                        print_write(print_str, self.log_file)

                        loss_info = {
                            'Epoch': epoch,
                            'Step': step,
                            'Total': minibatch_loss_total,
                            'CE': minibatch_loss_perf,
                            'route': minibatch_loss_route,
                        }

                        self.logger.log_loss(loss_info)

                # batch-level: sampler update
                if hasattr(self.train_loader.sampler, 'update_weights'):
                    if hasattr(self.train_loader.sampler, 'ptype'):
                        ptype = self.datatrain_loader.sampler.ptype 
                    else:
                        ptype = 'score'
                    ws = get_priority(ptype, self.logits.detach(), labels)

                    inlist = [indexes.cpu().numpy(), ws]
                    if self.training_opt['sampler']['type'] == 'ClassPrioritySampler':
                        inlist.append(labels.cpu().numpy())
                    self.train_loader.sampler.update_weights(*inlist)
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
            if epoch>0:
                mAP_total, mAP_many_shot, mAP_median_shot,mAP_low_shot = self.eval(epoch, phase='val')
                self.average_mAP=(mAP_many_shot+ mAP_median_shot+mAP_low_shot)/3

                # Reset class weights for sampling if pri_mode is valid
                results= {'Epoch': epoch, 'Total Shot':mAP_total, 'Many Shot':mAP_many_shot, 'Median Shot':mAP_median_shot, 'Low Shot': mAP_low_shot}
                # Log results
                self.logger.log_acc(results)

                # Under validation, the best model need to be updated
                if self.average_mAP > best_mAP:
                    best_epoch = epoch
                    best_mAP = self.average_mAP
                    best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                    best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
                
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


    def eval(self, epoch, phase='val', save_feat=False):
        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)
 
        torch.cuda.empty_cache()
        
        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        

        # feature saving initialization
        if save_feat:
            self.saving_feature_with_label_init()
        preds= []
        targets = []
        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.val_dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):
                # In validation or testing
                self.batch_forward(inputs, labels, phase="val")
                # feature saving update
                if save_feat:
                    self.saving_feature_with_label_update(self.features, self.logits, labels)
                output = self.Sig(self.logits).cpu()
                preds.append(output.cpu().detach())
                targets.append(labels.cpu().detach())

        # feature saving export
        if save_feat:
            self.saving_feature_with_label_export(save_name='test_statistics.pth')

        mAP_regular_total, mAP_regular_many_shot, mAP_regular_median_shot,mAP_regular_low_shot\
            = shot_mAP(self.train_loader.dataset.per_class_labels,torch.cat(targets).numpy(), torch.cat(preds).numpy(), self.training_opt["head_class_number"], self.training_opt["tail_class_number"])
        
        print_str = ['\n Epoch {}, mAP score Total Shot:{:.2f}'.format(epoch + 1, mAP_regular_total),
                     'Many Shot :{:.2f}'.format(mAP_regular_many_shot),
                     'Median Shot :{:.2f}'.format(mAP_regular_median_shot),
                     'Low Shot :{:.2f}'.format(mAP_regular_low_shot),
                     '\n']
        print_write(print_str, self.log_file)
        
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
        model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        # model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights,
            'embed': self.embed_mean,
        }

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'latest_model_checkpoint.pth')
        torch.save(model_states, model_dir)
        
    def save_model(self, epoch, best_epoch, best_model_weights, best_acc):
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'embed': self.embed_mean,}

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_tail_voc_checkpoint.pth')

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



