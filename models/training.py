import os
from os.path import join

import torch
from tqdm.std import tqdm
from models.lit_models import Lit_Resnet_Transformer

from models import *
from models.MLM_pretrain import MaskLanguageModel
from .utils import EarlyStopping
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
from tqdm import tqdm
def save(obj, path_name):
    with open(path_name, 'wb') as file:
        pickle.dump(obj, file)
class Trainer(object):
    def __init__(self, 
                optimizer, 
                model:Lit_Resnet_Transformer, 
                lr_scheduler,
                train_loader, 
                val_loader, 
                args,
                MLM:MaskLanguageModel = None,
                init_epoch=1, 
                last_epoch=15):
        
        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.step = 0
        self.epoch = init_epoch
        self.total_step = (init_epoch - 1) * len(train_loader)
        self.last_epoch = last_epoch

        self.best_val_loss = 1e18
        self.best_exact_match = -1e18
        self.device = torch.device("cuda:{}".format(args.cuda_index) if torch.cuda.is_available() else "cpu")
        # self.total_train_step = len(train_loader)
        self.batch_train_losses = []
        self.batch_val_losses = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.val_cer_list = []
        self.val_exact_match_list = []
        self.MLM = MLM
        self.total_val_step = len(self.train_loader)

    def train(self):
        mes = "Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}, Perplexity:{:.4f}"
        # early stopping patience; how long to wait after last time validation loss improved.
        patience = 20
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        while self.epoch <= self.last_epoch:
            self.model.models.train()
            losses = 0.0
            for i,batch   in enumerate(self.train_loader):
                (images, labels_ids, lengths) = batch
                if self.MLM is not None:
                    _,mask_label_ids = self.MLM.generate_batch_mask_ids(batch)
                
                else:
                    mask_label_ids = None
                step_loss = self.train_step(i, images, labels_ids, lengths,mask_label_ids)
                self.batch_train_losses.append(step_loss)
                losses += step_loss
                # log message
                if self.step % self.args.print_freq == 0:
                    avg_loss = losses / (i+1)
                    # print(mes.format(
                    #     self.epoch, self.step, len(self.train_loader),
                    #     100 * self.step / len(self.train_loader),
                    #     avg_loss,
                    #     2 ** avg_loss
                    # ))
                    
                    print('\r', "Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}".format(
                            self.epoch, self.step, len(self.train_loader),
                            100 * self.step / len(self.train_loader),
                            avg_loss), end='', flush=True)

                # 如果是测试模式
                if self.args.test_mode and i==2:break
                

            # if self.epoch>10: # 先训练10个epoch然后再validation(因为首先10个epoch val_loss往往达不到最好,并且validation的时候几乎和training的时间一样)
            val_loss = self.validate()
            self.batch_val_losses.append(val_loss.detach().cpu().item())
            train_loss = np.average(self.batch_train_losses)
            valid_loss = np.average(self.batch_val_losses)
            self.epoch_train_losses.append(train_loss)
            self.epoch_val_losses.append(valid_loss)
            save([self.epoch_val_losses,self.val_cer_list,self.val_exact_match_list],join(self.args.save_dir,'res.pkl'))
            # clear lists to track next epoch
            self.batch_train_losses = []
            self.batch_val_losses = []

            # one epoch Finished, calcute val loss
            self.lr_scheduler.step(val_loss)

            # self.save_model('ckpt-{}-{:.4f}'.format(self.epoch, val_loss))
            self.epoch += 1
            self.step = 0

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self.model.models)

            if early_stopping.early_stop and not self.args.test_mode:
                print("Early stopping")
                # load the last checkpoint with the best model
                self.save_model('checkpoint_ckpt')
                break

            # 可能会看情况随时中断训练,因此还是每个epoch都画一下把
            self.plot_losses()
            self.plot_scores()

    def train_step(self, i, images, labels_ids, lengths,mask_label_ids=None):
        images = images.to(self.device)
        labels_ids = labels_ids.to(self.device)
        if mask_label_ids is not None:
            mask_label_ids = mask_label_ids.to(self.device)
        batch_train_loss = self.model.cal_loss([images, labels_ids,mask_label_ids,lengths])
        #batch_train_loss = batch_train_loss/self.args.accumulation_step
        batch_train_loss.backward()
        self.step += 1
        self.total_step += 1
        if self.step % self.args.accumulation_step == 0:
            #nn.utils.clip_grad_norm_(self.model.models.parameters(), max_norm=self.args.clip) #position1
            self.optimizer.step()
            self.optimizer.zero_grad()
        #nn.utils.clip_grad_norm_(self.model.models.parameters(), max_norm=self.args.clip) #position2

        # max_clip = []
        # for p in filter(lambda p: p.grad is not None, self.model.models.parameters()):
        #     #print(torch.max(p.grad.data),torch.mean(p.grad.data))
        #     max_clip.append(torch.max(p.grad.data).detach().cpu().item())
        # print(max(max_clip))
        # print(torch.max(self.model.models.transformer_decoder.))
        return batch_train_loss.detach().cpu().item()

    def validate(self):
        self.model.models.eval()
        all_val_loss = 0.0
        mes = "Epoch {}, validation average loss:{:.4f}, Perplexity:{:.4f}, EditDistance: {}, ExactMatch: {:.4f} max_ExactMatch:{:.4f}"
        with torch.no_grad():
            all_val_edit_distance = 0
            all_val_exact_match = 0
            for i,batch in tqdm(enumerate(self.val_loader),total=len(self.val_loader)):
                (images, labels_ids, lengths) = batch
                if self.MLM is not None:
                    _,mask_label_ids = self.MLM.generate_batch_mask_ids(batch)
                    mask_label_ids = mask_label_ids.to(self.device)
                else:
                    mask_label_ids = None
                images = images.to(self.device)
                labels_ids = labels_ids.to(self.device)
                batch_val_loss, batch_val_edit_distance, batch_val_exact_match = self.model.cal_loss_editDistance_exactMatch([images, labels_ids,mask_label_ids,lengths])
                all_val_edit_distance += batch_val_edit_distance
                all_val_exact_match += batch_val_exact_match
                all_val_loss+= batch_val_loss
                if self.args.test_mode and i==2:break


            avg_loss = all_val_loss / len(self.val_loader)
            avg_batch_edit_distance = all_val_edit_distance / (1+i)
            avg_batch_exact_match = all_val_exact_match / (i + 1)
            
            self.val_cer_list.append(avg_batch_edit_distance)
            self.val_exact_match_list.append(avg_batch_exact_match)
            
            if  avg_loss < self.best_val_loss and not self.args.test_mode:
                self.best_val_loss = avg_loss
                self.save_model('best_ckpt')
            if avg_batch_exact_match>self.best_exact_match:
                self.best_exact_match = avg_batch_exact_match
            print(mes.format(
                self.epoch, avg_loss, 2 ** avg_loss,avg_batch_edit_distance,  avg_batch_exact_match,self.best_exact_match
            ))
        return avg_loss

    def save_model(self, model_name):
        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        save_path = join(self.args.save_dir, model_name + '.pt')
        print("Saving checkpoint to {}".format(save_path))

        # torch.save(self.model, model_path)

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.models.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_sche': self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'args': self.args
        }, save_path)

    def plot_losses(self):
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(self.epoch_train_losses) + 1), self.epoch_train_losses,label='Training Loss')
        plt.plot(range(1, len(self.epoch_val_losses) + 1), self.epoch_val_losses,  label='Validation Loss')

        # find position of lowest validation loss
        minposs = self.epoch_val_losses.index(min(self.epoch_val_losses)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig( join(self.args.save_dir,'loss_plot.png'), bbox_inches='tight')

    def plot_scores(self):
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(self.val_cer_list) + 1), self.val_cer_list, 'g--', label = 'Edit distance(reverse)')
        plt.plot(range(1, len(self.val_exact_match_list) + 1), self.val_exact_match_list, 'r--', label = 'Exact match rate')
        plt.title('Edit distance rate and exact match rate')
        plt.xlabel('Epoch')
        plt.ylabel('rate')
        plt.legend(loc = 'lower right')
        plt.show()
        fig.savefig(join(self.args.save_dir,'score_plot.png'), bbox_inches='tight')