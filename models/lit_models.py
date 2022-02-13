from models import *
import torch.nn as nn
# from metrics import CharacterErrorRate,ExactMatch
from typing import List
from models.score import  exact_match_score, edit_distance
from models.models import ResNet_Transformer
import transformers
import torch

def get_device(cuda_index:int):
    """
    选择合适的device
    """
    if cuda_index==-1:
        device = torch.device('cpu')
        return device
    device = torch.device('cuda:{}'.format(cuda_index)) if torch.cuda.is_available() else torch.device('cpu')
    
    return device

class Lit_Resnet_Transformer():
    def __init__(
        self,
        args,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        dropout: float,
        num_decoder_layers: int,
        max_output_len: int,
        sos_index: int,
        eos_index: int,
        pad_index: int,
        unk_index: int,
        num_classes: int,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        milestones: List[int] = [5],
        gamma: float = 0.1,
    ):

        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma
        self.args = args
        self.models = ResNet_Transformer(
            args,
            d_model, dim_feedforward,
            nhead, dropout,
            num_decoder_layers,
            max_output_len,
            sos_index, eos_index,
            pad_index, num_classes,
        ).to(get_device(args.cuda_index))
        self.ignore_indices = [sos_index, pad_index, eos_index, unk_index]
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)


    def cal_loss(self, batch):
        imgs, targets,mask_target,length = batch
        if mask_target is not None:
            logits = self.models(imgs, mask_target[:, :-1],length)  #original mask_target[:, :-1]
        else:
            logits = self.models(imgs, targets[:, :-1],length)
        loss = self.loss_fn(logits, targets[:, 1:])

        #self.log("train/loss", loss)
        return loss
    
    def cal_loss_editDistance_exactMatch(self, batch):
        with torch.no_grad():
            imgs, targets,mask_target,length = batch
            if mask_target is not None:
                logits = self.models(imgs, mask_target[:, :-1],length)
            else:
                logits = self.models(imgs, targets[:, :-1],length)
            # logits = self.models(imgs, targets[:, :-1],lengths)
            loss = self.loss_fn(logits, targets[:, 1:]) # 算loss始终和真正的target比较
            # val_cer = self.val_cer(logits, targets)
            #self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

            preds = self.models.predict(imgs)
            val_edit_distance = edit_distance(preds, targets,self.ignore_indices)
            val_exact_match = exact_match_score(preds, targets, self.ignore_indices)
        # print(val_edit_distance, val_exact_match)
        return loss, val_edit_distance, val_exact_match

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.models.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        scheduler = transformers.get_cosine_schedule_with_warmup(                                    
            optimizer,
            num_warmup_steps=self.args.step_per_epoch//2, #int(1000*len(train_loader)/2128.875)
            num_training_steps = self.args.epoches*(self.args.step_per_epoch)
        )
        return optimizer, scheduler
    

    
