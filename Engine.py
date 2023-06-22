import os
import os.path as osp
import sys
import json
from pprint import pprint

from tqdm import tqdm, trange
import numpy as np
from sklearn.metrics import precision_score, average_precision_score
import torch
import torch.optim as optim

from utils.meters import AverageMeter
from models.build import build as build_model
from dataset.build import build as build_dataset


class Engine():
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        # set seeds
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        # build dataloaders
        self.train_loader, self.val_loader, self.test_loader = build_dataset(
            self.args)
        # build model & optimizer
        self.model, self.optimizer = build_model(self.args, self.cfg)
        self.model.cuda()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=2, gamma=0.5)
        # experiment dir
        self.exp_dir = osp.join('./exp_real', self.args.exp)
        os.makedirs(self.exp_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = AverageMeter()
        for i, batch in tqdm(enumerate(self.train_loader), leave=False):
            self.optimizer.zero_grad()
            output = self.model(batch, calc_loss=True)
            loss = output['loss']
            loss.backward()
            self.optimizer.step()
            epoch_loss.update(loss.item(), self.args.batch_size)
            if i % 10 == 0:
                message = f'Train Epoch: {epoch}, loss: {epoch_loss.avg:.6f}'
                tqdm.write(message)

    @torch.no_grad()
    def eval_epoch(self, epoch=0, test=False):
        self.model.eval()
        epoch_loss = AverageMeter()
        data_loader = self.test_loader if test else self.val_loader
        pred, label = [], []
        for i, batch in tqdm(enumerate(data_loader), leave=False):
            output = self.model(batch, calc_loss=True)
            pred.append(output['pred'])
            label.append(batch['label'])
            loss = output['loss']
            epoch_loss.update(loss.item(), self.args.batch_size)
        pred = torch.cat(pred)
        label = torch.cat(label)
        label_one_hot = np.zeros_like(pred.detach().cpu().numpy())
        for i in range(label_one_hot.shape[0]):
            label_one_hot[i][label[i]] = 1
        mAP = average_precision_score(label_one_hot,pred.detach().cpu().numpy())*100
        accuracy = torch.sum(torch.argmax(pred, 1) ==
                            label.cuda())/label.shape[0]*100
        message = f'Eval Epoch: {epoch}, Acc: {accuracy:.2f}, mAP: {mAP:.2f} loss: {epoch_loss.avg:.6f}'
        tqdm.write(message)
        return accuracy, epoch_loss.avg

    def train(self):
        bst_acc = 0.0
        for epoch in range(self.args.epochs):
            print("Start Validation Epoch {}".format(epoch))
            accuracy, loss = self.eval_epoch(epoch)
            if accuracy > bst_acc:
                bst_acc = accuracy
                print(f"saving the best model to {self.exp_dir}")
                checkpoint_dict = self.model.export_module_checkpoint()
                for k, state_dict in checkpoint_dict.items():
                    save_dir = osp.join(self.exp_dir, f'{k}.pth')
                    torch.save(state_dict, save_dir)
                torch.save(self.model.state_dict(),
                        osp.join(self.exp_dir, 'bst.pth'))
            print("Start Training Epoch {}".format(epoch))
            self.train_epoch(epoch)
            self.scheduler.step()

    def test(self):
        print(f"Start testing, loading the best model from {self.exp_dir}")
        state_dict = torch.load(osp.join(self.exp_dir, 'bst.pth'))
        self.model.load_state_dict(state_dict)
        accuracy, loss = self.eval_epoch(test=True)
        save_dir = osp.join(self.exp_dir, 'results.json')
        print(f"Test ended, saving results to {save_dir}")
        with open(save_dir, 'w') as f:
            json.dump({'accuracy': float(accuracy), 'loss': loss}, f)

    def __call__(self):
        if not self.args.eval:
            self.train()
        self.test()
