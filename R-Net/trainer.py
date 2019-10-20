import time

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import ujson as json
import os

from dataset import MyDataset
from model import RNET
from util import convert_tokens, evaluate


class Trainer(object):
    def __init__(self, config, word_mat, char_mat, logger):
        super(Trainer, self).__init__()
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = RNET(self.config, self.device, word_mat=word_mat, char_mat=char_mat)
        # 多gpu并行的时候，一个batch的数据会均分到每块gpu上，因此batch_size = batch_size*gpu数
        if torch.cuda.device_count() > 1:
            self.device_count = torch.cuda.device_count()
            print("Let's use", self.device_count, "GPUs")
            self.model = nn.DataParallel(self.model, device_ids=[0, 1, 2, 3])

        self.load_parameters()
        self.model.to(self.device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=config.init_lr, rho=0.95, eps=1e-6)

        self.loss_save = 100
        self.patience = 0
        self.eval_best_acc = 0
        self.lr = config.init_lr

        self.digital_keys = ["context_idxs", 'ques_idxs', 'context_char_idxs', 'ques_char_idxs', 'y1', 'y2', 'id']

    def train(self):
        self.logger.info('load data...')
        start_time = time.time()
        # 加载数据化数据
        dev_loader = DataLoader(dataset=MyDataset(self.config.dev_data_file, self.digital_keys),
                                batch_size=self.config.val_num_batches * self.device_count)
        train_loader = DataLoader(dataset=MyDataset(self.config.train_data_file, self.digital_keys),
                                  batch_size=self.config.batch_size * self.device_count,
                                  shuffle=True)
        # 加载原始数据
        with open(self.config.dev_eval_file, "r") as fh:
            dev_eval_file = json.load(fh)

        time_diff = time.time() - start_time
        self.logger.info("time consumed: %dm%ds." % (time_diff // 60, time_diff % 60))

        save_model = self.config.save_dir + 'RNET_'

        self.logger.info('start train model...')
        self.model.train()
        for epoch in range(1, self.config.num_steps + 1):
            start_time = time.time()
            self.logger.info('Epoch {}/{}'.format(epoch, self.config.num_steps))
            for batch in train_loader:
                context_idxs = batch[0].to(self.device)
                ques_idxs = batch[1].to(self.device)
                context_char_idxs = batch[2].to(self.device)
                ques_char_idxs = batch[3].to(self.device)
                y1 = batch[4].to(self.device)
                y2 = batch[5].to(self.device)
                logits1, logits2 = self.model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs)
                self.model.zero_grad()
                loss = self.calc_loss(logits1, logits2, y1, y2)
                loss.backward()
                self.optimizer.step()
            time_diff = time.time() - start_time
            self.logger.info("epoch %d time consumed: %dm%ds." % (epoch + 1, time_diff // 60, time_diff % 60))

            self.logger.info('evaluate model...')
            if epoch % self.config.checkpoint == 0:
                metrics = self.eval_dev(dev_loader, dev_eval_file)
                if metrics['loss'] < self.loss_save:
                    self.loss_save = metrics['loss']
                    self.patience = 0
                else:
                    self.patience += 1

                # 当patience次loss不在减小时，调整lr
                if self.patience >= self.config.patience:
                    self.adjust_lr(self.optimizer, epoch)
                    self.loss_save = metrics['loss']
                    self.patience = 0

                self.logger.info(
                    'current exact_match：{}，f1：{}，loss：{}'.format(metrics['exact_match'], metrics['f1'],
                                                                  metrics['loss']))
                if metrics['exact_match'] > self.eval_best_acc:
                    save_filename = save_model + str(metrics['exact_match'])
                    # save model parameters
                    torch.save(self.model.state_dict(), save_filename)
                    self.eval_best_acc = metrics['exact_match']

    def eval_dev(self, dev_loader, dev_eval_file):
        self.model.eval()
        answer_dict = {}
        losses = []
        for batch in dev_loader:
            logits1, logits2 = self.model(batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device),
                                          batch[3].to(self.device))
            loss = self.calc_loss(logits1, logits2, batch[4].to(self.device), batch[5].to(self.device))
            # 开始位置
            p1 = logits1.argmax(dim=1)[0]
            # 结束位置
            p2 = logits2.argmax(dim=1)[0]
            answer_dict_, _ = convert_tokens(dev_eval_file, batch[6].to(self.device).tolist(), p1.tolist(), p2.tolist())
            answer_dict.update(answer_dict_)
            losses.append(loss)
        loss = torch.mean(torch.tensor(losses))
        metrics = evaluate(dev_eval_file, answer_dict)
        metrics['loss'] = loss
        return metrics

    def test(self):
        # 加载数据化数据
        test_loader = DataLoader(dataset=MyDataset(self.config.test_data_file, self.digital_keys))
        # 加载原始数据
        with open(self.config.test_eval_file, "r") as fh:
            test_eval_file = json.load(fh)

        answer_dict = {}
        self.logger.info('testing model...')
        answer_save_file = open(self.config.answer_file, 'w', encoding='utf-8')

        self.model.is_train = False
        self.model.eval()
        for batch in test_loader:
            logits1, logits2 = self.model(batch[0], batch[1], batch[2], batch[3])
            loss = self.calc_loss(logits1, logits2, batch[4], batch[5])
            # 开始位置
            p1 = logits1.argmax(dim=1)[0]
            # 结束位置
            p2 = logits2.argmax(dim=1)[0]
            answer_dict_, remapped_dict = convert_tokens(test_eval_file, batch[6].tolist(), p1.tolist(), p2.tolist())
            answer_dict.update(answer_dict_)
            uuid = test_eval_file[str(batch[6].tolist()[0])]["uuid"]
            # save answer
            answer_save_file.write(str(uuid + "：" + remapped_dict[uuid] + "\n"))

        metrics = evaluate(test_eval_file, answer_dict)
        self.logger.info('test exact_match：{}，f1：{}，loss：{}'.format(metrics['exact_match'], metrics['f1'], loss))

    def calc_loss(self, logits1, logits2, y1, y2):
        cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        loss1 = cross_entropy(logits1, y1).to(self.device)
        loss2 = cross_entropy(logits2, y2).to(self.device)
        return loss1 + loss2

    def adjust_lr(optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 2.0

    def load_parameters(self, model_file=""):
        if model_file != "":
            self.model.load_state_dict(torch.load(self.config.save_dir + '/' + model_file))
            self.logger.info('Load parameters file ' + model_file)
        else:
            bc = 0.0
            for file in os.listdir(self.config.save_dir):
                if file.startswith('RNET_'):
                    ac = float(file.split('_')[-1])
                    if bc < ac:
                        bc = ac
            if bc > 0.0:
                self.load_parameters('RNET_' + str(bc))
