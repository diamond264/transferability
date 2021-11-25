import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from matplotlib.lines import Line2D
import config
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD
import copy

device = torch.device("cuda:{:d}".format(7) if torch.cuda.is_available() else "cpu")



class DNN():
    def __init__(self, model, tensorboard, source_dataloader, target_dataloader):
        self.opt = config.OpportunityOpt
        self.device = device

        self.tensorboard = tensorboard

        # init dataloader
        if isinstance(source_dataloader['train'], tuple): # includes unlabeled data
            source_dataloader['train'] = source_dataloader['train'][0]
        self.source_dataloader = source_dataloader # baseline must not use unlabeled data

        self.target_dataloader = target_dataloader

        # init model

        self.feature_extractor = model.Extractor().to(device)
        self.class_classifier = model.Class_Classifier().to(device)


        self.class_criterion = nn.NLLLoss()
        self.optimizer = optim.Adam([{'params': self.feature_extractor.parameters()},
                                     {'params': self.class_classifier.parameters()}], lr=self.opt['learning_rate'],
                                    weight_decay=self.opt['weight_decay'])




    def save_checkpoint(self, epoch, epoch_acc, best_acc, checkpoint_path):
        state = {}
        state['epoch'] = epoch
        state['epoch_acc'] = epoch_acc
        state['best_acc'] = best_acc
        state['feature_extractor'] = self.feature_extractor.state_dict()
        state['class_classifier'] = self.class_classifier.state_dict()
        state['optimizer'] = self.optimizer.state_dict()

        torch.save(state, checkpoint_path)

        return

    def load_checkpoint(self, checkpoint_path, is_transfer=False):
        path = checkpoint_path
        checkpoint = torch.load(path, map_location=f'cuda:{7}')

        if is_transfer:
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            self.class_classifier.load_state_dict(checkpoint['class_classifier'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint

    def get_label_and_data(self, data):
        input_of_data, class_label_of_data, domain_label_of_data = data
        input_of_data = input_of_data.to(device)
        class_label_of_data = class_label_of_data.to(device)
        domain_label_of_data = domain_label_of_data.to(device)

        return input_of_data, class_label_of_data, domain_label_of_data


    def get_loss_and_confusion_matrix(self, classifier, criterion, data, label, constant=None):

        preds_of_data = classifier(data)
        labels = [i for i in range(len(self.opt['classes']))]
        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        return loss_of_data, cm


    def get_loss_and_confusion_matrix_f1(self, classifier, criterion, data, label, constant=None):

        preds_of_data = classifier(data)
        labels = [i for i in range(len(self.opt['classes']))]
        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        return loss_of_data, cm, label.cpu(), pred_label.cpu()


    def log_loss_results(self, condition, epoch, loss_avg, suffix):

        self.tensorboard.log_scalar(condition + '/' + suffix, loss_avg, epoch)
        # print loss
        print('{:s}: [epoch : {:d}]\tLoss: {:.6f}\t({:s})'.format(
            condition, epoch, loss_avg, suffix
        ))

        return loss_avg

    def log_accuracy_results(self, condition, suffix, epoch, cm_class):

        # assert (condition in ['valid', 'test'])
        # assert (suffix in ['labeled', 'unlabeled', 'test'])

        class_accuracy = 100.0 * np.sum(np.diagonal(cm_class)) / np.sum(cm_class)
        self.tensorboard.log_scalar(condition + '/' + 'accuracy_class_' + suffix, class_accuracy, epoch)

        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, suffix, class_accuracy))
        self.tensorboard.log_confusion_matrix(condition + '_accuracy_class_' + suffix, cm_class,
                                              self.opt['classes'], epoch)

        return class_accuracy


    def train(self, epoch, method): # method = "Src", or "Tgt"
        """
        Train the model
        """

        # setup models
        self.feature_extractor.train()
        self.class_classifier.train()
        class_loss_sum = 0.0
        cm_class_sum = 0
        total_iter = 0


        # if conf.args.method in ['Src', 'Src_Tgt']:
        if method in ['Src', 'Src_Tgt']:
            num_iter = len(self.source_dataloader['train'])
            total_iter += num_iter
            for batch_idx, labeled_data in tqdm(enumerate(self.source_dataloader['train']), total=num_iter):

                inputs, cls, _ = self.get_label_and_data(
                    labeled_data)

                # compute the feature
                feature_of_labeled_data = self.feature_extractor(inputs)

                # compute the class loss of feature_of_labeled_data
                class_loss_of_labeled_data, cm = self.get_loss_and_confusion_matrix(self.class_classifier,
                                                                                   self.class_criterion,
                                                                                   feature_of_labeled_data,
                                                                                   cls)

                class_loss = class_loss_of_labeled_data
                class_loss_sum += float(class_loss * inputs.size(0))

                self.optimizer.zero_grad()
                class_loss.backward()
                self.optimizer.step()
                cm_class_sum += cm


        # if conf.args.method in ['Tgt', 'Src_Tgt']:
        if method in ['Tgt', 'Src_Tgt']:

            num_iter = len(self.target_dataloader['train'])
            total_iter += num_iter
            for batch_idx, labeled_data in tqdm(enumerate(self.target_dataloader['train']), total=num_iter):

                inputs, cls, _ = self.get_label_and_data(
                    labeled_data)

                # compute the feature
                feature_of_labeled_data = self.feature_extractor(inputs)

                # compute the class loss of feature_of_labeled_data
                class_loss_of_labeled_data, cm = self.get_loss_and_confusion_matrix(self.class_classifier,
                                                                                    self.class_criterion,
                                                                                    feature_of_labeled_data,
                                                                                    cls)
                class_loss = class_loss_of_labeled_data
                class_loss_sum += float(class_loss * inputs.size(0))

                self.optimizer.zero_grad()
                class_loss.backward()
                self.optimizer.step()

        self.log_loss_results('train', epoch=epoch, loss_avg=class_loss_sum / total_iter, suffix='loss')

    def evaluation(self, epoch, condition):

        self.feature_extractor.eval()
        self.class_classifier.eval()


        class_loss_sum = 0.0
        class_cm_labeled_sum = 0
        class_cm_test_data_sum = 0

        num_iter = len(self.target_dataloader[condition]) if not isinstance(self.target_dataloader, list) else 0  # 0 if per-condition


        with torch.no_grad():
            for batch_idx, test_data in tqdm(enumerate(self.target_dataloader[condition]), total=num_iter):

                inputs, cls, _ = self.get_label_and_data(test_data)

                feature_of_test_data = self.feature_extractor(inputs)
                class_loss_of_test_data, class_cm_test_data = self.get_loss_and_confusion_matrix(
                    self.class_classifier,
                    self.class_criterion,
                    feature_of_test_data,
                    cls)

                class_loss_sum += float(class_loss_of_test_data * inputs.size(0))
                class_cm_test_data_sum += class_cm_test_data

            # logging
            epoch_avg_loss = self.log_loss_results(condition, epoch=epoch, loss_avg=class_loss_sum / num_iter, suffix='loss')

            class_accuracy_of_test_data = self.log_accuracy_results(condition,
                                                                    'test',
                                                                    epoch,
                                                                    cm_class=class_cm_test_data_sum)



            return class_accuracy_of_test_data, epoch_avg_loss


    def validation(self, epoch):
        """
        Validate the performance of the model
        """
        class_accuracy_of_test_data, loss = self.evaluation(epoch, 'valid')

        return class_accuracy_of_test_data, loss

    def test(self, epoch):
        """
        Test the performance of the model
        """

        #### for test data
        class_accuracy_of_test_data, loss = self.evaluation(epoch, 'test')

        return class_accuracy_of_test_data, loss

