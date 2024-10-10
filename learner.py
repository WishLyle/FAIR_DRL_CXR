from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tensorboardX
import os
import torch.optim as optim
import sys
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score
import models
import utils
from datasets import get_dataset
import datetime


class learner():
    def __init__(self, args):
        self.args = args
        self.train_df, self.test_df, self.white_test_df, self.other_test_df = utils.get_dataframe(args,path=args.dataframe_path)
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = "cuda:{}".format(args.device) if args.device != -1 else 'cpu'
        self.train_loader, self.test_loader, self.w_loader, self.o_loader = None, None, None, None
        self.disease = args.disease
        self.w_a, self.o_a, self.l_a, self.e_a = 0.0, 0.0, 0.0, 0.0
        save_dir, save_name = None, None
        self.train_set = None

        # for output
        self.confusion_mat = []
        self.test_accurate = 0.00
        self.roc_auc, self.f1 = 0.00, 0.00

        # record start time
        init_time = datetime.datetime.now()
        init_time_string = init_time.strftime("%Y-%m-%d %H:%M")
        save_dir = './saved_result/{}/'.format(self.args.mode)
        save_name = '{}.txt'.format(self.args.exp_name + init_time_string)
        self.result_save_path = os.path.join(save_dir, save_name)
        utils.make_dir(save_dir)

        # Dataloader
        if args.model == 'MLP':
            dataset_tag = 'CT_L'
        else:
            dataset_tag = 'CT'
        train_dataset = get_dataset(dataset_tag, self.train_df, self.data_path, 'train')
        val_dataset = get_dataset(dataset_tag, self.test_df, self.data_path, 'test')
        self.train_set = train_dataset

        w_dataset = get_dataset(dataset_tag, self.white_test_df, self.data_path, 'test')
        o_dataset = get_dataset(dataset_tag, self.other_test_df, self.data_path, 'test')

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=self.num_workers, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True,
                                                      num_workers=self.num_workers, pin_memory=True)
        self.w_loader = torch.utils.data.DataLoader(w_dataset, batch_size=self.batch_size, shuffle=True,
                                                    num_workers=self.num_workers, pin_memory=True)
        self.o_loader = torch.utils.data.DataLoader(o_dataset, batch_size=self.batch_size, shuffle=True,
                                                    num_workers=self.num_workers, pin_memory=True)
        # TensorBoardX
        if args.tensorboard == 1:
            self.writer = tensorboardX.SummaryWriter(f'result/summary/{args.exp_name}/{args.disease}')

    def write_result(self, string):
        with open(self.result_save_path, 'a+') as file:
            sys.stdout = file
            print(string)
        sys.stdout = sys.__stdout__

    def get_model(self, model_name, num_classes=2):
        if model_name == 'MLP':
            model = models.MLP(num_classes=num_classes)
        elif model_name == 'ResNet':
            model = models.ResNet34(num_classes=num_classes)
        elif model_name == 'DenseNet':
            model = models.DenseNet121(num_classes=num_classes)
        return model

    def get_test_loader(self, key):
        if key == 'l':
            test_loader = self.val_loader
        elif key == 'w':
            test_loader = self.w_loader
        elif key == 'o':
            test_loader = self.o_loader
        return test_loader

    def output_result(self, key, acc, auc, f1, TP, FN, FP, TN):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        with open(self.result_save_path, 'a+') as file:
            sys.stdout = file
            # time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            # print('{},ACC,AUC,F1'.format(time))
            # print(',Acc,Auc,F1,Recall,Precision,TP,FN,FP,TN')
            print(
                '{},{:6f},{:6f},{:6f},{:6f},{:6f},{},{},{},{}'.format(key, acc, auc, f1, recall, precision, TP, FN, FP,
                                                                      TN))
        sys.stdout = sys.__stdout__

    def GetSavePath_pth(self, suffix='model'):

        save_dir = r'./checkpoints/{}/{}/'.format(self.args.mode, self.args.model)
        utils.make_dir(save_dir)
        save_name = str(self.disease) + '_' + '{}'.format(suffix) + datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M")
        return save_dir + save_name + r'_acc.pth', save_dir + save_name + r'_auc.pth', save_dir + save_name + r'_f1.pth'

    def test_basic(self, path_name, key):
        device = self.device
        model = self.get_model(self.args.model).to(device)

        model.load_state_dict(torch.load(path_name))
        model.eval()

        test_loader = self.get_test_loader(key)

        ground_truth = []
        predict_score = []
        predict_label = []
        test_num = torch.zeros(1).to(device)
        ac = torch.zeros(1).to(device)
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                index, _images, _labels, _races = test_data
                outputs = model(_images.to(device))
                scores = outputs[:, 1]
                predict_y = torch.max(outputs, dim=1)[1]
                predict_label += predict_y.cpu().numpy().tolist()
                # print('score={},y={}'.format(scores, predict_y))
                ground_truth += _labels.cpu().numpy().tolist()
                predict_score += scores.cpu().numpy().tolist()
                # print(len(ground_truth),len(predict_score))# 这里有问题
                test_num += len(predict_y)
                ac += torch.eq(predict_y, _labels.to(device)).sum().item()

        test_accurate = ac / test_num
        roc_auc = roc_auc_score(ground_truth, predict_score)
        fpr, tpr, thresholds = roc_curve(ground_truth, predict_score)
        f1 = f1_score(ground_truth, predict_label, average='weighted')
        confusion_mat = confusion_matrix(ground_truth, predict_label)
        TP = confusion_mat[1, 1]
        FN = confusion_mat[1, 0]
        FP = confusion_mat[0, 1]
        TN = confusion_mat[0, 0]
        TPR = TP / (TP + FN)

        if key == 'l':
            self.l_a = test_accurate
        elif key == 'w':
            self.w_a = test_accurate
        elif key == 'o':
            self.o_a = test_accurate

        self.output_result(key, float(test_accurate), roc_auc, f1, TP, FN, FP, TN)

    def train_basic(self):
        steps = 0
        device = self.device
        epochs = self.args.epochs
        lr = self.args.lr

        best_acc = torch.zeros(1).to(device)
        best_auc = torch.zeros(1).to(device)
        best_f1 = torch.zeros(1).to(device)

        ground_truth = []
        predict_score = []
        predict_label = []# may be the crime of cuda:0 GPU utilize

        save_path_acc, save_path_auc, save_path_f1 = self.GetSavePath_pth()

        model = self.get_model(self.args.model).to(device)

        loss_function = nn.CrossEntropyLoss()

        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        train_steps = len(self.train_loader)
        for epoch in range(epochs):
            # train
            model.train()
            running_loss = torch.zeros(1).to(device)
            train_bar = tqdm(self.train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                index, images, labels, races = data

                logits = model(images.to(device))
                #

                loss = loss_function(logits, labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if self.args.tensorboard == 1:
                    self.writer.add_scalar(f"loss/steps", loss.item(), steps)
                    steps += 1
                train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)
            model.eval()
            acc = torch.zeros(1).to(device)
            val_num = torch.zeros(1).to(device)
            with torch.no_grad():
                val_bar = tqdm(self.val_loader, file=sys.stdout)
                for val_data in val_bar:
                    index, val_images, val_labels, val_races = val_data
                    outputs = model(val_images.to(device))
                    scores = outputs[:, 1]
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    predict_label += predict_y.cpu().numpy().tolist()
                    # print('score={},y={}'.format(scores, predict_y))
                    ground_truth += val_labels.cpu().numpy().tolist()
                    predict_score += scores.cpu().numpy().tolist()
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    val_num += len(predict_y)
                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)
            val_accurate = acc / val_num
            f1 = f1_score(ground_truth, predict_label, average='weighted')
            roc_auc = roc_auc_score(ground_truth, predict_score)
            fpr, tpr, thresholds = roc_curve(ground_truth, predict_score)
            print('[epoch %d] loss: %.4f  acc: %.4f  auc: %.4f  f1: %.4f' %
                  (epoch + 1, running_loss / train_steps, val_accurate, roc_auc, f1))
            # modify & simplify!
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model.state_dict(), save_path_acc)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), save_path_f1)
            if roc_auc > best_auc:
                best_auc = roc_auc
                torch.save(model.state_dict(), save_path_auc)

            if self.args.tensorboard == 1:
                self.writer.add_scalar(f"acc/epoch", val_accurate, epoch)

        date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.write_result("Time, [ {} ]  ".format(date_string))
        self.write_result(
            "Backbone,{},mode,{},Exp_Name,{}\nDisease,{}".format(self.args.model, self.args.mode, self.args.exp_name,
                                                                 self.args.disease))
        self.write_result(',Acc,Auc,F1,Recall,Precision,TP,FN,FP,TN')
        self.test_basic(save_path_acc, 'w')
        self.test_basic(save_path_acc, 'o')
        self.e_a = (self.o_a + self.w_a) / 2.0
        self.test_basic(save_path_acc, 'l')
        self.write_result('-\n')

        # self.write_result("Time [ {} ]  Disease [  {}  ] auc,ACC,AUC,F1  ".format(date_string, self.disease))

        self.test_basic(save_path_auc, 'w')
        self.test_basic(save_path_auc, 'o')
        self.e_a = (self.o_a + self.w_a) / 2.0
        self.test_basic(save_path_auc, 'l')
        self.write_result('-\n')

        # self.write_result("Time [ {} ]  Disease [  {}  ] f1,ACC,AUC,F1  ".format(date_string, self.disease))
        self.test_basic(save_path_f1, 'w')
        self.test_basic(save_path_f1, 'o')
        self.e_a = (self.o_a + self.w_a) / 2.0
        self.test_basic(save_path_f1, 'l')
        self.write_result('-\n')

    def pretrain_races(self):
        print("[--Start pretrain--]")
        steps = 0
        device = self.device
        epochs = self.args.epochs
        lr = self.args.lr

        best_acc = torch.zeros(1).to(device)
        best_auc = torch.zeros(1).to(device)
        best_f1 = torch.zeros(1).to(device)

        ground_truth = []
        predict_score = []
        predict_label = []

        save_path_acc, save_path_auc, save_path_f1 = self.GetSavePath_pth('race')

        model = self.get_model(self.args.model).to(device)

        loss_function = nn.CrossEntropyLoss()

        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        train_steps = len(self.train_loader)
        for epoch in range(epochs):
            # train
            model.train()
            running_loss = torch.zeros(1).to(device)
            train_bar = tqdm(self.train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                index, images, labels, races = data

                logits = model(images.to(device))
                #
                loss = loss_function(logits, races.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            model.eval()
            acc = torch.zeros(1).to(device)
            val_num = torch.zeros(1).to(device)
            with torch.no_grad():
                val_bar = tqdm(self.val_loader, file=sys.stdout)
                for val_data in val_bar:
                    index, val_images, val_labels, val_races = val_data
                    outputs = model(val_images.to(device))
                    scores = outputs[:, 1]
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    predict_label += predict_y.cpu().numpy().tolist()
                    # print('score={},y={}'.format(scores, predict_y))
                    ground_truth += val_races.cpu().numpy().tolist()
                    predict_score += scores.cpu().numpy().tolist()
                    acc += torch.eq(predict_y, val_races.to(device)).sum().item()
                    val_num += len(predict_y)
                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)
            val_accurate = acc / val_num
            f1 = f1_score(ground_truth, predict_label, average='weighted')
            roc_auc = roc_auc_score(ground_truth, predict_score)
            fpr, tpr, thresholds = roc_curve(ground_truth, predict_score)
            print('[epoch %d] loss: %.4f  acc: %.4f  auc: %.4f  f1: %.4f' %
                  (epoch + 1, running_loss / train_steps, val_accurate, roc_auc, f1))
            # modify & simplify!
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model.state_dict(), save_path_acc)
            if roc_auc > best_auc:
                best_auc = roc_auc
                torch.save(model.state_dict(), save_path_auc)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), save_path_f1)
            if epoch % 6 == 0:
                self.write_result(
                    "acc : {:.5f}, auc :{:.5f}, f1 : {:.5f}".format(float(best_acc), float(best_auc), float(best_f1)))
        self.write_result("{:.5f},{:.5f},{:.5f}".format(float(best_acc), float(best_auc), float(best_f1)))
        return save_path_auc

    def test_debias(self, path_name, key):
        device = self.device
        model = self.get_model(self.args.model).to(device)
        model.load_state_dict(torch.load(path_name))
        model.eval()

        test_loader = self.get_test_loader(key)

        ground_truth = []
        predict_score = []
        predict_label = []

        test_num = torch.zeros(1).to(device)
        ac = torch.zeros(1).to(device)
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                index, _images, _labels, _races = test_data
                _images = _images.to(device)
                f_b = model.extract(_images)
                pred_d = model.d_fc(f_b)
                predict_y = torch.max(pred_d, dim=1)[1]
                scores = pred_d[:, 1]
                predict_label += predict_y.cpu().numpy().tolist()

                # print('score={},y={}'.format(scores, predict_y))
                ground_truth += _labels.cpu().numpy().tolist()
                predict_score += scores.cpu().numpy().tolist()
                # print(len(ground_truth),len(predict_score))# 这里有问题
                test_num += len(predict_y)
                ac += torch.eq(predict_y, _labels.to(device)).sum().item()

        test_accurate = ac / test_num
        roc_auc = roc_auc_score(ground_truth, predict_score)
        fpr, tpr, thresholds = roc_curve(ground_truth, predict_score)
        f1 = f1_score(ground_truth, predict_label, average='weighted')
        confusion_mat = confusion_matrix(ground_truth, predict_label)
        TP = confusion_mat[1, 1]
        FN = confusion_mat[1, 0]
        FP = confusion_mat[0, 1]
        TN = confusion_mat[0, 0]
        TPR = TP / (TP + FN)
        if key == 'l':
            self.l_a = test_accurate
        elif key == 'w':
            self.w_a = test_accurate
        elif key == 'o':
            self.o_a = test_accurate

        self.output_result(key, float(test_accurate), roc_auc, f1, TP, FN, FP, TN)

    def train_debias(self):

        steps = 0
        device = self.device
        epochs = self.args.epochs
        lr = self.args.lr
        weight_decay = self.args.weight_decay
        best_acc = torch.zeros(1).to(device)
        best_auc = torch.zeros(1).to(device)
        best_f1 = torch.zeros(1).to(device)

        save_path_1_acc, save_path_1_auc, save_path_1_f1 = self.GetSavePath_pth(suffix='-1')
        save_path_2_acc, save_path_2_auc, save_path_2_f1 = self.GetSavePath_pth(suffix='-2')

        sl_b = utils.EMA(torch.LongTensor(self.train_set.races[:]), num_classes=2, alpha=self.args.ema_alpha)
        sl_i = utils.EMA(torch.LongTensor(self.train_set.races[:]), num_classes=2, alpha=self.args.ema_alpha)

        model_b = self.get_model(self.args.model).to(device)
        model_i = self.get_model(self.args.model).to(device)

        if (self.args.pretrain):
            path_r = utils.get_path_r(self.args.model)
            model_i.load_state_dict(torch.load(path_r))

        optimizer_b = optim.Adam(model_b.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_i = optim.Adam(model_i.parameters(), lr=lr, weight_decay=weight_decay)

        if (self.args.lock):
            # path_r = self.args.pathr
            # path_r = self.pretrain_races()

            for name, para in model_i.named_parameters():
                if "resnet" in name:
                    para.requires_grad_(False)
                if self.args.model == 'MLP':
                    model_i.hidden1.weight.requires_grad = False
                    model_i.act1.requires_grad = False
                    model_i.hidden2.weight.requires_grad = False
                    model_i.act2.requires_grad = False
                    model_i.hidden3.weight.requires_grad = False
                    model_i.act3.requires_grad = False
                if self.args.model == 'DenseNet':
                    if "feature" in name:
                        para.requires_grad_(False)

            pg = [p for p in model_i.parameters() if p.requires_grad]
            optimizer_i = optim.Adam(pg, lr=lr, weight_decay=weight_decay)

        scheduler_b = optim.lr_scheduler.MultiStepLR(optimizer_b, milestones=self.args.mile_d, gamma=0.1)
        scheduler_i = optim.lr_scheduler.MultiStepLR(optimizer_i, milestones=self.args.mile_r, gamma=0.1)

        CE = nn.CrossEntropyLoss(reduction='none')
        GCE = utils.GeneralizedCELoss()
        train_steps = len(self.train_loader)
        for epoch in range(epochs):
            model_b.train()
            model_i.train()
            running_loss = torch.zeros(1).to(device)
            train_bar = tqdm(self.train_loader, file=sys.stdout)
            for step, datas in enumerate(train_bar):
                index, data, label, race = datas
                index = torch.Tensor(index).long().to(device)
                data = data.to(device)
                label = label.to(device)
                race = race.to(device)

                f_i = model_i.extract(data)
                f_b = model_b.extract(data)

                f_align = torch.cat((f_b.detach(), f_i), dim=1)
                f_conflict = torch.cat((f_b, f_i.detach()), dim=1)

                pred_a = model_i.r_fc(f_align)
                pred_c = model_b.r_fc(f_conflict)

                # lda : loss_dis_align
                # ldc : loss_dis_conflict
                # ldd : loss_dis_disease

                lda = CE(pred_a, race).detach()
                ldc = CE(pred_c, race).detach()

                pred_d = model_b.d_fc(f_b)
                pred_r = model_i.r2_fc(f_i)

                # train race first ?
                lambda_d = 0
                lambda_r = self.args.lambda_r

                if epoch >= self.args.class_epoch:
                    lambda_d = self.args.lambda_d
                    lambda_r = 0

                ldd = CE(pred_d, label)
                ldr = CE(pred_r, race)

                # class-wise normalize
                sl_i.update(lda, index)
                sl_b.update(ldc, index)
                lda = sl_i.parameter[index].clone().detach()
                ldc = sl_b.parameter[index].clone().detach()
                lda = lda.to(device)
                ldc = ldc.to(device)
                for c in range(2):
                    class_index = torch.where(race == c)[0].to(self.device)
                    max_loss_align = sl_i.max_loss(c)
                    max_loss_conflict = sl_b.max_loss(c)
                    lda[class_index] /= max_loss_align
                    ldc[class_index] /= max_loss_conflict

                loss_weight = ldc / (ldc + lda + 1e-8)
                lda = CE(pred_a, race) * loss_weight.to(device)
                ldc = GCE(pred_c, race)

                loss_dis = lda.mean() + ldc.mean()
                loss = loss_dis + ldd.mean() * lambda_d + ldr.mean() * lambda_r

                # lsa : loss_swap_align
                # lsc : loss_swap_conflict
                lsa = torch.tensor([0]).float()
                lsc = torch.tensor([0]).float()

                lambda_swap = 0
                if epoch >= self.args.swap_epoch:
                    indices = np.random.permutation(f_b.size(0))
                    fb_swap = f_b[indices]
                    race_swap = race[indices]

                    f_mix_align = torch.cat((fb_swap.detach(), f_i), dim=1)
                    f_mix_conflict = torch.cat((fb_swap, f_i.detach()), dim=1)
                    # pred_mix_align
                    pma = model_i.r_fc(f_mix_align)
                    pmc = model_b.r_fc(f_mix_conflict)

                    lsa = CE(pma, race) * loss_weight.to(device)
                    lsc = GCE(pmc, race_swap)
                    lambda_swap = self.args.lambda_swap

                loss_swap = lsa.mean() + lsc.mean()
                loss = loss + lambda_swap * loss_swap

                optimizer_i.zero_grad()
                optimizer_b.zero_grad()
                loss.backward()
                optimizer_i.step()
                optimizer_b.step()
                scheduler_b.step()
                scheduler_i.step()

                running_loss += loss.item()
                if self.args.tensorboard == 1:
                    self.writer.add_scalar(f"loss/steps", loss.item(), steps)
                    self.writer.add_scalar(f"lda/steps", lda.mean(), steps)
                    self.writer.add_scalar(f"ldc/steps", ldc.mean(), steps)
                    self.writer.add_scalar(f"lsa/steps", lsa.mean(), steps)
                    self.writer.add_scalar(f"lsc/steps", lsc.mean(), steps)
                    steps += 1

                train_bar.desc = "epoch[{}/{}] loss:{:.4f} ".format(epoch + 1, epochs, loss)

            model_b.eval()
            model_i.eval()

            acc = torch.zeros(1).to(device)
            acc2 = torch.zeros(1).to(device)  # acc for race classification

            ground_truth = []
            predict_score = []
            predict_label = []

            val_num = torch.zeros(1).to(device)
            with torch.no_grad():
                val_bar = tqdm(self.val_loader, file=sys.stdout)
                for val_data in val_bar:
                    index, val_images, val_labels, val_races = val_data
                    val_images = val_images.to(device)
                    f_b = model_b.extract(val_images)
                    pred_d = model_b.d_fc(f_b)
                    predict_y = torch.max(pred_d, dim=1)[1]
                    scores = pred_d[:, 1]
                    predict_label += predict_y.cpu().numpy().tolist()
                    ground_truth += val_labels.cpu().numpy().tolist()
                    predict_score += scores.cpu().numpy().tolist()
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    f_i = model_i.extract(val_images)
                    pred_r = model_i.r2_fc(f_i)
                    predict_r = torch.max(pred_r, dim=1)[1]
                    acc2 += torch.eq(predict_r, val_races.to(device)).sum().item()

                    val_num += len(predict_y)
                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)
            val_accurate = acc / val_num
            race_acc = acc2 / val_num
            f1 = f1_score(ground_truth, predict_label, average='weighted')
            roc_auc = roc_auc_score(ground_truth, predict_score)
            print('[epoch %d] loss: %.4f  acc: %.4f  auc: %.4f  f1: %.4f' %
                  (epoch + 1, running_loss / train_steps, val_accurate, roc_auc, f1))
            if epoch >= self.args.swap_epoch:
                if val_accurate > best_acc:
                    best_acc = val_accurate
                    torch.save(model_i.state_dict(), save_path_1_acc)
                    torch.save(model_b.state_dict(), save_path_2_acc)
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model_i.state_dict(), save_path_1_f1)
                    torch.save(model_b.state_dict(), save_path_2_f1)
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    torch.save(model_i.state_dict(), save_path_1_auc)
                    torch.save(model_b.state_dict(), save_path_2_auc)

            if self.args.tensorboard == 1:
                self.writer.add_scalar(f"acc/epoch", val_accurate, epoch)

        date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.write_result("Time, [ {} ]  ".format(date_string))
        self.write_result(
            "Backbone,{},mode,{},Exp_Name,{}\nDisease,{}".format(self.args.model, self.args.mode, self.args.exp_name,
                                                                 self.args.disease))
        self.test_debias(save_path_2_acc, 'w')
        self.test_debias(save_path_2_acc, 'o')
        self.e_a = (self.w_a + self.o_a) / 2.0
        self.test_debias(save_path_2_acc, 'l')
        self.write_result('-\n')
        # self.write_result("Time [ {} ]  Disease [  {}  ] auc,ACC,AUC,F1 ".format(date_string, self.disease))
        self.test_debias(save_path_2_auc, 'w')
        self.test_debias(save_path_2_auc, 'o')
        self.e_a = (self.w_a + self.o_a) / 2.0
        self.test_debias(save_path_2_auc, 'l')
        self.write_result('-\n')
        # self.write_result("Time [ {} ]  Disease [  {}  ] f1,ACC,AUC,F1 ".format(date_string, self.disease))
        self.test_debias(save_path_2_f1, 'w')
        self.test_debias(save_path_2_f1, 'o')
        self.e_a = (self.w_a + self.o_a) / 2.0
        self.test_debias(save_path_2_f1, 'l')
        self.write_result('-\n')
