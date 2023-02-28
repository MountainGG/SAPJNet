import os
import torch
import torch.nn as nn
import numpy as np
from resnet import ResModel2d, ResModel3d
from transformer import ViT
from loss import ArcLoss
from itertools import combinations
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# example： channels=[[12,12,24,48,96,192,384],[12,12,24,48,96,192,384],[12,12,24,48,96,192,384]]

class SAPJNet(nn.Module):
    def __init__(self, pool, patch, depth, channels):
        super(SAPJNet, self).__init__()
        self.CNet = ResModel3d(channels[0])
        self.SNet = ResModel3d(channels[1])
        self.MNet = ResModel2d(channels[2])
        self.merge = ViT(pool, patch, channels[0][-1], depth, dropout=0.2)

    def forward(self, in1, in2, in3):
        y1 = self.CNet(in1.cuda())
        y2 = self.SNet(in2.cuda())
        y3 = self.MNet(in3.cuda())
        y = torch.cat([y1, y2, y3.unsqueeze(-1)], -1)
        s1 = y.view(y.size(0), -1)
        s2 = self.merge(y)
        sim1 = torch.Tensor(np.zeros([1, len(in1)])).cuda()
        sim2 = torch.Tensor(np.zeros([1, len(in1)])).cuda()
        for i in range(1, len(in1)):
            sim1[:, i] = torch.cosine_similarity(s1[0], s1[i], dim=0)
            sim2[:, i] = torch.cosine_similarity(s2[0], s2[i], dim=0)
        out1 = sim1[:, 1:]
        out2 = sim2[:, 1:]
        return out1, out2


class SAPJNet_A(nn.Module):
    def __init__(self, pool, patch, depth, channels):
        super(SAPJNet_A, self).__init__()
        self.CNet = ResModel3d(channels[0])
        self.SNet = ResModel3d(channels[1])
        self.MNet = ResModel2d(channels[2])
        self.merge = ViT(pool, patch, channels[0][-1], depth, dropout=0.2)
        self.loss = ArcLoss(pool**2*patch, 2)

    def forward(self, in1, in2, in3, label):
        y1 = self.CNet(in1.cuda())
        y2 = self.SNet(in2.cuda())
        y3 = self.MNet(in3.cuda())
        y = torch.cat([y1, y2, y3.unsqueeze(-1)], -1)
        y = self.merge(y)
        out, loss = self.loss(y, label)
        return out, loss


class Train():
    def __init__(self, pool, patch, depth, channels, rate):
        self.Net = SAPJNet(pool, patch, depth, channels).cuda()
        self.CELoss = nn.CrossEntropyLoss().cuda()
        self.opt = torch.optim.AdamW(self.Net.parameters(), lr=rate)

    def train_pah(self, data_c, data_s, data_m, epoch, path):
        self.Net.train()
        for e in range(epoch):
            loss = torch.zeros(1).cuda()
            shuffle_ix = np.random.permutation(data_c[0].shape[0])
            d1_c = data_c[0][shuffle_ix]
            d1_s = data_s[0][shuffle_ix]
            d1_m = data_m[0][shuffle_ix]
            d2_c = data_c[1]
            d2_s = data_s[1]
            d2_m = data_m[1]
            for i, j in combinations(np.arange(data_c[0].shape[0]), 2):
                batch1_c = torch.Tensor([d1_c[i], d1_c[j], d2_c[j]])
                batch1_s = torch.Tensor([d1_s[i], d1_s[j], d2_s[j]])
                batch1_m = torch.Tensor([d1_m[i], d1_m[j], d2_m[j]])
                batch2_c = torch.Tensor([d2_c[i], d1_c[j], d2_c[j]])
                batch2_s = torch.Tensor([d2_s[i], d1_s[j], d2_s[j]])
                batch2_m = torch.Tensor([d2_m[i], d1_m[j], d2_m[j]])
                o11, o12 = self.Net(batch1_c, batch1_s, batch1_m)
                o21, o22 = self.Net(batch2_c, batch2_s, batch2_m)
                lbl = torch.arange(0, 2, dtype=torch.long).cuda()
                loss = self.CELoss(torch.cat((o11, o21), 0), lbl)
                loss += self.CELoss(torch.cat((o12, o22), 0), lbl)
                loss.backward()
            print('loss={:.6f}'.format(loss.item()))
            self.opt.step()
            self.opt.zero_grad()
            torch.save(self.Net.state_dict(), path +
                       'checkpoint{:2d}.pth'.format(e))


class Train_A():
    def __init__(self, pool, patch, depth, channels, rate):
        self.Net = SAPJNet_A(pool, patch, depth, channels).cuda()
        self.opt = torch.optim.AdamW(self.Net.parameters(), lr=rate)

    def train_pah_A(self, data_c, data_s, data_m, epoch, batch, _path, path):
        self.Net.train(True)
        self.Net.load_state_dict(torch.load(_path), strict=False)
        d1_c = data_c[0]
        d1_s = data_s[0]
        d1_m = data_m[0]
        d2_c = data_c[1]
        d2_s = data_s[1]
        d2_m = data_m[1]
        for e in range(epoch):
            traindat = TensorDataset(d1_c, d1_s, d1_m, d2_c, d2_s, d2_m)
            trainload = DataLoader(traindat, batch_size=batch, shuffle=True)
            for a, b, c, d, e, f in trainload:
                _, loss1 = self.Net(
                    a, b, c, torch.zeros(batch, dtype=torch.long))
                _, loss2 = self.Net(
                    d, e, f, torch.ones(batch, dtype=torch.long))
                loss = (loss1 + loss2) / 2
                loss.backward()
                print('loss={:.6f}'.format(loss.item()))
                self.opt.step()
                self.opt.zero_grad()
                torch.save(self.Net.state_dict(), path +
                           'checkpoint{:2d}.pth'.format(e))


class Val_A():
    def __init__(self, pool, patch, depth, channels):
        self.Net = SAPJNet_A(pool, patch, depth, channels).cuda()

    def val_pah_A(self, data_c, data_s, data_m, _path):
        self.Net.train(False)
        d1_c = data_c[0]
        d1_s = data_s[0]
        d1_m = data_m[0]
        d2_c = data_c[1]
        d2_s = data_s[1]
        d2_m = data_m[1]
        traindat = TensorDataset(d1_c, d1_s, d1_m, d2_c, d2_s, d2_m)
        trainload = DataLoader(traindat, batch_size=1)
        ws = os.listdir(_path)
        ws.sort()
        for w in ws:
            self.Net.load_state_dict(torch.load(_path+w))
            pred = []
            for a, b, c, d, e, f in trainload:
                cls1, _ = self.Net(a, b, c, torch.zeros(1, dtype=torch.long))
                cls2, _ = self.Net(d, e, f, torch.ones(1, dtype=torch.long))
                pred.append(torch.argmax(cls1).item())
                pred.append(torch.argmax(cls2).item())
            res = np.array(pred)
            tru = np.repeat(np.array([[0], [1]]),
                            data_c[0].shape[0], 1).T.flatten()
            print(w, end=' ')
            print(confusion_matrix(tru, res))

