import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcLoss(nn.Module):

    def __init__(self, in_features, out_features, eps=1e-7, s=None, m=None):
        super(ArcLoss, self).__init__()
        self.s = 64.0 if not s else s
        self.m = 0.5 if not m else m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        x[0, -48:] = 0
        wf = self.fc(x)
        numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(
            wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0)
                          for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + \
            torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return wf, -torch.mean(L)
