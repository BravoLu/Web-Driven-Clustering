from __future__ import absolute_import
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist = torch.pow(inputs,2).sum(dim=1, keepdim=True).expand(n,n)

        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()
        mask = targets.expand(n,n).eq(targets.expand(n,n).t())
        dist_ap, dist_an = [] , []

        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i]==0].min())


        for i in range(len(dist_ap)):
            dist_ap[i] = torch.unsqueeze(dist_ap[i], dim=-1)
        for i in range(len(dist_an)):
            dist_an[i] = torch.unsqueeze(dist_an[i], dim=-1)

        #print("dist_ap:{}".format(dist_ap[0].size()))
        #print(dist_an)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        #dist_ap = torch.stack(dist_ap)
        #dist_an = torch.stack(dist_an)

        y = dist_an.data.new()
        # Constructs a new tensor of the same data type as self tensor.
        # For CUDA tensors, this method will create new tensor on the same device as this tensor.

        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
