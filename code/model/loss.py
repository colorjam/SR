import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import squeezenet1_1
import torch.nn.functional as F

class Robust_loss(nn.Module):
    """Robust Loss."""
    def __init__(self):
        super(Robust_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        n = X.size()[0]
        diff = X - Y
        # print(diff)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error) 
        return loss

class Squeeze_loss(nn.Module):
    """Perceptual Loss"""
    def __init__(self):
        super().__init__()
        self.squeezenet = squeezenet1_1(pretrained=True).features
        for param in self.squeezenet.parameters():
            param.requires_grad = False

    def _extract_features(self, x, cnn):
        features = []
        prev_feat = x
        for i, module in enumerate(cnn._modules.values()):
            next_feat = module(prev_feat)
            features.append(next_feat)
            prev_feat = next_feat
        return features

    def forward(self, sr, hr):
        _, c, h, w = sr.shape
        layer = 8
        sr_features = self._extract_features(sr, self.squeezenet)
        hr_features = self._extract_features(hr, self.squeezenet)
        loss = F.mse_loss(sr_features[layer], hr_features[layer]) / (c * h * w)
        return loss
    