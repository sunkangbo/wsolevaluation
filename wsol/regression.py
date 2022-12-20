import os
import timm
import torch
import torch.nn as nn

class regression_timm(nn.Module):
    def __init__(self, backbone='inception_v3', pretrained=True, feature_index=-1):
        super(regression_timm, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.feature_index = feature_index


        self.backbone = timm.create_model(model_name=backbone, pretrained=pretrained, features_only=True, out_indices=[self.feature_index])
        self.feature_channels = self.backbone.feature_info.channels()[-1]


        self.fc  = nn.Sequential(
            nn.BatchNorm1d(num_features=self.feature_channels),
            nn.Linear(in_features=self.feature_channels,out_features=self.feature_channels//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.feature_channels//2,out_features=4),
            nn.Sigmoid()
        )
        
    
    def forward(self, x):

        fea = self.backbone(x)[-1]
        fea = self.avgpool(fea)
        fea = fea.view(fea.size(0),-1)

        logit = self.fc(fea)

        return logit

