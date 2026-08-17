import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2

class WideResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = wide_resnet50_2(weights="DEFAULT")
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )

        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)

        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)

        feat3_up = F.interpolate(feat3, size=feat2.shape[2:], mode="bilinear", align_corners=False)
        combined = torch.cat([feat2, feat3_up],dim=1)

        embeddings = combined.permute(0, 2, 3, 1).contiguous()
        embeddings = embeddings.reshape(combined.shape[0],-1,combined.shape[1]) 
        
        return embeddings