import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from pytorch_lightning import LightningModule


class Jigsaw_Solver(LightningModule):
    def __init__(self):
        super(Jigsaw_Solver, self).__init__()
        segformer = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        segformer.decode_head.classifier = nn.Conv2d(768, 16, kernel_size=(1, 1), stride=(1, 1))
        
        self.segformer = segformer
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.reduce = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.segformer(x).logits
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.reduce(x)
        return x
