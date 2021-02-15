import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

from conf import *

#
# return: a resnet18 model

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()

        # if you have GPU, you can try resnet 18, else use a customized simplier CNN
        if args.model == 'resnet':
            # download the resnet from torchvision.model
            self.model = models.resnet18(pretrained=args.pretrain)
            # must define a new layer to override the old one, can't change in_channels only
            self.model.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, stride=2, padding=3) # 4 frames stacked in one input
            num_ftrs = self.model.fc.in_features
            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
            self.model.fc = nn.Linear(num_ftrs, 512)
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=1),
                nn.MaxPool2d(2,2),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.MaxPool2d(2,2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2),
                nn.MaxPool2d(2,2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(256, 512)
            )

        # architecture of resnet18
        self.last = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=512, out_features=args.ACTIONS)
        )
        
    
    def achitect_summary(self, input_size):
        summary(self.model, input_size)

    
    def forward(self, x):
        x = self.model(x)
        x = self.last(x)
        return x
    