import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=9),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        if num_classes == 10:
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 10),
            )
        elif num_classes == 200:
            self.fc_layers = nn.Sequential(
                nn.AvgPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 200),
            )
        elif num_classes == 1000:
            self.fc_layers = nn.Sequential(
                nn.AvgPool2d(kernel_size=4),
                nn.Flatten(),
                nn.Linear(9216, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 1000),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.fc_layers(x)
        return x


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.vgg = models.vgg16(pretrained=False)
        
        self.vgg.features[4] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.vgg.features[9] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0,  ceil_mode=False)
        self.vgg.features[16] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0,  ceil_mode=False)
        self.vgg.features[23] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0,  ceil_mode=False)
        self.vgg.features[30] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0,  ceil_mode=False)

        if num_classes == 10:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            )
        elif num_classes == 200:
            self.classifier = nn.Sequential(
                nn.AvgPool2d(2),
                nn.Flatten(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 200),
            )
        elif num_classes == 1000:
            self.classifier = nn.Sequential(
                nn.AvgPool2d(2),
                nn.Flatten(),
                nn.Linear(4608, 4096),
                nn.Linear(4096, 4096),
                nn.Linear(4096, 1000)
            )

    def forward(self, x):
        x = self.vgg.features(x)
        x = self.classifier(x)
        return x
    
class SecureML(nn.Module):
    def __init__(self, num_classes=10):
        super(SecureML, self).__init__()
    
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.classifier(x)
        return x
    
    
class Sarda(nn.Module):
    def __init__(self, num_classes=10):
        super(Sarda, self).__init__()
    
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(980, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
    def forward(self, x):
        x = self.classifier(x)
        return x
    
class MiniONN(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniONN, self).__init__()
    
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),  #MaxPool2d(kernel_size=2, stride=2); Crypten not implemted
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*4*16, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.classifier(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
    
        self.resnet = models.resnet18(pretrained=False)
        
    def forward(self, x):
        x = self.resnet(x)
        return x
    
class Trans(nn.Module):  
    def __init__(self, num_classes=10):
        super(Trans, self).__init__()
        # self.trans = nn.Transformer(d_model=280, nhead=2, num_encoder_layers=1, num_decoder_layers=2, 
        #                             dim_feedforward=200)
        self.trans = torch.nn.MultiheadAttention(embed_dim=280, num_heads=2)
        
    
    def forward(self, x):
        output_tensor, weights = self.trans(x,x,x)
        return output_tensor