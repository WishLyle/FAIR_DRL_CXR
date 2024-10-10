import torch
import torch.nn as nn
import torchvision.models
import torchvision.models as models
import torch.nn.functional as F
import torchinfo


class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34()
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.r_fc = nn.Sequential(nn.Linear(1024, num_classes), torch.nn.Softmax(dim=1))  #
        self.d_fc = nn.Sequential(nn.Linear(512, num_classes), torch.nn.Softmax(dim=1))
        self.r2_fc = nn.Sequential(nn.Linear(512, num_classes), torch.nn.Softmax(dim=1))  # classification for race

    def forward(self, x):
        x = self.resnet(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

    def extract(self, x):
        x = self.resnet(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)

        return x


class MLP(nn.Module):
    # define model elements
    def __init__(self, num_classes=2, ):
        super(MLP, self).__init__()

        self.hidden1 = nn.Linear(256 * 256, 1024)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()

        self.hidden2 = nn.Linear(1024, 1024)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()

        self.hidden3 = nn.Linear(1024, 512)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Sigmoid()

        self.fc = nn.Sequential(nn.Linear(512, num_classes), nn.Softmax(dim=1))
        self.r_fc = nn.Sequential(nn.Linear(1024, num_classes), nn.Softmax(dim=1))
        self.d_fc = nn.Sequential(nn.Linear(512, num_classes), nn.Softmax(dim=1))
        self.r2_fc = nn.Sequential(nn.Linear(512, num_classes), torch.nn.Softmax(dim=1))

    # forward propagate input
    def forward(self, X):
        X = X.view(X.size(0), -1)
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.fc(X)
        return X

    def extract(self, X):
        X = X.view(X.size(0), -1)
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X


class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        self.feature = models.densenet121().features
        self.fc0 = nn.Linear(1024, 512)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.r_fc = nn.Sequential(nn.Linear(1024, num_classes), torch.nn.Softmax(dim=1))  #
        self.d_fc = nn.Sequential(nn.Linear(512, num_classes), torch.nn.Softmax(dim=1))
        self.r2_fc = nn.Sequential(nn.Linear(512, num_classes), torch.nn.Softmax(dim=1))  # classification for race

    def forward(self, x):
        x = self.feature(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

    def extract(self, x):
        x = self.feature(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        return x
