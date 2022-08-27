import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Net(nn.Module):
    def __init__(self, c: int):
        super(Net, self).__init__()
        # img = images
        self.fc = nn.Linear(512, c)

    def forward(self, x):
        x = x.view(-1, 512, 7 * 7).mean(2)
        x = self.fc(x)
        return F.softmax(x, dim=1)


def build_model(class_num: int):
    vgg16 = models.vgg16(pretrained=True)
    mod = nn.Sequential(*list(vgg16.children())[:-1])
    model = nn.Sequential(mod, Net(class_num))
    return mod, model