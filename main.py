import os
import torch
from torch.nn.modules import module
from torch.optim import optimizer
from torchvision import datasets, models, transforms

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from distilation.train_step import train_model


data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                            shuffle=True, num_workers=1)
            for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Defines teacher and student
teacher = torch.load("models/resnet50_bees.pt", map_location=torch.device('cpu'))
student = models.resnet18(pretrained=True)

for param in student.parameters():
        param.requires_grad = False

num_ftrs = student.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
student.fc = nn.Linear(num_ftrs, 2)
student = student.to(device)

# Linear combination in the loss function
def loss(s_outputs, t_outputs, labels):
    a = 0.5
    criterion_mse = torch.nn.MSELoss()
    criterion_ce = torch.nn.CrossEntropyLoss()
    return a * criterion_ce(s_outputs, labels) + (1 - a) * criterion_mse(s_outputs, t_outputs)


# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(student.parameters(), lr=0.001, momentum=0.9)


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

if __name__ == "__main__":

    train_model(student, teacher, loss, optimizer_ft, exp_lr_scheduler, device, dataloaders, 6, dataset_sizes)