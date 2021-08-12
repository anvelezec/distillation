import os
import torch
from torch.nn.modules import module
from torch.optim import optimizer
from torchvision import datasets, models, transforms

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from distillation.train_step import train_model
from loss.cross_entropy_distill import cross_entropy_distill

import joblib 

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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                            shuffle=True, num_workers=1)
            for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for i in range(1, 5):
    print("Run = {}".format(i))

    # Defines teacher and student
    if device.type == "cuda":
        teacher = torch.load("models/resnet50_bees.pt")
    else:
        teacher = torch.load("models/resnet50_bees.pt", map_location=torch.device('cpu'))
    student = models.resnet18(pretrained=True)

    for param in student.parameters():
            param.requires_grad = False

    num_ftrs = student.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    student.fc = nn.Linear(num_ftrs, 2)
    student = student.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(student.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model, results = train_model(student, teacher, cross_entropy_distill, optimizer_ft, exp_lr_scheduler, device, dataloaders, 25, dataset_sizes, alpha=0.5)

    with open("results/v" + str(i) + "_0.5_32.joblib", "wb") as file:
        joblib.dump(results, file)