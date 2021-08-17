import os
from typing import Tuple
import torch
from torch.nn.modules import module
from torch.optim import optimizer
from torchvision import datasets, models, transforms

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from distillation.train_step import train_model
from loss.cross_entropy_distill import cross_entropy_distill


def trainer_wrapper(
    teacher_file: str,
    num_epochs: int,
    dataloaders: dict,
    dataset_sizes: dict,
    lr: float,
    momentum: float,
    s_lr_step_size: int,
    s_rl_gamma: float,
) -> Tuple[dict, dict]:
    """
    Wrapper to distill a model
    Args:
        teacher_file (str): Teacher models path
        num_epochs (int): Number of epoch to run the train step
        dataloaders (dict): Train and val data loaders
        dataset_sizes (dict): Number elements in train and val folders
        lr (float): Learing rate. Defaults to 0.001.
        momentum (float): Momentum. Defaults to 0.9.
        s_lr_step_size (int): Learning rate scheduler step size. Defaults to 7.
        s_rl_gamma (float): Learning rate scheduler gamma. Defaults to 0.1.

    Returns:
        Tuple[, dict]: [description]
    """

    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Defines teacher
    if device.type == "cuda":
        teacher = torch.load(teacher_file)
    else:
        teacher = torch.load(teacher_file, map_location=torch.device("cpu"))

    # Defines student
    student = models.resnet18(pretrained=True)

    for param in student.parameters():
        param.requires_grad = False
    num_ftrs = student.fc.in_features

    # Sets linear outputs
    student.fc = nn.Linear(num_ftrs, 2)
    student = student.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(student.parameters(), lr=lr, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=s_lr_step_size, gamma=s_rl_gamma
    )

    model, results = train_model(
        student,
        teacher,
        cross_entropy_distill,
        optimizer_ft,
        exp_lr_scheduler,
        device,
        dataloaders,
        num_epochs,
        dataset_sizes,
        alpha=0.5,
    )

    return model, results
