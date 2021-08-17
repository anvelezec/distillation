import os
import torch
from torchvision import datasets, transforms

from distillation.train_step import train_model
from loss.cross_entropy_distill import cross_entropy_distill

import joblib
from trainer import trainer_wrapper

from utils.parser import Parser


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


def multiple_exp(
    alpha: int,
    n_exp: int,
    data_dir: str,
    teacher_file: str,
    num_epochs: int,
    batch_size: int,
    lr=0.001,
    momentum=0.9,
    s_lr_step_size=7,
    s_rl_gamma=0.1,
    **kwards
):
    """
    Performs multiple trains and validations steps in order to examinate the process variance.

    Args:
        alpha (int): Coheficient of ponderation similarity and cross entropi coheficients in the loss function
        n_exp (int): Number experiments to execute
        data_dir (str): Path with data to use during train and validation step
        teacher_file (str): Teacher models path
        num_epochs (int): Number of epoch to run the train step
        batch_size (int): Batch size to use for each iterateration
        lr (float, optional): Learing rate. Defaults to 0.001.
        momentum (float, optional): Momentum. Defaults to 0.9.
        s_lr_step_size (int, optional): Learning rate scheduler step size. Defaults to 7.
        s_rl_gamma (float, optional): Learning rate scheduler gamma. Defaults to 0.1.
    """
    
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    try:
        os.mkdir("results")
    except Exception as e:
        print("result folder already exist")

    try:
        os.mkdir("models")
    except Exception as e:
        print("models folder already exist")

    for i in range(1, n_exp):

        print("Run = {}".format(i))

        model, results = trainer_wrapper(
            teacher_file,
            num_epochs,
            dataloaders,
            dataset_sizes,
            lr,
            momentum,
            s_lr_step_size,
            s_rl_gamma,
        )
        
        with open("results/v" + str(i) + "_" + str(alpha) + "_" + str(batch_size) + ".joblib", "wb") as file:
            joblib.dump(results, file)

        torch.save(model, "models/v" + str(i) + "_" + str(alpha) + "_" + str(batch_size) + ".pt")


if __name__ == "__main__":

    parser = Parser()
    args = parser.parse_args()
    multiple_exp(**vars(args))

