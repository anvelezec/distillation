import torch

# Linear combination in the loss function
def cross_entropy_distill(s_outputs, t_outputs, labels, alpha):
    criterion_mse = torch.nn.MSELoss()
    criterion_ce = torch.nn.CrossEntropyLoss()

    ce = criterion_ce(s_outputs, labels)
    mse = criterion_mse(s_outputs, t_outputs)
    return alpha * ce + (1 - alpha) * mse