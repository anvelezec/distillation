import torch

# Linear combination in the loss function
def cross_entropy_distill(s_outputs: torch.Tensor, t_outputs: torch.Tensor, labels:torch.Tensor, alpha:float) -> torch.Tensor:
    """
    Blended loss function. Mix between MSE and cross entropy
    Args:
        s_outputs (torch.Tensor): Student output loggits
        t_outputs (torch.Tensor): Teacher output loggits 
        labels (torch.Tensor): Ground thruth labels
        alpha (float): MSE and cross entropy mix coheficient factor 

    Returns:
        torch.Tensor: Mixed loss
    """
    criterion_mse = torch.nn.MSELoss()
    criterion_ce = torch.nn.CrossEntropyLoss()

    ce = criterion_ce(s_outputs, labels)
    mse = criterion_mse(s_outputs, t_outputs)
    return alpha * ce + (1 - alpha) * mse
