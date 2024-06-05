from torch.nn import functional as F

def get_loss(name):
    if name == "crossentropy":
        return F.cross_entropy
    
    raise ValueError(f'{name}: invalid loss name')

