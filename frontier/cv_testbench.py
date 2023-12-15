import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#risky
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

class Net(nn.Module):
    def__init__(self):
        super(Net, self).__init__()
        # Original layers
        