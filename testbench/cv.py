import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#risky
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

#ATTEMPTED CONSTANTS
#   DATASET
INPUT_CHANNELS = 3
#   CLS
KERNEL_SIZE = 3
PADDING = 1

class Net(nn.Module):
    def __init__(self, cl_num: int, fc_num: int, pl_num: int):
        super(Net, self).__init__()
        
        in_channel_cond = lambda i: INPUT_CHANNELS if i == 0 else 32*(2**(i-1))
        
        self.conv_layers = nn.ModuleList()
        for i in range(cl_num):
            output_channels = 32*(2**i)
            self.conv_layers.append(nn.Conv2d(in_channels=in_channel_cond(i), out_channels=output_channels, kernel_size=3, padding=1))
            
        
        self.conv_layers = [nn.Conv2d(in_channels=16*(2**i), out_channels=32*(2**i), kernel_size=KERNEL_SIZE, padding=PADDING) for i in range(cl_num)]
        self.fully_conn_layers = [nn.Linear(in_features=1024/(2**i), out_features=512/(2**i)) for i in range(fc_num)]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        def forward(self, x)
