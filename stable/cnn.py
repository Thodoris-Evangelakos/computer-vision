"""
This module contains a convolutional neural network model for image classification using PyTorch.

The module includes the following classes and functions:
- Net: A convolutional neural network model.
- main: Train and evaluate the neural network.
- load_data_for_model: Loads and prepares the CIFAR-10 dataset for training and testing.
- quick_test: Performs a quick test of the convolutional neural network model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

#ATTEMPTED CONSTANTS
#   DATASET
INPUT_CHANNELS = 3
#   CLS
KERNEL_SIZE = 3
PADDING = 1
#   POOLING
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    """
    A convolutional neural network model.

    Args:
        cl_num (int): The number of convolutional layers.
        fc_num (int): The number of fully connected layers.
        pl_num (int): The number of pooling layers.

    Attributes:
        conv_layers (nn.ModuleList): A list of convolutional layers.
        pool (nn.MaxPool2d): The pooling layer.
        fully_conn_layers (nn.ModuleList): A list of fully connected layers.

    Methods:
        _get_flattened_size: Calculates the size of the flattened features 
            after conv and pool layers.
        forward: Performs the forward pass of the network.
    """
    def __init__(self, cl_num: int, fc_num: int, pl_num: int):
        super(Net, self).__init__()

        in_channel_cond = lambda i: INPUT_CHANNELS if i == 0 else 32*(2**(i-1))

        self.conv_layers = nn.ModuleList()
        for i in range(cl_num):
            #could make it prettier by using out = calc(i) and in = calc(i-1)
            output_channels = 32*(2**i)
            self.conv_layers.append(nn.Conv2d(in_channels=in_channel_cond(i),
                                              out_channels=output_channels,
                                              kernel_size=KERNEL_SIZE, padding=PADDING))

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)
        # Calculate the size of the flattened features after conv and pool layers
        flattened_size = self._get_flattened_size(32, cl_num)  # CIFAR-10 images have 32x32 pixels

        in_features = lambda i: flattened_size if i == 0 else 512 // (2 ** (i-1))

        #self.to(device)

        self.fully_conn_layers = nn.ModuleList()
        for i in range(fc_num):
            out_features = 512 // (2 ** i)
            self.fully_conn_layers.append(nn.Linear(in_features(i), out_features))


    def _get_flattened_size(self, input_size, cl_num):
        size = input_size
        for _ in range(cl_num):
            size = size // POOL_STRIDE
        output_channels = 32 * (2 ** (cl_num-1)) # Last output_channels value
        flattened_size = size * size * output_channels
        return flattened_size

    def forward(self, x):
        """
        Performs forward pass through the convolutional neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        #applying conv layers with pooling
        for conv_layer in self.conv_layers:
            x = self.pool(F.relu(conv_layer(x)))

        # flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        for i, fc_layer in enumerate(self.fully_conn_layers):
            x = F.relu(fc_layer(x)) if i < len(self.fully_conn_layers) - 1 else fc_layer(x)
        return x

def main(cl_num, fc_num, pl_num, num_epochs, trainloader,
         testloader, optimizer_type='adam', learning_rate=0.001):
    """
    Train and evaluate the neural network.

    Args:
        cl_num (int): Number of convolutional layers.
        fc_num (int): Number of fully connected layers.
        pl_num (int): Number of pooling layers.
        num_epochs (int): Number of training epochs.
        trainloader: DataLoader for training data.
        testloader: DataLoader for test data.
        optimizer_type (str): Type of optimizer ('adam' or 'sgd').
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        float: Accuracy of the model on the test set.
    """

    # Initialize the network and move it to the chosen device (GPU or CPU)
    net = Net(cl_num, fc_num, pl_num).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer type. Choose 'adam' or 'sgd'.")

    # Training loop with tqdm progress bar
    for epoch in range(num_epochs):
        with tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for _, data in enumerate(pbar):
                inputs, labels = data
                # Move data to the device (GPU or CPU)
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Update progress bar with loss information
                pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # Move data to the device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def load_data_for_model(batch_size=4, num_workers=2):
    """
    Loads and prepares the CIFAR-10 dataset for training and testing.

    Args:
        batch_size (int): The number of samples per batch. Default is 4.
        num_workers (int): The number of worker threads for data loading. Default is 2.

    Returns:
        trainloader (torch.utils.data.DataLoader): DataLoader for training set.
        testloader (torch.utils.data.DataLoader): DataLoader for test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print("Data loaded successfully.")
    return trainloader, testloader

def quick_test():
    """
    Function to perform a quick test of the convolutional neural network model.

    This function loads the data for the model, defines test configurations,
    and runs the main function for each configuration.
    It prints the configuration and accuracy for each test.

    Parameters:
    None

    Returns:
    None
    """
    trainloader, testloader = load_data_for_model()

    # Test configurations
    test_configs = [
        {'cl_num': 2, 'fc_num': 2, 'pl_num': 2, 'num_epochs': 2},
        {'cl_num': 3, 'fc_num': 2, 'pl_num': 2, 'num_epochs': 2}
    ]

    for config in test_configs:
        accuracy = main(config['cl_num'], config['fc_num'], config['pl_num'],
                        config['num_epochs'], trainloader, testloader)
        print(f"Config: {config}, Accuracy: {accuracy}%")

if __name__ == "__main__":
    quick_test()
