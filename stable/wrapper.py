# Importing the necessary libraries and the cnn module.
import torch
import torchvision
import torchvision.transforms as transforms
import cnn

# Defining the global variables.
EPOCHS = [2, 4, 6, 8]
CLS = [2, 4, 6]
FCS = 2
PLS = 2

def load_data_for_model(batch_size=4, num_workers=2):
    """
    Loads the CIFAR-10 dataset and creates data loaders for training and testing.

    Args:
        batch_size (int): The number of samples per batch. Default is 4.
        num_workers (int): The number of worker threads for data loading. Default is 2.

    Returns:
        trainloader (torch.utils.data.DataLoader): Data loader for training set.
        testloader (torch.utils.data.DataLoader): Data loader for testing set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print("Data loaded successfully.")

    return trainloader, testloader

def model_eval():
    """
    Evaluate the model's performance on different configurations.

    This function loads the data for the model and iterates over different configurations
    of convolutional layers and epochs. It calls the `cnn.main` function to train and test
    the model with each configuration and prints the accuracy.

    Args:
        None

    Returns:
        None
    """
    trainloader, testloader = load_data_for_model()

    for cl in CLS:
        for epoch in EPOCHS:
            accuracy = cnn.main(cl, FCS, PLS, epoch, trainloader, testloader)
            print(f'Epochs: {epoch}, Convolutional Layers: {cl}, Accuracy: {accuracy}%')

def gpu_test():
    """
    Function to test GPU availability and name.

    Prints the GPU availability and name using torch.cuda.is_available()
    and torch.cuda.get_device_name(0) respectively.
    """
    print("GPU Test")
    print("GPU Available: ", torch.cuda.is_available())
    print("GPU Name: ", torch.cuda.get_device_name(0))
    print(20*"~")

def menu():
    """
    Displays a menu and prompts the user for a choice.
    
    The function prints a menu with options to evaluate a model, perform a GPU test, or exit the program.
    It then prompts the user to enter their choice and executes the corresponding action based on the input.
    If an invalid choice is entered, it displays an error message and recursively calls itself to display the menu again.
    """
    print("========== MENU ==========")
    print("What do you want to do?")
    print("1. Evaluate model")
    print("2. GPU Test")
    print("3. Exit")
    
    choice = input("Enter your choice: ")
    
    match choice:
        case "1":
            model_eval()
        case "2":
            gpu_test()
        case "3":
            exit()
        case _:
            print("Invalid choice. Try again.")
            menu()

if __name__ == "__main__":
    menu()

