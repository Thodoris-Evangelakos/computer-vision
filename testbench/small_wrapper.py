import torch
import torchvision
import torchvision.transforms as transforms
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F

import conv_neural_network as cnn

# Assuming the Net class is defined elsewhere in your code
# from your_model_file import Net

EPOCHS = [2, 4, 6, 8]
CLS = [2, 4, 6]  # Convolutional Layers
FCS = 2       # Fully Connected Layers
PLS = 2       # Pooling Layers

def load_data_for_model(batch_size=4, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Data loaded successfully.")
    
    return trainloader, testloader

'''
def train_and_evaluate_model(trainloader, testloader, cl_num, fc_num, epochs):
    net = Net(cl_num, fc_num, PLS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Training
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
'''

def model_eval():
    trainloader, testloader = load_data_for_model()

    for cl in CLS:
        for epoch in EPOCHS:
            accuracy = cnn.main(cl, FCS, PLS, epoch, trainloader, testloader)
            print(f'Epochs: {epoch}, Convolutional Layers: {cl}, Accuracy: {accuracy}%')

def gpu_test():
    print("GPU Test")
    print("GPU Available: ", torch.cuda.is_available())
    print("GPU Name: ", torch.cuda.get_device_name(0))
    print(20*"~")

def menu():
    print("What do you want to do?\n1   Evaluate model\n2   GPU Testt\n3   Exit")
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
    menu()

if __name__ == "__main__":
    menu()
