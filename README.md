# Convolutional Neural Network Project

This project contains a Convolutional Neural Network (CNN) model for image classification using PyTorch. The model is designed to be flexible, allowing for a variable number of convolutional layers, fully connected layers, and pooling layers.

## Project Structure

The project is organized into two main directories:

- `stable/`: Contains the main implementation of the CNN model and a wrapper for testing different configurations.
- `testbench/`: Contains test scripts and a smaller version of the wrapper.

## Key Files

- `stable/cnn.py`: Contains the CNN model implementation, a function for training and evaluating the model, and functions for loading the CIFAR-10 dataset.
- `stable/wrapper.py`: Contains a function for evaluating the model with different configurations, a function for testing GPU availability, and a menu for user interaction.
- `testbench/conv_neural_network.py`: Contains a test version of the CNN model.
- `testbench/small_wrapper.py`: Contains a smaller version of the wrapper for testing purposes.

## How to Run

1. Ensure you have the necessary dependencies installed (PyTorch, torchvision).
2. Run the `wrapper.py` script in the `stable/` directory to interact with the menu and choose an action.
3. Choose "1" to evaluate the model with different configurations, or "2" to perform a GPU test.

## Dataset

The model is trained and tested on the CIFAR-10 dataset, which is automatically downloaded when running the scripts.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the terms of the MIT license.