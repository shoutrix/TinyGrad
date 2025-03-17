# TinyGrad: Custom Autograd and Neural Network Library

## Overview

This repository contains a **PyTorch-like** Tensor class implemented in **NumPy**, supporting **automatic differentiation** through a computation graph. The library enables the training of **simple neural networks** with a fully functional autograd system. The framework includes fundamental neural network modules, activation functions, and optimizers, enabling training on datasets like MNIST and Fashion-MNIST.

## Features
- **Custom Tensor Class**: Implements basic tensor operations with support for automatic differentiation.
- **Neural Network Modules**:
  - Linear Layers
  - Batch Normalization
  - Dropout
  - Activation Functions (ReLU, Sigmoid, Tanh, etc.)
  - Loss Functions (CrossEntropyLoss, MSELoss)
- **Optimizers**:
  - SGD
  - RMSprop
  - Adam
  - NAdam
  - AdamW
- **Weight Initialization**:
  - Random
  - Xavier_uniform
  - Xavier_normal
  - kaiming_uniform
  - kaiming_normal


## Model Performance
A neural network was trained using this custom autograd system and evaluated on Fashion-MNIST and MNIST datasets:
- **Fashion-MNIST**: Achieved **90.7% accuracy** on the validation set.
- **MNIST**: Achieved **98% accuracy** on the validation set.

## How to Run the Training Script
To train a model using the `train.py` script, you can pass various hyperparameters as command-line arguments. Below are some examples:

### Basic Training Command
```bash
python train.py --dataset fashion_mnist --epochs 20 --batch_size 128 --learning_rate 0.001 --optimizer adamw
```

### Using WandB for Logging
```bash
python train.py --dataset mnist --epochs 15 --batch_size 64 --learning_rate 0.0005 --optimizer rmsprop --wandb_project MyProject --wandb_entity MyEntity
```

### Custom Weight Initialization and Activation
```bash
python train.py --dataset fashion_mnist --epochs 25 --batch_size 128 --learning_rate 0.001 --optimizer adam --activation ReLU --weight_init kaiming_normal
```

## Command-Line Arguments
- `-d`, `--dataset` : Choose between `mnist` or `fashion_mnist`. Default: `fashion_mnist`
- `-e`, `--epochs` : Number of training epochs.
- `-b`, `--batch_size` : Size of training batches.
- `-l`, `--loss` : Choose between `mean_squared_error` or `cross_entropy`. Default: `cross_entropy`
- `-o`, `--optimizer` : Choose optimizer (`sgd`, `adam`, `rmsprop`, etc.).
- `-lr`, `--learning_rate` : Learning rate.
- `-w_d`, `--weight_decay` : weight_decay.
- `-w_i`, `--weight_init` : Weight initialization (`kaiming_normal`, `Xavier`, etc.).
- `-a`, `--activation` : Activation function (`ReLU`, `tanh`, etc.).
- `-bn`, `--batch_norm` : Enable/disable batch normalization.
- `-do`, `--dropout_p` : Dropout probability.
- `-nhl`, `--num_layers` : Number of layers.
- `-sz`, `--hidden_size` : Number of neurons in each hidden layer.
- `-wp`, `--wandb_project` : WandB project name (optional).
- `-we`, `--wandb_entity` : WandB entity name (optional).

