import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from tensor import Tensor
from modules import Linear, Tanh, ReLU, Sigmoid, CrossEntropyLoss, add_nonlinearity
from data_utils import load_data, FashionMnistDataloader
from optimizers import SGD, MomentumSGD
from dataclasses import dataclass
from typing import List, Optional
import wandb


@dataclass
class Config:
    n_layers: int
    non_linearity: Optional[str]
    n_classes: int
    in_dim: int
    hidden_layer_size: Optional[int] = None
    layer_dims: Optional[List[int]] = None
    weight_init: Optional[str] = "kaiming"

    def __post_init__(self):
        if self.layer_dims is None and self.hidden_layer_size is None:
            raise ValueError("At least one of layer_dims or hidden_layer_size should be provided")
        
        if self.layer_dims is not None and len(self.layer_dims) != self.n_layers:
            raise ValueError("layer_dims length must be equal to n_layers")

        if self.layer_dims is None:
            self.layer_dims = [self.hidden_layer_size] * self.n_layers



class NN:
    def __init__(self, config):
        self.n_layers = config.n_layers
        self.non_linearity = config.non_linearity
        self.n_classes = config.n_classes
        self.in_dim = config.in_dim

        layer_dims = config.layer_dims if config.layer_dims is not None else [config.hidden_layer_size] * self.n_layers
        layer_dims = [self.in_dim] + layer_dims

        self.modules = []
        for dim1, dim2 in zip(layer_dims, layer_dims[1:]):
            self.modules.append(Linear(dim1, dim2, config.weight_init))
            if self.non_linearity is not None:
                self.modules.append(add_nonlinearity(self.non_linearity))
        self.modules.append(Linear(dim2, self.n_classes, config.weight_init))

        self.loss_fn = CrossEntropyLoss()
 
    def parameters(self):
        params = []
        for mod in self.modules:
            if isinstance(mod, Linear):
                params.extend([mod.weight, mod.bias])
        return params
        
    
    def forward(self, x, y):
        # print(x.shape)
        for mod in self.modules:
            # print(mod)
            x = mod(x)
            # print(x.shape)

        loss, acc = self.get_loss_and_accuracy(x, y)
        return loss, acc

    def get_loss_and_accuracy(self, x, y):
        loss_ = self.loss_fn(x, y)
        pred = np.argmax(x.data, axis=1)
        acc = (pred == y.data).sum() / pred.shape[0]
        return loss_, acc



class Trainer:
    def __init__(self, args):
        self.args = args

        # wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

        self.trainset, self.validset, self.evalset = load_data(args.dataset)

        self.train_loader = FashionMnistDataloader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.valid_loader = FashionMnistDataloader(self.validset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = FashionMnistDataloader(self.evalset, batch_size=args.batch_size, shuffle=False)

        self.model_config = Config(
            n_layers=args.num_layers,
            non_linearity=args.activation,
            n_classes=10,
            in_dim=784,
            hidden_layer_size=args.hidden_size,
            weight_init=args.weight_init
        )
        self.model = NN(self.model_config)
        self.loss_fn = CrossEntropyLoss()

        self.optimizer = self.get_optimizer()


    def get_optimizer(self):
        if self.args.optimizer == "sgd":
            return SGD(params=self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if self.args.optimizer == "momentum":
            return MomentumSGD(params=self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        else:
            raise ValueError(f"Ugghh, unsupported optimizer: {self.args.optimizer}")

    def train(self):
        for epoch in range(self.args.epochs):
            print("Starting Epoch : ", epoch+1)
            total_loss, total_acc = 0, 0
            for i, (batch_data, batch_labels) in enumerate(self.train_loader):
                loss, acc = self.model.forward(batch_data, batch_labels)
                # print(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_acc += acc
                # print(i)
                
                if i%100==0:
                    print(f"Step {i}: Loss={loss.item():.4f}, Accuracy={acc:.4f}")

            avg_loss = total_loss / len(self.train_loader)
            avg_acc = total_acc / len(self.train_loader)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")

            # wandb.log({"train_loss": avg_loss, "train_accuracy": avg_acc})


            valid_total_loss, valid_total_acc = 0, 0
            for batch_data, batch_labels in self.valid_loader:
                loss, acc = self.model.forward(batch_data, batch_labels)
                valid_total_loss += loss.item()
                valid_total_acc += acc

            valid_avg_loss = valid_total_loss / len(self.valid_loader)
            valid_avg_acc = valid_total_acc / len(self.valid_loader)
            print(f"Validation: Loss={valid_avg_loss:.4f}, Accuracy={valid_avg_acc:.4f}")
            # wandb.log({"val_loss": avg_loss, "val_accuracy": avg_acc})
