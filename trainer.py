import numpy as np
import matplotlib.pyplot as plt
from modules import Linear, CrossEntropyLoss, add_nonlinearity
from data_utils import load_data, FashionMnistDataloader
from optimizers import SGD, RMSprop, Adam, NAdam
from dataclasses import dataclass
from typing import List, Optional
import wandb
import os

os.environ["OMP_NUM_THREADS"] = "4"
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

        self.modules = {}
        for i, (dim1, dim2) in enumerate(zip(layer_dims, layer_dims[1:])):
            self.modules[f"Linear_{i}"] = Linear(dim1, dim2, config.weight_init)
            if self.non_linearity is not None and self.non_linearity != "identity":
                self.modules[f"{self.non_linearity}_{i}"] = add_nonlinearity(self.non_linearity)
        self.modules[f"Linear_out"] = Linear(dim2, self.n_classes, config.weight_init)

        self.loss_fn = CrossEntropyLoss()
 
    def parameters(self):
        params = {}
        for name, mod in self.modules.items():
            if isinstance(mod, Linear):
                params[f"{name}_weight"] = mod.weight
                params[f"{name}_bias"] = mod.bias
        return params
        
    
    def forward(self, x, y):
        # print(x.data.mean(), x.data.std(), x.data.max(), x.data.min(), np.linalg.norm(x.data))
        for name, mod in self.modules.items():
            x = mod(x)
        loss, acc = self.get_loss_and_accuracy(x, y)
        return loss, acc

    def get_loss_and_accuracy(self, x, y):
        loss_ = self.loss_fn(x, y)
        pred = np.argmax(x.data, axis=1)
        acc = (pred == y.data).sum() / pred.shape[0]
        return loss_, acc



class Trainer:
    def __init__(self, args, logging):
        self.args = args
        self.logging = logging
        if self.logging:
            print("logging to Wandb !!")
        self.trainset, self.validset, self.evalset, n_classes, flattened_dim = load_data(args.dataset)

        self.train_loader = FashionMnistDataloader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.valid_loader = FashionMnistDataloader(self.validset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = FashionMnistDataloader(self.evalset, batch_size=args.batch_size, shuffle=False)

        self.model_config = Config(
            n_layers=args.num_layers,
            non_linearity=args.activation,
            n_classes=n_classes,
            in_dim=flattened_dim,
            hidden_layer_size=args.hidden_size,
            weight_init=args.weight_init
        )
        self.model = NN(self.model_config)
        self.optimizer = self.get_optimizer()

    def get_optimizer(self):
        if self.args.optimizer in ["sgd", "momentum", "nag"]:
            return SGD(params=self.model.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum, dampening=0.0, nesterov=True, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "rmsprop":
            return RMSprop(params=self.model.parameters(), lr=self.args.learning_rate, alpha=self.args.beta, eps=self.args.epsilon, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        elif self.args.optimizer == "adam":
            return Adam(params=self.model.parameters(), lr=self.args.learning_rate, betas=(self.args.beta1, self.args.beta2), eps=self.args.epsilon, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "nadam":
            return NAdam(params=self.model.parameters(), lr=self.args.learning_rate, betas=(self.args.beta1, self.args.beta2), eps=self.args.epsilon, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f"Ugghh, unsupported optimizer: {self.args.optimizer}")

    def train(self):
        for epoch in range(self.args.epochs):
            print("Starting Epoch : ", epoch+1)
            total_loss, total_acc = 0, 0
            for i, (batch_data, batch_labels) in enumerate(self.train_loader):
                loss, acc = self.model.forward(batch_data, batch_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.logging:
                    for name, param in self.model.parameters().items():
                        wandb.log({f"{name}_grad_norm":np.linalg.norm(param.grad.data), f"{name}_norm":np.linalg.norm(param.data)})
                
                total_loss += loss.item()
                total_acc += acc
                
                if i%100==0:
                    print(f"Step {i}: Loss={loss.item():.4f}, Accuracy={acc:.4f}")
            avg_loss = total_loss / len(self.train_loader)
            avg_acc = total_acc / len(self.train_loader)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")

            if self.logging:
                wandb.log({"train_loss": avg_loss, "train_accuracy": avg_acc})

            valid_total_loss, valid_total_acc = 0, 0
            for batch_data, batch_labels in self.valid_loader:
                loss, acc = self.model.forward(batch_data, batch_labels)
                valid_total_loss += loss.item()
                valid_total_acc += acc

            valid_avg_loss = valid_total_loss / len(self.valid_loader)
            valid_avg_acc = valid_total_acc / len(self.valid_loader)
            print(f"Validation: Loss={valid_avg_loss:.4f}, Accuracy={valid_avg_acc:.4f}")
            if self.logging:
                wandb.log({"val_loss": avg_loss, "val_accuracy": avg_acc})
