import numpy as np
import matplotlib.pyplot as plt
from modules import Linear, CrossEntropyLoss, add_nonlinearity, BatchNorm, Dropout, MSELoss
from data_utils import load_data, FashionMnistDataloader
from optimizers import SGD, RMSprop, Adam, NAdam, AdamW
from dataclasses import dataclass
from typing import List, Optional
import wandb
import os

@dataclass
class Config:
    n_layers: int
    non_linearity: Optional[str]
    n_classes: int
    in_dim: int
    hidden_layer_size: Optional[int] = None
    hidden_layer_size_list: Optional[list] = None
    layer_dims: Optional[List[int]] = None
    weight_init: Optional[str] = "kaiming"
    batch_norm: bool = True
    dropout_p: float = 0.2
    loss_fn:str = "mse"

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
        self.dropout_p = config.dropout_p
        self.apply_dropout = True if config.dropout_p>0.0 else False

        layer_dims = config.layer_dims if config.layer_dims is not None else [config.hidden_layer_size] * self.n_layers
        layer_dims = [self.in_dim] + layer_dims

        print("layer dims : ", layer_dims)

        self.modules = {}
        for i, (dim1, dim2) in enumerate(zip(layer_dims, layer_dims[1:])):
            self.modules[f"Linear_{i}"] = Linear(dim1, dim2, config.weight_init, config.non_linearity)
            if config.batch_norm:
                self.modules[f"BatchNorm_{i}"] = BatchNorm(dim2)
            if self.non_linearity is not None and self.non_linearity != "identity":
                self.modules[f"{self.non_linearity}_{i}"] = add_nonlinearity(self.non_linearity)
            if self.dropout_p > 0.0:
                self.modules[f"Dropout_{i}"] = Dropout(self.dropout_p)
        self.modules[f"Linear_out"] = Linear(dim2, self.n_classes, config.weight_init, config.non_linearity)

        if config.loss_fn  == "cross_entropy":
            self.loss_fn = CrossEntropyLoss(label_smoothing=0.1)
        elif config.loss_fn == "mean_squared_error":
            self.loss_fn = MSELoss()
        
        
    def eval(self):
        self.apply_dropout = False
 
    def parameters(self):
        params = {}
        for name, mod in self.modules.items():
            if isinstance(mod, Linear):
                params[f"{name}_weight"] = mod.weight
                params[f"{name}_bias"] = mod.bias
            elif isinstance(mod, BatchNorm):
                params[f"{name}_gamma"] = mod.gamma
                params[f"{name}_beta"] = mod.beta
        return params
        
    
    def forward(self, x, y, training):
        # print(x.data.mean(), x.data.std(), x.data.max(), x.data.min(), np.linalg.norm(x.data))
        for name, mod in self.modules.items():
            x = mod(x, training)
            # print(x.name)

        loss, acc = self.get_loss_and_accuracy(x, y)
        return loss, acc, x

    def get_loss_and_accuracy(self, x, y):
        loss_ = self.loss_fn(x, y)
        pred = np.argmax(x.data, axis=1)
        acc = (pred == y.data).sum() / pred.shape[0]
        return loss_, acc



class Trainer:
    def __init__(self, args, logging):
        
        os.environ["OMP_NUM_THREADS"] = "4"
        
        self.args = args
        self.logging = logging
        if self.logging:
            print("logging to Wandb !!")
        self.trainset, self.validset, self.evalset, n_classes, flattened_dim , self.labels = load_data(args.dataset)

        self.train_loader = FashionMnistDataloader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.valid_loader = FashionMnistDataloader(self.validset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = FashionMnistDataloader(self.evalset, batch_size=args.batch_size, shuffle=False)

        self.model_config = Config(
            n_layers=args.num_layers,
            non_linearity=args.activation,
            n_classes=n_classes,
            in_dim=flattened_dim,
            hidden_layer_size=args.hidden_size,
            layer_dims=args.layer_dims,
            weight_init=args.weight_init,
            batch_norm=args.batch_norm,
            dropout_p=args.dropout_p,
            loss_fn=args.loss_fn
        )
        self.model = NN(self.model_config)
        self.optimizer = self.get_optimizer()
        self.max_grad_norm = args.max_grad_norm

    def get_optimizer(self):
        if self.args.optimizer in "sgd":
            return SGD(params=self.model.parameters(), lr=self.args.learning_rate, momentum=0.0, dampening=0.0, nesterov=False, weight_decay=self.args.weight_decay)
        elif self.args.optimizer in "momentum":
            return SGD(params=self.model.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum, dampening=0.0, nesterov=False, weight_decay=self.args.weight_decay)
        elif self.args.optimizer in "nag":
            return SGD(params=self.model.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum, dampening=0.0, nesterov=True, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "rmsprop":
            return RMSprop(params=self.model.parameters(), lr=self.args.learning_rate, alpha=self.args.beta, eps=self.args.epsilon, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        elif self.args.optimizer == "adam":
            return Adam(params=self.model.parameters(), lr=self.args.learning_rate, betas=(self.args.beta1, self.args.beta2), eps=self.args.epsilon, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "nadam":
            return NAdam(params=self.model.parameters(), lr=self.args.learning_rate, betas=(self.args.beta1, self.args.beta2), eps=self.args.epsilon, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "adamw":
            return AdamW(params=self.model.parameters(), lr=self.args.learning_rate, betas=(self.args.beta1, self.args.beta2), eps=self.args.epsilon, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f"Ugghh, unsupported optimizer: {self.args.optimizer}")


    def apply_grad_norm(self):
        # print("Applying Grad Norm ...")
        total_norm = 0.0
        if self.max_grad_norm is not None:
            for param in self.model.parameters().values():
                if param.grad is not None and param.requires_grad:
                    total_norm += np.sum(param.grad ** 2)

            total_norm = np.sqrt(total_norm)
            clip_coef = self.max_grad_norm / (total_norm + 1e-6) if total_norm > self.max_grad_norm else 1.0
        
        for param in self.model.parameters().values():
            param.grad *= clip_coef


    def train(self):
        for epoch in range(self.args.epochs):
            print("Starting Epoch : ", epoch+1)
            total_loss, total_acc = 0, 0
            for i, (batch_data, batch_labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                loss, acc, _ = self.model.forward(batch_data, batch_labels, training=True)
                loss.backward()
                
                if self.max_grad_norm!=0.0:
                    self.apply_grad_norm()
                    
                self.optimizer.step()
                
                if self.logging:
                    for name, param in self.model.parameters().items():
                        wandb.log({f"{name}_grad_norm":np.linalg.norm(param.grad), f"{name}_norm":np.linalg.norm(param.data)})
                # for name, param in self.model.parameters().items():
                #     print(f"{name}_weight_norm : ", np.linalg.norm(param.data))
                
                total_loss += loss.item()
                total_acc += acc
                
                if i%100==0:
                    print(f"Step {i}: Loss={loss.item():.4f}, Accuracy={acc:.4f}")
            avg_loss = total_loss / len(self.train_loader)
            avg_acc = total_acc / len(self.train_loader)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")

            if self.logging:
                print("Logging Train stats ...")
                wandb.log({"train_loss": avg_loss, "train_accuracy": avg_acc})

            valid_total_loss, valid_total_acc = 0, 0
            for batch_data, batch_labels in self.valid_loader:
                loss, acc, _ = self.model.forward(batch_data, batch_labels, training=False)
                valid_total_loss += loss.item()
                valid_total_acc += acc

            valid_avg_loss = valid_total_loss / len(self.valid_loader)
            valid_avg_acc = valid_total_acc / len(self.valid_loader)
            print(f"Validation: Loss={valid_avg_loss:.4f}, Accuracy={valid_avg_acc:.4f}")
            if self.logging:
                print("Logging Valid stats ...")
                wandb.log({"val_loss": valid_avg_loss, "val_accuracy": valid_avg_acc})
                
                
    def infer(self):
        test_total_loss, test_total_acc = 0, 0
        preds = []
        target = []
        for batch_data, batch_labels in self.test_loader:
            loss, acc, logits = self.model.forward(batch_data, batch_labels, training=False)
            predicted = np.argmax(logits.data, axis=1)
            preds.append(predicted)
            target.append(batch_labels.data)
            # print(predicted.shape, batch_labels.data.shape)
            
            test_total_loss += loss.item()
            test_total_acc += acc

        test_avg_loss = test_total_loss / len(self.test_loader)
        test_avg_acc = test_total_acc / len(self.test_loader)
        print(f"\n\nInference: Loss={test_avg_loss:.4f}, Accuracy={test_avg_acc:.4f}")
        if self.logging:
            print("Logging Inference stats ...")
            wandb.log({"test_loss": test_avg_loss, "test_accuracy": test_avg_acc})
            
            preds = np.concatenate(preds)
            target = np.concatenate(target)
            
            # print(preds.shape, target.shape)
            
            
            wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(probs=None, y_true=target, preds=preds, class_names=self.labels)})


