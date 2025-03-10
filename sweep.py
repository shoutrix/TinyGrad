from dataclasses import dataclass
from trainer import Trainer
import wandb

@dataclass
class TrainerConfig:
    dataset: str = "fashion_mnist"
    epochs: int = 10
    batch_size: int = 32
    loss: str = "cross_entropy"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    momentum: float = 0.5
    beta: float = 0.5
    beta1: float = 0.5
    beta2: float = 0.5
    epsilon: float = 1e-6
    weight_decay: float = 0.005
    weight_init: str = "random"
    num_layers: int = 3
    hidden_size: int = 256
    activation: str = "ReLU"
    max_grad_norm: float = 0.0

def main():
    wandb.init()
    wandb.run.name = f"epoch{wandb.config.epochs}_batch{wandb.config.batch_size}_optim{wandb.config.optimizer}_lr{wandb.config.learning_rate}_init{wandb.config.weight_init}_layers{wandb.config.num_layers}_hidden{wandb.config.hidden_size}_activation{wandb.config.activation}"

    config = TrainerConfig(
        epochs=wandb.config.epochs,
        batch_size=wandb.config.batch_size,
        loss="cross_entropy",
        optimizer=wandb.config.optimizer,
        learning_rate=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
        weight_init=wandb.config.weight_init,
        num_layers=wandb.config.num_layers,
        hidden_size=wandb.config.hidden_size,
        activation=wandb.config.activation,
    )

    trainer = Trainer(config, logging=True)
    trainer.train()

if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "name": "sweep_trial01",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "epochs": {"values": [10, 15]},
            "num_layers": {"values": [3, 4, 5]},
            "hidden_size": {"values": [64, 128, 256]},
            "weight_decay": {"values": [0, 0.0005, 0.005]},
            "learning_rate": {"values": [1e-2, 1e-3, 1e-4]},
            "optimizer": {"values": ["momentum", "nag", "rmsprop", "adam", "nadam"]},
            "batch_size": {"values": [32, 64]},
            "weight_init": {"values": ["random", "Xavier", "kaiming"]},
            "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Tinygrad_sweep_trial01", entity="shoutrik")
    wandb.agent(sweep_id, function=main, count=50)