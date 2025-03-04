from trainer import Trainer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Trainer Script")

    parser.add_argument("-wp", "--wandb_project", type=str, required=False, help="WandB project name", default=None)
    parser.add_argument("-we", "--wandb_entity", type=str, required=False, help="WandB entity", default=None)

    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], required=True)

    parser.add_argument("-e", "--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], required=True)
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], required=True)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum (for momentum-based optimizers)")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta (for RMSprop)")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 (for Adam/Nadam)")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 (for Adam/Nadam)")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon for numerical stability")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")

    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier", "kaiming"], default="random", help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
