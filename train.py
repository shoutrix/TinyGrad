# from torch_trainer import Trainer
from trainer import Trainer
import argparse
import wandb
import os 


def parse_args():
    parser = argparse.ArgumentParser(description="Trainer Script")

    parser.add_argument("-wp", "--wandb_project", type=str, required=False, help="WandB project name", default=None)
    parser.add_argument("-we", "--wandb_entity", type=str, required=False, help="WandB entity", default=None)

    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], required=False, default="fashion_mnist")

    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], required=False, default="cross_entropy")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], required=False, default="nadam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum (for momentum-based optimizers)")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta (for RMSprop)")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 (for Adam/Nadam)")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 (for Adam/Nadam)")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8, help="Epsilon for numerical stability")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=5e-3, help="Weight decay (L2 regularization)")

    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier", "kaiming", "Xavier_normal", "kaiming_normal"], default="Xavier_normal", help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], required=False, default="tanh")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.wandb_entity is not None and args.wandb_project is not None:
        logging = True
        wandb.init(project=f"{args.wandb_project}_{args.dataset}", 
                            entity=args.wandb_entity, 
                            name=f"batch{args.batch_size}_optim{args.optimizer}_init{args.weight_init}_layers{args.num_layers}_hidden{args.hidden_size}_activation{args.activation}",
                            config=vars(args))
    else:
        logging = False
    trainer = Trainer(args, logging=logging)
    trainer.train()
    if logging:
        wandb.finish()

