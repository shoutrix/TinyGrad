# from torch_trainer import Trainer
from trainer import Trainer
import argparse
import wandb
import os 
import sys

def trace_calls(frame, event, arg):
    if event == "call":
        return trace_lines
    return None

def trace_lines(frame, event, arg):
    if event == "line":
        code = frame.f_code
        filename = code.co_filename
        lineno = frame.f_lineno
        # print(filename.split("/")[-1])
        if filename.split("/")[-1] in ["tensor.py", "modules.py", "trainer.py", "train.py", "data_utils.py", "optimizers.py"]:
            print(f"Executed: {filename}:{lineno}")
    return trace_lines


def parse_args():
    parser = argparse.ArgumentParser(description="Trainer Script")

    parser.add_argument("-wp", "--wandb_project", type=str, required=False, help="WandB project name", default=None)
    parser.add_argument("-we", "--wandb_entity", type=str, required=False, help="WandB entity", default=None)

    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], required=False, default="fashion_mnist")

    parser.add_argument("-e", "--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], required=False, default="cross_entropy")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], required=False, default="rmsprop")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum (for momentum-based optimizers)")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta (for RMSprop)")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 (for Adam/Nadam)")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 (for Adam/Nadam)")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8, help="Epsilon for numerical stability")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.5, help="Weight decay (L2 regularization)")

    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "He", "Xavier", "kaiming", "Xavier_normal", "kaiming_normal"], default="kaiming", help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], required=False, default="sigmoid")
    parser.add_argument("-bn", "--batch_norm", type=bool, required=False, default=False)


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # sys.settrace(trace_calls)
    if args.wandb_entity is not None and args.wandb_project is not None:
        logging = True
        wandb.init(project=f"{args.wandb_project}_{args.dataset}", 
                            entity=args.wandb_entity, 
                            name=f"Batch.{args.batch_size}_Optim.{args.optimizer}_Init.{args.weight_init}_Layers.{args.num_layers}_Hidden.{args.hidden_size}_Activation.{args.activation}_BatchNorm.{args.batch_norm}",
                            config=vars(args))
    else:
        logging = False
    trainer = Trainer(args, logging=logging)
    trainer.train()
    trainer.infer()
    if logging:
        wandb.finish()
    # sys.settrace(None)

