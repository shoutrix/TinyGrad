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

    parser.add_argument("-e", "--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], required=False, default="cross_entropy")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], required=False, default="adamw")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum (for momentum-based optimizers)")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta (for RMSprop)")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 (for Adam/Nadam)")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 (for Adam/Nadam)")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8, help="Epsilon for numerical stability")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")

    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "He", "Xavier", "kaiming", "Xavier_normal", "kaiming_normal"], default="kaiming_normal", help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=5, help="Number of hidden layers")
    parser.add_argument("-lds", "--layer_dims", type=list, default=[1024, 512, 256, 128, 64], help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=1024, help="Hidden layer size")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], required=False, default="ReLU")
    parser.add_argument("-bn", "--batch_norm", type=bool, required=False, default=True)
    parser.add_argument("-mgn", "--max_grad_norm", type=float, required=False, default=0.0)
    parser.add_argument("-do", "--dropout_p", type=float, required=False, default=0.4)    
    parser.add_argument("-lo", "--loss_fn", type=str, required=False, default="cross_entropy_loss")    


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # sys.settrace(trace_calls)
    print(args)
    if args.wandb_entity is not None and args.wandb_project is not None:
        logging = True
        wandb.init(project=f"{args.wandb_project}", 
                            entity=args.wandb_entity, 
                            name=f"VarHidden_LabelSmoothing_epoch{args.epochs}_batch{args.batch_size}_batchnorm{args.batch_norm}_dropout{args.dropout_p}_optim{args.optimizer}_lr{args.learning_rate}_init{args.weight_init}_layers{args.num_layers}_hidden{args.hidden_size}_activation{args.activation}_weightdecay{args.weight_decay}",
                            config=vars(args))
    else:
        logging = False
    trainer = Trainer(args, logging=logging)
    trainer.train()
    trainer.infer()
    if logging:
        wandb.finish()
    # sys.settrace(None)

