import os
import argparse
import glob
import re
import tqdm
from pathlib import Path
import wandb



def parse_arguments(parser):

    # Set random seed
    parser.add_argument('--seed', type=int, default=None,
                        help="random seed (default: None)")
    parser.add_argument('--verbose', type=str, default="n",
                        choices=["y", "n"], help="verbose (default: n)")

    # Container environment
    parser.add_argument('--data_dir',  type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/dataset'))
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', './saved'))
    parser.add_argument('--log_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', './logs'))
    parser.add_argument('--name', type=str, default="exp",
                        help='name of the custom model and experiment')
    parser.add_argument('--load_model', type=str,
                        help="Load pretrained model if not None")

    # Load Dataset and construct DataLoader
    parser.add_argument('--dataset', type=str, default='MAPSDataset',
                        help="name of dataset (default: MAPSDataset)")
    # parser.add_argument('--additional', type=str, nargs='*',
    #                     help="list of additional dataset file names")
    parser.add_argument('--batch_size', metavar='B', type=int,
                        default=1, help="train set batch size (default: 1)")
    # parser.add_argument('--val_file', type=str, choices=["y", "n"],
    #                     default="n", help="whether to use valid.csv file (default: n)")
    # parser.add_argument('--val_ratio', type=float, default=0.2,
    #                     help="valid set ratio (default: 0.2)")
    parser.add_argument('--val_batch_size', metavar='B', type=int,
                        help="valid set batch size (default set to batch_size)")

    # Preprocessor and Data Augmentation
    # parser.add_argument('--preprocessor', type=str, default='BaselinePreprocessor',
    #                     help="type of preprocessor (default: BaselinePreprocessor)")
    # parser.add_argument('--augmentation', type=str,
    #                     help="type of augmentation (default: None)")

    # Load model and set optimizer
    parser.add_argument('--model', type=str, default='BaselineModel',
                        help="model name (default: BaselineModel)")
    parser.add_argument('--optim', type=str, default='AdamW',
                        help="optimizer name (default: AdamW)")
    parser.add_argument('--momentum', type=float, default=0.,
                        help="SGD with momentum (default: 0.0)")

    # training setup
    parser.add_argument('--epochs', type=int, metavar='N',
                        default=3, help="number of epochs (default 3)")
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="learning rate (default: 1e-5)")
    parser.add_argument('--log_every', type=int, metavar='N',
                        default=500, help="log every N steps (default: 500)")
    parser.add_argument('--eval_every', type=int, metavar='N',
                        default=500, help="evaluation interval for every N steps (default: 500)")
    parser.add_argument('--save_every', type=int, metavar='N',
                        default=500, help="save model interval for every N steps (default: 500)")
    parser.add_argument('--save_total_limit', type=int, metavar='N',
                        default=5, help="save total limit (choosing the best eval scores) (default: 5)")

    # Learning Rate Scheduler
    group_lr = parser.add_argument_group('lr_scheduler')
    group_lr.add_argument("--lr_type",  type=str, metavar='TYPE',
                          default="constant", help="lr scheduler type (default: constant)")
    group_lr.add_argument("--lr_weight_decay", type=float, metavar='LAMBDA',
                          default=0.01, help="weight decay rate for AdamW (default: 0.01)")
    group_lr.add_argument("--lr_gamma", type=float, metavar='GAMMA',
                          default=0.95, help="lr scheduler gamma (default: 0.95)")
    group_lr.add_argument("--lr_decay_step", type=int, metavar='STEP',
                          default=100, help="lr scheduler decay step (default: 100)")
    group_lr.add_argument("--lr_warmup_steps", type=int, metavar='N',
                          default=500, help="lr scheduler warmup steps (default: 500)")
    group_lr.add_argument("--lr_warmup_ratio", type=float, metavar='N',
                          default=0.1, help="lr scheduler warmup ratio (default: 0.1)")
    group_lr.add_argument("--lr_adamw_beta2", type=float, metavar='BETA2',
                          default=0.99, help="AdamW BETA2 (default: 0.99)")    

    args = parser.parse_args()

    return args



def increment_path(path, overwrite=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        overwrite (bool): whether to overwrite or increment path (increment if False).
    Returns:
        path: new path
    """
    path = Path(path)

    if (path.exists() and overwrite) or (not path.exists()):
        if not os.path.exists(str(path).split('/')[0]):
            os.mkdir(str(path).split('/')[0])
        if not path.exists():
            os.mkdir(path)
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = f"{path}{n}"
        if not os.path.exists(path):
            os.mkdir(path)
        return path




