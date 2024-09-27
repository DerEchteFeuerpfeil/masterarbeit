import os
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import yaml
from dataset import GenericRasterizedSVGDataModule, GenericRasterDatamodule, MNISTDataset, MNISTppDataset, CenterShapeLayersFromSVGDataModule, VQDataModule, VSQDatamodule
from models import VectorVAEnLayers, VSQ, VQ_SVG_Stage2
from experiment import VAEXperiment, VectorVQVAE_Experiment_Stage1, SVG_VQVAE_Stage2_Experiment
from utils import get_rank
import wandb
import torch
from pytorch_lightning.profilers import SimpleProfiler
import pydiffvg

print(f"[INFO] diffvg running on GPU: {pydiffvg.get_use_gpu()}")
torch.set_float32_matmul_precision('high')

DATASETMAP = {
    "mnistpp": MNISTppDataset,
    "mnist": MNISTDataset,
    "centeredShapeLayers": CenterShapeLayersFromSVGDataModule,
    "tokens": VQDataModule,
    "figr8": VSQDatamodule,
    "raster_figr8": GenericRasterDatamodule,
    "raster_fonts": GenericRasterizedSVGDataModule,
}

MODELS = {
    "Im2Vec": VectorVAEnLayers,
    "VSQ": VSQ,
}

print("CUDA enabled: ",torch.cuda.is_available())

parser = argparse.ArgumentParser(description='Generic runner for VAE models')

# Model parameters
parser.add_argument('--model_name', type=str, default='VSQ')
parser.add_argument('--vector_decoder_model', type=str, default='mlp')
parser.add_argument('--quantized_dim', type=int, default=512)
parser.add_argument('--codebook_size', type=int, default=4096)
parser.add_argument('--image_loss', type=str, default='mse')
parser.add_argument('--vq_method', type=str, default='fsq')
parser.add_argument('--fsq_levels', nargs='+', type=int, default=[7, 5, 5, 5, 5])
parser.add_argument('--num_segments', type=int, default=1)
parser.add_argument('--num_codes_per_shape', type=int, default=1)
parser.add_argument('--pred_color', action='store_true')
parser.add_argument("--alpha", type=float, default=0.0)

# Data parameters
parser.add_argument('--dataset', type=str, default='figr8')
parser.add_argument('--csv_path', type=str, default='/scratch2/moritz_data/figr8/vsq_sweep.csv')
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--train_batch_size', type=int, default=2)
parser.add_argument('--val_batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--individual_max_length', type=float, default=5.0)
# parser.add_argument('--stroke_width', type=float, default=0.4)
parser.add_argument('--use_single_paths', type=bool, default=False)
parser.add_argument('--max_shapes_per_svg', type=int, default=128)
parser.add_argument('--color_mode', type=str, default='None')  # for colorful
parser.add_argument('--use_random_stroke_widths', action='store_true')


# assert ...

# Experiment parameters
parser.add_argument('--lr', type=float, default=2.0e-05)
parser.add_argument('--weight_decay', type=float, default=1.e-4)
parser.add_argument('--scheduler_gamma', type=float, default=0.98)
parser.add_argument('--train_log_interval', type=float, default=0.1)
parser.add_argument('--manual_seed', type=int, default=42)
parser.add_argument('--continue_checkpoint', type=str, default=None)

# Trainer parameters
parser.add_argument('--devices', type=int, default=-1)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--accumulate_grad_batches', type=int, default=1)
parser.add_argument('--val_check_interval', type=float, default=0.5)
parser.add_argument('--limit_val_batches', type=float, default=1.0)

# Logging parameters
parser.add_argument('--entity', type=str, default='mfeuer')
parser.add_argument('--project', type=str, default='thesis')
parser.add_argument('--save_base_dir', type=str, default='/scratch2/moritz_logs/thesis/untitled')
# parser.add_argument('--run_name', type=str, default='VSQ_nseg=1_ncode=1_lseg=5')
parser.add_argument('--prefix', type=str, default=None)
parser.add_argument('--version', type=int, default=0)
parser.add_argument('--author', type=str, default='Moritz')
parser.add_argument('--id', type=str, default=None)

parser.add_argument('--wandb', action='store_true', help="want to log the run with wandb? (default false)")
parser.add_argument('--debug', action='store_true', help='disable wandb logs, set workers to 0. (default false)')

args = parser.parse_args()

# Assertions and initial setups
assert args.dataset in DATASETMAP.keys(), f"dataset {args.dataset} not supported, try one of {list(DATASETMAP.keys())}"
assert args.model_name in MODELS.keys(), f"model {args.model_name} not supported, try one of {list(MODELS.keys())}"
assert args.num_segments > 1 or args.alpha == 0.0, "you cannot set alpha > 0 for num_segments = 1, does not make sense"

# do some processing
stroke_width = (args.individual_max_length + 2) * 0.4 / 5.0
run_name = f'{args.prefix+"_" if args.prefix is not None else ""}VSQ_nseg={args.num_segments}_ncode={args.num_codes_per_shape}_lseg={args.individual_max_length}_alpha={args.alpha}{args.color_mode if args.color_mode is not None else ""}{"_r_stroke_w" if args.use_random_stroke_widths else ""}_L{"".join(map(str, args.fsq_levels))}'
save_dir = args.save_base_dir

if args.debug:
    args.num_workers = 0

current_process_rank = get_rank()


# For reproducibility
seed_everything(args.manual_seed, True)
print("Loading model...")
model_params = {
    "vector_decoder_model": args.vector_decoder_model,
    "quantized_dim": args.quantized_dim,
    "codebook_size": args.codebook_size,
    "image_loss": args.image_loss,
    "vq_method": args.vq_method,
    "fsq_levels": args.fsq_levels,
    "num_segments": args.num_segments,
    "num_codes_per_shape": args.num_codes_per_shape,
    "pred_color": args.pred_color,
    "alpha": args.alpha,
}

if args.wandb:
    model = MODELS[args.model_name](patch_size=args.width, **model_params, wandb_logging=True)
else:
    model = MODELS[args.model_name](patch_size=args.width, **model_params)
print("Loading dataset...")

data_params = {
    "dataset": args.dataset,
    "csv_path": args.csv_path,
    "channels": args.channels,
    "width": args.width,
    "train_batch_size": args.train_batch_size,
    "val_batch_size": args.val_batch_size,
    "num_workers": args.num_workers,
    "individual_max_length": args.individual_max_length,
    "stroke_width": stroke_width,
    "use_single_paths": args.use_single_paths,
    "max_shapes_per_svg": args.max_shapes_per_svg,
    "use_random_stroke_widths" : args.use_random_stroke_widths,
    "color_mode" : args.color_mode,
}

exp_params = {
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "scheduler_gamma": args.scheduler_gamma,
    "train_log_interval": args.train_log_interval,
    "manual_seed": args.manual_seed,
    "max_epochs": args.max_epochs,
}

if args.continue_checkpoint is not None:
    assert os.path.exists(args.continue_checkpoint), f"checkpoint {args.continue_checkpoint} does not exist"
    print(f"Found checkpoint to continue training from: {args.continue_checkpoint}")
    if args.id is None:
        print(f"wandb id must be set in logging_params to continue the logging in wandb")
        input("Press Enter to continue without continuing in wandb or CTRL+C to cancel")
    exp_params["continue_checkpoint"] = args.continue_checkpoint
else:
    assert args.id is None, f"wandb id must not be set if not continuing from a checkpoint"

if args.continue_checkpoint is not None:
    exp_params["continue_checkpoint"] = args.continue_checkpoint

if args.model_name == "VSQ":
    data = DATASETMAP[args.dataset](**data_params)
    data.setup()
    experiment = VectorVQVAE_Experiment_Stage1(model, **exp_params, wandb=args.wandb, datamodule=data)
else:
    raise ValueError("Unknown model provided: ", args.model_name)


if args.wandb:
    wandb_logger = WandbLogger(
        name=run_name,
        save_dir=save_dir,
        tags=[args.author],
        project=args.project,
        log_model=True,
        entity=args.entity,
        mode="disabled" if args.debug else "online",
        resume="must" if args.continue_checkpoint is not None else "allow",
        id=args.id
    )
else:
    wandb_logger = TensorBoardLogger(
        save_dir=save_dir,
        name=run_name
    )

if current_process_rank == 0:
    config = {
        "model_params": model_params,
        "data_params": data_params,
        "exp_params": exp_params,
        "logging_params": {
            "name": run_name,
            "save_dir": save_dir,
            "author": args.author,
            "project": args.project,
            "entity": args.entity,
            "id": args.id
            }
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), 'w+') as file:
        yaml.dump(config, file)
    if args.wandb:
        allow_val_change = True if config["logging_params"].get("allow_val_change") or "continue_checkpoint" in config["exp_params"] else False
        wandb_logger.experiment.config.update(config, allow_val_change=allow_val_change)

# profiler = SimpleProfiler(dirpath=os.path.join(save_dir))
runner = Trainer(
    logger=wandb_logger,
    callbacks=[
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        # EarlyStopping("val_loss", 0.0002, 6, verbose=True),
        ModelCheckpoint(save_top_k=2,
                        dirpath=os.path.join(save_dir, "checkpoints"),
                        monitor="val_loss",
                        save_last=True,
                        filename="{epoch}-{val_loss:.4f}"),
    ],
    # profiler=profiler,
    devices=args.devices,
    max_epochs=args.max_epochs,
    accumulate_grad_batches=args.accumulate_grad_batches,
    val_check_interval=args.val_check_interval,
    limit_val_batches=args.limit_val_batches,
)

print(f"======= Training {args.model_name} =======")
try:
    if args.continue_checkpoint is not None:
        runner.fit(experiment, datamodule=data, ckpt_path=args.continue_checkpoint)
    else:
        runner.fit(experiment, datamodule=data)
    # profiler.describe()
    # with open("profiler_results.txt", "w") as f:
    #     f.write(profiler.summary())
except KeyboardInterrupt:
    print("Training interrupted by user.")
    # profiler.describe()
    # with open("profiler_results.txt", "w") as f:
    #     f.write(profiler.summary())
