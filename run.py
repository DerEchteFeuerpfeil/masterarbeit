import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder, EarlyStopping
from dataset import GenericRasterizedSVGDataModule,GenericRasterDatamodule, MNISTDataset, MNISTppDataset, CenterShapeLayersFromSVGDataModule, VQDataModule, VSQDatamodule
from models import VectorVAEnLayers, VSQ, VQ_SVG_Stage2
from experiment import VAEXperiment, VectorVQVAE_Experiment_Stage1, SVG_VQVAE_Stage2_Experiment
import wandb
from utils import get_rank
import torch
from pytorch_lightning.profilers import SimpleProfiler
import pydiffvg
print(f"[INFO] diffvg running on GPU: {pydiffvg.get_use_gpu()}")
torch.set_float32_matmul_precision('high')

DATASETMAP = {
    "mnistpp": MNISTppDataset,
    "mnist": MNISTDataset,
    "centeredShapeLayers" : CenterShapeLayersFromSVGDataModule,
    "tokens" : VQDataModule,
    "figr8" : VSQDatamodule,
    "raster_figr8" : GenericRasterDatamodule,
    "raster_fonts" : GenericRasterizedSVGDataModule,
}

MODELS = {
    "Im2Vec": VectorVAEnLayers,
    "VSQ" : VSQ,
  }


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c', dest="filename", metavar='FILE', help='path to the config file', default='configs/vae.yaml')
parser.add_argument("--wandb", "-w", dest="wandb", action='store_true', help="want to log the run with wandb? (default false)")
parser.add_argument('--debug', action='store_true', help='disable wandb logs, set workers to 0. (default false)')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# assertions for the config file
if "context_length" in config["model_params"]:
    assert config["model_params"]["context_length"] == config["data_params"]["context_length"], f"context length in model and data params must be the same"
assert config["data_params"]["dataset"] in DATASETMAP.keys(), f"dataset {config['data_params']['dataset']} not supported, try one of {list(DATASETMAP.keys())}"
assert config["model_params"]["name"] in MODELS.keys(), f"model {config['model_params']['name']} not supported, try one of {list(MODELS.keys())}"

# fix the ratio of viewbox size to stroke width to stay consistent
if config["model_params"]["name"] == "VSQ":
    # 0.4 stroke width on a lseg=3 (viewbox 5)
    config["data_params"]["stroke_width"] = (config["data_params"]["individual_max_length"] + 2) * 0.4 / 5.0

# disabling multi-threading when debugging
if args.debug:
    config["data_params"]["num_workers"] = 0

current_process_rank = get_rank()

if "continue_checkpoint" in config["exp_params"] and config["exp_params"]["continue_checkpoint"] is not None:
    assert os.path.exists(config["exp_params"]["continue_checkpoint"]), f"checkpoint {config['exp_params']['continue_checkpoint']} does not exist"
    print(f"Found checkpoint to continue training from: {config['exp_params']['continue_checkpoint']}")
    if "id" not in config["logging_params"]:
        print(f"wandb id must be set in logging_params to continue the logging in wandb")
        input("Press Enter to continue without continuing in wandb or CTRL+C to cancel")
else:
    assert "id" not in config["logging_params"], f"wandb id must not be set if not continuing from a checkpoint"

if args.wandb:
    if "entity" not in config['logging_params']:
        entity = "mfeuer"
    else:
        entity = config['logging_params']['entity']
    wandb_logger = WandbLogger(
        name=config['logging_params']['name'],
        save_dir=config['logging_params']['save_dir'],
        tags=[config['logging_params']['author']],
        project=config["logging_params"]["project"],
        log_model=True,
        entity=entity,
        mode="disabled" if args.debug else "online",
        resume="must" if "continue_checkpoint" in config["exp_params"] else "allow",
        id=config["logging_params"].get("id")
    )
    if current_process_rank == 0:
        allow_val_change = True if config["logging_params"].get("allow_val_change") or "continue_checkpoint" in config["exp_params"] else False
        wandb_logger.experiment.config.update(config, allow_val_change=allow_val_change)
else:
    wandb_logger = TensorBoardLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name']
    )

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)
print("Loading model...")
if args.wandb:
    model = MODELS[config['model_params']['name']](patch_size = config['data_params']["width"],**config['model_params'], wandb_logging=True)
    # wandb_logger.watch(model, log="gradients", log_freq=500, log_graph=False)
    # wandb.watch(model, log='all', log_freq=100)  # can be "all"
else:
    model = MODELS[config['model_params']['name']](patch_size = config['data_params']["width"], **config['model_params'])
print("Loading dataset...")

if config['model_params']['name'] == "VSQ":
    # NOTE stroke width has been set at top of file
    data = DATASETMAP[config["data_params"]["dataset"]](**config["data_params"])
    data.setup()
    experiment = VectorVQVAE_Experiment_Stage1(model,
                                               **config['exp_params'],
                                               wandb = args.wandb,
                                               datamodule = data,
                                               max_epochs=config["trainer_params"]["max_epochs"])
elif config['model_params']['name'] == "VQ_Transformer":
    raise ValueError("VQ_Transformer is deprecated, please use run_stage2.py instead.")
elif config['model_params']['name'] == "Im2Vec":
    data = DATASETMAP[config["data_params"]["dataset"]](**config["data_params"])
    data.setup()
    experiment = VAEXperiment(model, config['exp_params'])
else:
    raise ValueError("Unknown model provided: ", config['model_params']['name'])

# write config to file in save dir
os.makedirs(config['logging_params']['save_dir'], exist_ok=True)
with open(os.path.join(config['logging_params']['save_dir'], "config.yaml"), 'w+') as file:
    yaml.dump(config, file)

# profiler = SimpleProfiler(dirpath=os.path.join(config['logging_params']['save_dir']))
runner = Trainer(
    logger=wandb_logger,
    # strategy='ddp_find_unused_parameters_true',
    callbacks=[
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        #  LearningRateFinder(early_stop_threshold=None, num_training_steps=200),
        # EarlyStopping("val_loss", 0.001, 6 if config['model_params']['name'] != "Im2Vec" else 15, verbose=True),
        ModelCheckpoint(save_top_k=1,
                        dirpath=os.path.join(config['logging_params']['save_dir'], "checkpoints"),
                        monitor="val_loss",
                        save_last=True,
                        filename="e{epoch}-{val_loss:.4f}"),
    ],
    #  overfit_batches=20,
    # profiler=profiler,
    **config['trainer_params']
)


print(f"======= Training {config['model_params']['name']} =======")
try:
    # Start training
    if "continue_checkpoint" in config["exp_params"] and os.path.exists(config["exp_params"]["continue_checkpoint"]):
        runner.fit(experiment, datamodule=data, ckpt_path=config["exp_params"]["continue_checkpoint"])
    else:
        runner.fit(experiment, datamodule=data)
    # profiler.describe()
    # print(profiler.summary())
    # with open("profiler_results.txt", "w") as f:
        # f.write(profiler.summary())
except KeyboardInterrupt:
    # Handle the interrupt and save the profiling results
    print("Training interrupted by user.")
    # profiler.describe()
    # print(profiler.summary())
    # with open("profiler_results.txt", "w") as f:
    #     f.write(profiler.summary())

