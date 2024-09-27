import gc
import json
import math
import os
from typing import List
import yaml
from models import VQ_SVG_Stage2, VSQ
from tokenizer import VQTokenizer
from experiment import SVG_VQVAE_Stage2_Experiment
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import AutoProcessor, CLIPModel
from dataset import GenericRasterizedSVGDataset, VSQDatamodule, VQDataModule, VSQDataset
from torch import nn
from math import ceil, sqrt
import time
import random
import argparse
from torchvision.utils import make_grid, save_image
torch.cuda.is_available()
from utils import calculate_global_positions, shapes_to_drawing, drawing_to_tensor, map_wand_config, svg_to_tensor
from svg_fixing import get_fixed_svg_drawing, get_fixed_svg_render, get_svg_render, min_dist_fix
import re 
from glob import glob
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_item_if_tensor(potential_tensor):
    if isinstance(potential_tensor, Tensor):
        return potential_tensor.cpu().detach().item()
    else:
        return potential_tensor

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CLIPWrapper(nn.Module):
    def __init__(self, model, processor, device):
        super().__init__()
        self.device = device
        self.processor = processor
        self.model = model.to(self.device)

    @torch.no_grad()
    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=False)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return self.model.get_image_features(**inputs)

parser = argparse.ArgumentParser(description='eval configs')
parser.add_argument('--path',  '-c', dest="path", metavar='FILE', help='path to the svgs', default='results/test_top_p_05/vq_context_0/svgs')
parser.add_argument('--debug', action='store_true', help='disable wandb logs, set workers to 0. (default false)')
parser.add_argument('--delay',type=int, dest="delay", help='time to sleep in seconds before execution', default=1)

args = parser.parse_args()

# CONFIG EVERYTHING HERE
DEBUGGING = True if args.debug else False
STAGE2_SVG_PATH = args.path
NUM_SAMPLES = 10000 if not DEBUGGING else 50
BASE_PROMPT = "Black and white icon of {x}, vector graphic, outline"
BASE_OUT_DIR = "results/test_top_p_05"
# ---------------------
BATCH_SIZE = 4 if not DEBUGGING else 2
# CLIP_MODELS = ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14-336", "openai/clip-vit-large-patch14"]
CLIP_MODELS = ["openai/clip-vit-base-patch32"]
RENDER_WIDTH = 480
GLOBAL_STROKE_WIDTH_FRACTION = 0.7 / 72
GLOBAL_STROKE_WIDTH = 0.7
NUM_REAL_IMAGES = 10000 if not DEBUGGING else 50
SEED = 42
# ---------------------

# save all of these settings above in a config file
settings = {
    "NUM_SAMPLES": NUM_SAMPLES,
    "STAGE2_SVG_PATH" : STAGE2_SVG_PATH,
    "BASE_PROMPT": BASE_PROMPT,
    "BATCH_SIZE": BATCH_SIZE,
    "CLIP_MODELS": CLIP_MODELS,
    "RENDER_WIDTH": RENDER_WIDTH,
    "NUM_REAL_IMAGES": NUM_REAL_IMAGES,
    "SEED": SEED,
    "GLOBAL_STROKE_WIDTH_FRACTION" : GLOBAL_STROKE_WIDTH_FRACTION,
    "GLOBAL_STROKE_WIDTH" : GLOBAL_STROKE_WIDTH,
    "BASE_OUT_DIR" : BASE_OUT_DIR
}


os.makedirs(BASE_OUT_DIR, exist_ok=True)

with open(os.path.join(BASE_OUT_DIR, "config.yaml"), "w") as f:
    yaml.dump(settings, f)

seed_everything(SEED)


for CLIP_MODEL in CLIP_MODELS:

    # see if clip model already exists in results json and then skip
    if os.path.exists(os.path.join(BASE_OUT_DIR, "results.json")):
        with open(os.path.join(BASE_OUT_DIR, "results.json"), "r") as f:
            results_json = json.load(f)
            if f"unfixed_fid_{CLIP_MODEL}" in results_json:
                print(f"Skipping {CLIP_MODEL} since it already exists in results.json")
                continue

    # print(f"Computing FID with model {model_str} on device {device}")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL)
    clip_wrapper = CLIPWrapper(clip_model, clip_processor, device)
    fid = FrechetInceptionDistance(feature=clip_wrapper, normalize=True)
    fid = fid.to(device)
    clip_score = CLIPScore(model_name_or_path=CLIP_MODEL)
    clip_score = clip_score.to(device)


    def read_caption_file(path):
        with open(path, "r") as f:
            return f.read()


    print("collecting paths")
    all_svgs = glob(os.path.join(STAGE2_SVG_PATH, "*.svg"))
    r_idxs = random.sample(range(len(all_svgs)), min(NUM_SAMPLES, len(all_svgs)))
    all_svgs = [all_svgs[i] for i in r_idxs]
    all_ids = [os.path.basename(x).replace(".svg", "") for x in tqdm(all_svgs, desc="collecting ids")]
    # captions = [read_caption_file(os.path.join(STAGE2_SVG_PATH, f"{x}.txt")) for x in tqdm(all_ids, desc="reading captions")]
    # raster_svgs = [svg_to_tensor(x, new_stroke_width_fraction=GLOBAL_STROKE_WIDTH_FRACTION, output_width=RENDER_WIDTH) for x in tqdm(all_svgs, desc="rasterizing svgs")]

    print("Computing CLIP scores and FID from generated images...")
    all_clip_scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_svgs), BATCH_SIZE), total=len(all_svgs)//BATCH_SIZE):
            batch_svgs = all_svgs[i:i+BATCH_SIZE]
            batch_ids = all_ids[i:i+BATCH_SIZE]
            batch_captions = [read_caption_file(os.path.join(STAGE2_SVG_PATH, f"{x}.txt")) for x in batch_ids]
            batch_raster_svgs = [svg_to_tensor(x, new_stroke_width_fraction=GLOBAL_STROKE_WIDTH_FRACTION, output_width=RENDER_WIDTH).to(device) for x in batch_svgs]
            batch_clip_score = clip_score.forward(batch_raster_svgs, [BASE_PROMPT.format(x=c) for c in batch_captions]).cpu().item()
            fid.update(torch.stack(batch_raster_svgs).to(device), real=False)
            all_clip_scores.append(batch_clip_score)



    print("Loading reference dataset...")
    rasterized_ds = GenericRasterizedSVGDataset(".data/stage2_split.csv",
                                train=None,
                                fill=False,
                                img_size=RENDER_WIDTH,
                                global_stroke_width=GLOBAL_STROKE_WIDTH,
                                subset=None)
    indices = random.sample(range(len(rasterized_ds)), min(NUM_REAL_IMAGES, len(rasterized_ds)))

    for i in tqdm(range(0, len(indices), BATCH_SIZE), total=len(indices) // BATCH_SIZE):
        curr_idxs = indices[i:i+BATCH_SIZE]
        real_imgs = []
        for i in curr_idxs:
            real_imgs.append(rasterized_ds[i][0])
        
        fid.update(torch.stack(real_imgs).to(device), real=True)
        
    print("Computing FID...")
    with torch.no_grad():
        unfixed_fid_score = fid.compute()


    print("Computing CLIP score...")
    prompt_adjusted_unfixed_clip_score = sum(all_clip_scores) / len(all_clip_scores)

    results_json = {
        f"unfixed_fid_{CLIP_MODEL}": unfixed_fid_score.cpu().item(),
        f"prompt_adjusted_unfixed_clip_{CLIP_MODEL}": get_item_if_tensor(prompt_adjusted_unfixed_clip_score),
    }

    if os.path.exists(os.path.join(BASE_OUT_DIR, "results.json")):
        with open(os.path.join(BASE_OUT_DIR, "results.json"), "r") as f:
            results_json_old = json.load(f)
            results_json = {**results_json_old, **results_json}
    
    with open(os.path.join(BASE_OUT_DIR, "results.json"), "w") as f:
        json.dump(results_json, f)


    print("DONE.")
    print(results_json)
