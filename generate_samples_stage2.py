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
from utils import calculate_global_positions, shapes_to_drawing, drawing_to_tensor, map_wand_config
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

def load_model_from_basepath(basepath, device="cpu"):
    """
    returns model, ds, config
    """
    config = yaml.load(open(os.path.join(basepath, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    config["data_params"]["max_shapes_per_svg"] = 2000
    config["data_params"]["train_batch_size"] = 2
    config["data_params"]["val_batch_size"] = 2
    model = VSQ(**config["model_params"]).to(device)
    all_ckpts = glob(os.path.join(basepath, "checkpoints", "*.ckpt"))
    # sort by date
    latest_ckpt_path = sorted(all_ckpts, key=os.path.getmtime)[-1]
    state_dict = torch.load(latest_ckpt_path, map_location=device)["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except:
        model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
    ds = VSQDataset(**config["data_params"], train=False)
    model = model.eval()
    return model, ds, config

def load_stage2_model_from_basepath(vsq_model, basepath, device="cpu"):
    """
    returns model, ds, config
    """
    config = yaml.load(open(os.path.join(basepath, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    config = map_wand_config(config)
    config["data_params"]["fraction_of_class_only_inputs"] = 0.0
    config["data_params"]["fraction_of_blank_inputs"] = 0.0
    config["data_params"]["fraction_of_iconshop_chatgpt_inputs"] = 0.3
    config["data_params"]["fraction_of_full_description_inputs"] = 0.7

    tokenizer = VQTokenizer(vsq_model, 
                        config["data_params"].get("grid_size") or config["data_params"].get("width"), 
                        config['stage1_params']["num_codes_per_shape"], 
                        config["model_params"]["text_encoder_str"],
                        lseg = config["stage1_params"]["lseg"], 
                        device = device,
                        max_text_token_length=config["data_params"].get("max_text_token_length") or 50)

    model = VQ_SVG_Stage2(tokenizer, **config["model_params"], device=device)

    text_only_tokenizer = VQTokenizer(vsq_model, 
                                  config["data_params"].get("grid_size") or config["data_params"].get("width"), 
                                  config['stage1_params']["num_codes_per_shape"], 
                                  config["model_params"]["text_encoder_str"], 
                                  use_text_encoder_only=True, 
                                  lseg=config["stage1_params"]["lseg"],
                                  codebook_size=tokenizer.codebook_size,
                                  max_text_token_length=config["data_params"].get("max_text_token_length") or 50,)
    dm = VQDataModule(tokenizer=text_only_tokenizer,
                    **config["data_params"], 
                    context_length=config['model_params']['max_seq_len'],
                    train=False)
    dm.setup(return_ids=True)
    for ds in [dm.train_dataset, dm.val_dataset, dm.test_dataset]:
        ds.fraction_of_class_only_inputs = config["data_params"]["fraction_of_class_only_inputs"]
        ds.fraction_of_blank_inputs = config["data_params"]["fraction_of_blank_inputs"]
        ds.fraction_of_iconshop_chatgpt_inputs = config["data_params"]["fraction_of_iconshop_chatgpt_inputs"]
        ds.fraction_of_full_description_inputs = config["data_params"]["fraction_of_full_description_inputs"]
    
    all_ckpts = glob(os.path.join(basepath, "checkpoints", "*.ckpt"))
    # filter out last and instead take lowest eval loss
    all_ckpts = [x for x in all_ckpts if not "last.ckpt" in x]
    # sort by date
    latest_ckpt_path = sorted(all_ckpts, key=os.path.getmtime)[-1]
    state_dict = torch.load(latest_ckpt_path, map_location=device)["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except:
        model.load_state_dict({k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()})
    model = model.eval()
    return model, dm, config
import time

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--base-path',  '-c', dest="filename", metavar='FILE', help='path to the config file', default='results/TM_l')
parser.add_argument('--debug', action='store_true', help='reduces num_samples to 50')
parser.add_argument('--delay',type=int, dest="delay", help='time to sleep in seconds before execution', default=1)
parser.add_argument('--split',type=str, dest="split", help='which split', default=1)
parser.add_argument('--base-out',  '-o', dest="out", metavar='FILE', help='path to the output dir', default='results')

args = parser.parse_args()

# CONFIG EVERYTHING HERE
DEBUGGING = True if args.debug else False
STAGE2_BASE_PATH = args.filename
BASE_OUT_DIR = args.out
# STAGE2_BASE_PATH = "/scratch2/moritz_logs/thesis/Stage2_figr8/nseg=4_ncode=2_lseg=5"
NUM_SAMPLES = 30000 if not DEBUGGING else 50
VQ_CONTEXTS = [0]
BASE_PROMPT = "Black and white icon of {x}, vector graphic, outline"
# BASE_OUT_DIR = os.path.join(STAGE2_BASE_PATH, "validation")
# ---------------------
BATCH_SIZE = 16 if not DEBUGGING else 2
RENDER_WIDTH = 480
GLOBAL_STROKE_WIDTH = 0.7
SEED = 42
TEMPERATURE = 1.0
NUM_SVGS_TO_SAVE = NUM_SAMPLES if not DEBUGGING else 5
SAMPLING_METHOD = "top_p"
SAMPLING_KWARGS = {"thres":0.5}
MAX_DIST_FRAC = 3/72
FIXING_METHODS = ["min_dist_clip", "min_dist_interpolate"]
# FIXING_METHODS = []
# ---------------------

# save all of these settings above in a config file
settings = {
    "STAGE2_BASE_PATH": STAGE2_BASE_PATH,
    "NUM_SAMPLES": NUM_SAMPLES,
    "VQ_CONTEXTS": VQ_CONTEXTS,
    "BASE_PROMPT": BASE_PROMPT,
    "BATCH_SIZE": BATCH_SIZE,
    "RENDER_WIDTH": RENDER_WIDTH,
    "SEED": SEED,
    "TEMPERATURE": TEMPERATURE,
    "NUM_SVGS_TO_SAVE": NUM_SVGS_TO_SAVE,
    "SAMPLING_METHOD": SAMPLING_METHOD,
    "MAX_DIST_FRAC": MAX_DIST_FRAC,
    "FIXING_METHODS": FIXING_METHODS,
    "GLOBAL_STROKE_WIDTH" : GLOBAL_STROKE_WIDTH,
}

time.sleep(args.delay)
# for split in ["validation", "test"]:
for split in [args.split]:
    BASE_OUT_DIR = os.path.join(BASE_OUT_DIR, split+"_top_p_05")
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    with open(os.path.join(BASE_OUT_DIR, "config.yaml"), "w") as f:
        yaml.dump(settings, f)

    seed_everything(SEED)

    # load config to extract stage1 params
    config = yaml.load(open(os.path.join(STAGE2_BASE_PATH, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    config = map_wand_config(config)

    # load VSQ
    vsq_base_path = config["stage1_params"]["checkpoint_path"].split("checkpoints")[0]
    vsq_model = load_model_from_basepath(vsq_base_path, device=device)[0]

    # get model and data module
    stage_2_model, stage2_dm, stage2_config = load_stage2_model_from_basepath(vsq_model, STAGE2_BASE_PATH, device=device)
    stage2_dm.test_batch_size = BATCH_SIZE
    stage2_dm.val_batch_size = BATCH_SIZE

    # generation pipeline
    for vq_context in tqdm(VQ_CONTEXTS, leave=True):
        print(f"VQ Context: {vq_context}")
        if split == "validation":
            dl = stage2_dm.val_dataloader()
        else:
            dl = stage2_dm.test_dataloader()

        curr_out_dir = os.path.join(BASE_OUT_DIR, f"vq_context_{vq_context}")
        curr_svg_out_dir = os.path.join(curr_out_dir, "svgs")
        os.makedirs(os.path.join(curr_svg_out_dir), exist_ok=True)

        for fixing_method in FIXING_METHODS:
            os.makedirs(os.path.join(curr_svg_out_dir, fixing_method), exist_ok=True)

        num_already_generated = len(glob(os.path.join(curr_svg_out_dir, "*.svg")))
        print(f"Already generated {num_already_generated} samples.")
        print(f"Generating {split} set...")
        for i, [text_tokens, attention_mask, vq_tokens, _, _, all_ids] in enumerate(tqdm(dl, total=math.ceil(NUM_SAMPLES/BATCH_SIZE))):
            if num_already_generated >= NUM_SAMPLES:
                break
            # skipping mechanism
            if i * BATCH_SIZE <= num_already_generated:
                continue

            text_tokens = text_tokens.to(device)
            attention_mask = attention_mask.to(device)
            curr_vq_tokens = vq_tokens[:, :vq_context+1].clone().to(device)
            generation, reason = stage_2_model.generate(text_tokens, attention_mask, curr_vq_tokens, temperature=TEMPERATURE, sampling_method=SAMPLING_METHOD, sampling_kwargs=SAMPLING_KWARGS)
            captions = [stage_2_model.tokenizer.decode_text(tok) for tok in text_tokens]
            for j, g in enumerate(generation):
                idx = i * BATCH_SIZE + j
                drawing = stage_2_model.tokenizer._tokens_to_svg_drawing(g, global_stroke_width=GLOBAL_STROKE_WIDTH, post_process=False, num_strokes_to_paint=vq_context)
                drawing.saveas(os.path.join(curr_svg_out_dir, f"{all_ids[j]}.svg"))
                caption = captions[j]
                with open(os.path.join(curr_svg_out_dir, f"{all_ids[j]}.txt"), "w") as f:
                    f.write(caption)

                for fixing_method in FIXING_METHODS:
                    out_path = os.path.join(curr_svg_out_dir, fixing_method)
                    drawing_fixed = stage_2_model.tokenizer._tokens_to_svg_drawing(g, global_stroke_width=GLOBAL_STROKE_WIDTH, method=fixing_method, num_strokes_to_paint=vq_context, post_process=True, connect_last=False, max_dist_frac=MAX_DIST_FRAC)
                    drawing_fixed.saveas(os.path.join(out_path, f"{all_ids[j]}_{fixing_method}_fixed.svg"))
                    with open(os.path.join(out_path, f"{all_ids[j]}_{fixing_method}_fixed.txt"), "w") as f:
                        f.write(caption)
            num_already_generated += BATCH_SIZE


print("DONE.")
