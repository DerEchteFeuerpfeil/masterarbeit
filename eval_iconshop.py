import json
import os
from typing import List
import cairosvg
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import AutoProcessor, CLIPModel
from torch import nn
import random
import argparse
from glob import glob
from torchvision.utils import make_grid, save_image
import yaml

from tokenizer import VQTokenizer
from utils import map_wand_config, svg_to_tensor
from dataset import GenericRasterizedSVGDataset, VQDataModule

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ",device)
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

@torch.no_grad()
def compute_fid_score(generated_images, real_images, device, model_str:str = "openai/clip-vit-base-patch32", bs=32):
    # print(f"Computing FID with model {model_str} on device {device}")
    model = CLIPModel.from_pretrained(model_str)
    processor = AutoProcessor.from_pretrained(model_str)
    wrapper = CLIPWrapper(model, processor, device)
    fid = FrechetInceptionDistance(feature=wrapper, normalize=True)
    fid = fid.to(device)
    # print("Adding generated images...")
    for i in tqdm(range(0, len(generated_images), bs)):
        generated_images_batch = torch.stack(generated_images[i:i+bs]).to(device)
        fid.update(generated_images_batch, real=False)
    # print("Adding real images...")
    for i in tqdm(range(0, len(real_images), bs)):
        real_images_batch = torch.stack(real_images[i:i+bs]).to(device)
        fid.update(real_images_batch, real=True)

    return fid.compute()

@torch.no_grad()
def compute_clip_score(generated_images:List, captions:List, device, model_str:str = "openai/clip-vit-base-patch32",do_rescale=False, bs=32):
    # print(f"Computing CLIP score with model {model_str} on device {device}")
    metric = CLIPScore(model_name_or_path=model_str)
    metric = metric.to(device)
    for i in tqdm(range(0, len(generated_images), bs)):
        generated_images_batch = torch.stack(generated_images[i:i+bs]).to(device)
        captions_batch = captions[i:i+bs]
        metric.update(generated_images_batch, captions_batch,do_rescale=do_rescale)

    return metric.compute()

def load_stage2_dm(basepath):
    """
    to fully reproduce the same eval setup
    """
    config = yaml.load(open(os.path.join(basepath, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    config = map_wand_config(config)
    config["data_params"]["fraction_of_class_only_inputs"] = 0.0
    config["data_params"]["fraction_of_blank_inputs"] = 0.0
    config["data_params"]["fraction_of_iconshop_chatgpt_inputs"] = 0.3
    config["data_params"]["fraction_of_full_description_inputs"] = 0.7

    text_only_tokenizer = VQTokenizer(None, 
                                  config["data_params"]["width"], 
                                  config['stage1_params']["num_codes_per_shape"], 
                                  config["model_params"]["text_encoder_str"], 
                                  use_text_encoder_only=True, 
                                  lseg=config["stage1_params"]["lseg"],
                                  codebook_size=np.prod([int(x) for x in config["stage1_params"]["fsq_levels"]]),
                                  max_text_token_length=config["data_params"].get("max_text_token_length") or 50,)
    dm = VQDataModule(tokenizer=text_only_tokenizer,
                    **config["data_params"], 
                    context_length=config['model_params']['max_seq_len'],
                    train=False)
    dm.setup()

    return dm

parser = argparse.ArgumentParser(description='Eval SVG generations')
parser.add_argument('--svg_dir', help='directory of the SVGs', default='/scratch2/moritz_logs/thesis/IconShop_fixed/test/keyword')
parser.add_argument('--csv_path', help='path to split csv', default='/scratch2/moritz_data/figr8/stage2_split.csv')
parser.add_argument('--out_dir', help='path to split csv', default='/scratch2/moritz_logs/thesis/IconShop_fixed')
parser.add_argument('--debug', action='store_true', help='disable wandb logs, set workers to 0. (default false)')
args = parser.parse_args()


DEBUGGING = True if args.debug else False
USE_TEST = True # use test or eval split
SKIP_FID = False
NUM_SAMPLES = 10000 if not DEBUGGING else 50
BASE_PROMPT_FILLED = "Black and white icon of {x}, vector graphic"
BASE_PROMPT_OUTLINE = "Black and white icon of {x}, vector graphic, outline"
BASE_OUT_DIR = args.out_dir
STAGE_2_BASE_PATH = "results/TM_l"
# ---------------------
BATCH_SIZE = 4 if not DEBUGGING else 2
# CLIP_MODELS = ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14-336", "openai/clip-vit-large-patch14"]
CLIP_MODELS = ["openai/clip-vit-base-patch32"]
RENDER_WIDTH = 480
GLOBAL_STROKE_WIDTH = 0.7 / 72 * 200 # 200 is the viewbox of IconShop svg output, 72 was from my model
GLOBAL_STROKE_WIDTH_FRACTION = 0.7 / 72
NUM_REAL_IMAGES = 10000 if not DEBUGGING else 50
SEED = 42
seed_everything(SEED)

assert os.path.exists(args.svg_dir), f"SVG directory {args.svg_dir} does not exist"
assert os.path.exists(args.csv_path), f"CSV file {args.csv_path} does not exist"

dm = load_stage2_dm(STAGE_2_BASE_PATH)
dm.test_batch_size = BATCH_SIZE
dm.val_batch_size = BATCH_SIZE
if USE_TEST:
    df = dm.test_dataset.split
else:
    df = dm.val_dataset.split

# df = pd.read_csv(args.csv_path) -> not needed anymore, we take the same loading as in the stage2 eval to ensure we have the same data
df.index = df["id"]

if not os.path.exists(BASE_OUT_DIR):
    os.makedirs(BASE_OUT_DIR)

settings = {
    "NUM_SAMPLES": NUM_SAMPLES,
    "STAGE_2_BASE_PATH" : STAGE_2_BASE_PATH,
    "BASE_PROMPT_FILLED": BASE_PROMPT_FILLED,
    "BASE_PROMPT_OUTLINE": BASE_PROMPT_OUTLINE,
    "BATCH_SIZE": BATCH_SIZE,
    "CLIP_MODELS": CLIP_MODELS,
    "RENDER_WIDTH": RENDER_WIDTH,
    "NUM_REAL_IMAGES": NUM_REAL_IMAGES,
    "SEED": SEED,
    "GLOBAL_STROKE_WIDTH_FRACTION" : GLOBAL_STROKE_WIDTH_FRACTION,
    "GLOBAL_STROKE_WIDTH" : GLOBAL_STROKE_WIDTH,
    "BASE_OUT_DIR" : BASE_OUT_DIR
}

for CLIP_MODEL in CLIP_MODELS:
    # see if clip model already exists in results json and then skip
    if os.path.exists(os.path.join(BASE_OUT_DIR, "results.json")):
        with open(os.path.join(BASE_OUT_DIR, "results.json"), "r") as f:
            results_json = json.load(f)
            if f"unfixed_fid_{CLIP_MODEL}" in results_json:
                print(f"Skipping {CLIP_MODEL} since it already exists in results.json")
                continue
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL)
    clip_wrapper = CLIPWrapper(clip_model, clip_processor, device)
    if not SKIP_FID:
        fid_outline = FrechetInceptionDistance(feature=clip_wrapper, normalize=True)
        fid_filled = FrechetInceptionDistance(feature=clip_wrapper, normalize=True)
        fid_outline = fid_outline.to(device)
        fid_filled = fid_filled.to(device)
    clip_score = CLIPScore(model_name_or_path=CLIP_MODEL)
    clip_score = clip_score.to(device)

    all_svgs = glob(os.path.join(args.svg_dir, "*.svg"))
    all_svgs = [x for x in all_svgs if os.path.basename(x).split(".")[0] in df.index and os.path.exists(x)]
    r_idxs = random.sample(range(len(all_svgs)), min(NUM_SAMPLES, len(all_svgs)))
    all_svgs = [all_svgs[i] for i in r_idxs]
    all_ids = [os.path.basename(x).replace(".svg", "") for x in tqdm(all_svgs, desc="collecting ids")]


    print("Computing CLIP scores and FID from generated images...")
    all_clip_scores_outline = []
    all_clip_scores_filled = []
    total_skips = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(all_svgs), BATCH_SIZE), total=len(all_svgs)//BATCH_SIZE):
            try:
                batch_svgs = all_svgs[i:i+BATCH_SIZE]
                batch_ids = all_ids[i:i+BATCH_SIZE]
                batch_captions = [df.loc[x]["description"] for x in batch_ids]

                batch_raster_svgs_outline = [svg_to_tensor(x, new_stroke_width_fraction=GLOBAL_STROKE_WIDTH_FRACTION, output_width=RENDER_WIDTH, new_fill_color="none", new_stroke_color="black").to(device) for x in batch_svgs]
                batch_raster_svgs_filled = [svg_to_tensor(x, output_width=RENDER_WIDTH).to(device) for x in batch_svgs]

                batch_clip_score_outline = clip_score.forward(batch_raster_svgs_outline, [BASE_PROMPT_OUTLINE.format(x=c) for c in batch_captions]).cpu().item()
                batch_clip_score_filled = clip_score.forward(batch_raster_svgs_filled, [BASE_PROMPT_FILLED.format(x=c) for c in batch_captions]).cpu().item()
                if not SKIP_FID:
                    fid_outline.update(torch.stack(batch_raster_svgs_outline).to(device), real=False)
                    fid_filled.update(torch.stack(batch_raster_svgs_filled).to(device), real=False)

                all_clip_scores_outline.append(batch_clip_score_outline)
                all_clip_scores_filled.append(batch_clip_score_filled)
            except Exception as e:
                print(f"Error processing batch {i} - skipping")
                print(e)
                total_skips += 1

    if total_skips*BATCH_SIZE / len(all_svgs) > 0.1:
        raise Exception(f"Too many skips: {total_skips} - aborting")
    else:
        print(f"Skipped {total_skips} batches")

    if not SKIP_FID:
        print("Loading reference dataset...")
        rasterized_ds_outline = GenericRasterizedSVGDataset(args.csv_path,
                                    train=None if USE_TEST else None,
                                    fill=False,
                                    img_size=RENDER_WIDTH,
                                    global_stroke_width_fraction=GLOBAL_STROKE_WIDTH_FRACTION,
                                    subset=None)
        rasterized_ds_filled = GenericRasterizedSVGDataset(args.csv_path,
                                    train=None if USE_TEST else None,
                                    fill=True,
                                    img_size=RENDER_WIDTH,
                                    global_stroke_width=0.00001,
                                    subset=None,
                                    path_column="iconshop_filename")
        indices = random.sample(range(len(rasterized_ds_outline)), min(NUM_REAL_IMAGES, len(rasterized_ds_outline)))


        for i in tqdm(range(0, len(indices), BATCH_SIZE), total=len(indices) // BATCH_SIZE):
            curr_idxs = indices[i:i+BATCH_SIZE]
            real_imgs_outline = []
            for i in curr_idxs:
                real_imgs_outline.append(rasterized_ds_outline[i][0])
            fid_outline.update(torch.stack(real_imgs_outline).to(device), real=True)

            real_imgs_filled = []
            for i in curr_idxs:
                real_imgs_filled.append(rasterized_ds_filled[i][0])
            fid_filled.update(torch.stack(real_imgs_filled).to(device), real=True)


        print("Computing FID...")
        filled_fid_score = fid_filled.compute()
        unfilled_fid_score = fid_outline.compute()


        print(f"Filled FID: {filled_fid_score}")
        print(f"Outline FID: {unfilled_fid_score}")

        results_json = {
            "filled_fid": filled_fid_score.cpu().item(),
            "outline_fid": unfilled_fid_score.cpu().item()
        }

        with open(os.path.join(BASE_OUT_DIR, "results.json"), "w") as f:
            json.dump(results_json, f)
    else:
        results_json = {}

    prompt_adjusted_filled_clip_score = sum(all_clip_scores_filled) / len(all_clip_scores_filled)
    prompt_adjusted_outline_clip_score = sum(all_clip_scores_outline) / len(all_clip_scores_outline)

    results_json = {
        f"prompt_adjusted_filled_clip_score_{CLIP_MODEL}": get_item_if_tensor(prompt_adjusted_filled_clip_score),
        f"prompt_adjusted_outline_clip_score_{CLIP_MODEL}": get_item_if_tensor(prompt_adjusted_outline_clip_score),
        **results_json
    }

    if os.path.exists(os.path.join(BASE_OUT_DIR, "results.json")):
        with open(os.path.join(BASE_OUT_DIR, "results.json"), "r") as f:
            results_json_old = json.load(f)
            results_json = {**results_json_old, **results_json}

    with open(os.path.join(BASE_OUT_DIR, "results.json"), "w") as f:
        json.dump(results_json, f)

    print("DONE.")
    print(results_json)
