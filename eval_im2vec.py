from typing import List
import torch
from tqdm import tqdm
import yaml
from models.vector_vae_nlayers import VectorVAEnLayers
import os
import yaml
from dataset import GenericRasterizedSVGDataset
import torch
import random
import pydiffvg
from torchvision.utils import make_grid, save_image
torch.cuda.is_available()
from models import VectorVAEnLayers
import os
from typing import List
import yaml
import torch
import random
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import AutoProcessor, CLIPModel
from dataset import GenericRasterizedSVGDataset
from torch import nn
import random
from torchvision.utils import make_grid, save_image
torch.cuda.is_available()

def map_wand_config(config):
    new_config = {}
    for k, v in config.items():
        if not "wandb" in k:
            new_config[k] = v["value"]
    return new_config

def save_im2vec_points_to_svg(model:VectorVAEnLayers,
                            all_points:List, 
                            imsize, 
                            save_base_dir, 
                            filename):
        # z, log_var = model.encode(x)
        # all_points = model.decode(z)
        # print(all_points.std(dim=1))
        # all_points = ((all_points-0.5)*2 + 0.5)*self.imsize
        # if type(self.sort_idx) == type(None):
        #     angles = torch.atan(all_points[:,:,1]/all_points[:,:,0]).detach()
        #     self.sort_idx = torch.argsort(angles, dim=1)
        # Process the batch sequentially
        outputs = []
        shape_groups = []
        shapes = []
        for k in range(len(all_points)):
            # Get point parameters from network
            points = all_points[k].cpu()#[self.sort_idx[k]]
            if points.ndim > 2:
                points = points.squeeze(0)
            points = points * imsize
            color = torch.cat([torch.tensor([0,0,0,1]),])
            num_ctrl_pts = torch.zeros(model.curves, dtype=torch.int32) + 2

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                is_closed=True)

            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color)
            shape_groups.append(path_group)
        pydiffvg.save_svg(f"{save_base_dir}/{filename}",
                            imsize, imsize, shapes, shape_groups)

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
def compute_fid_score(generated_images, real_images, device, model_str:str = "openai/clip-vit-base-patch32"):
    print(f"Computing FID with model {model_str} on device {device}")
    model = CLIPModel.from_pretrained(model_str)
    processor = AutoProcessor.from_pretrained(model_str)
    wrapper = CLIPWrapper(model, processor, device)
    fid = FrechetInceptionDistance(feature=wrapper, normalize=True) # true is correct here
    fid = fid.to(device)
    bs = 32
    print("Adding generated images...")
    for i in tqdm(range(0, len(generated_images), bs)):
        generated_images_batch = torch.stack(generated_images[i:i+bs]).to(device)
        fid.update(generated_images_batch, real=False)
    print("Adding real images...")
    for i in tqdm(range(0, len(real_images), bs)):
        real_images_batch = torch.stack(real_images[i:i+bs]).to(device)
        fid.update(real_images_batch, real=True)

    return fid.compute()

@torch.no_grad()
def compute_clip_score(generated_images:List, captions:List, device, model_str:str = "openai/clip-vit-base-patch32",do_rescale=False):
    print(f"Computing CLIP score with model {model_str} on device {device}")
    metric = CLIPScore(model_name_or_path=model_str)
    metric = metric.to(device)
    bs = 32
    for i in tqdm(range(0, len(generated_images), bs)):
        generated_images_batch = torch.stack(generated_images[i:i+bs]).to(device)
        captions_batch = captions[i:i+bs]
        metric.update(generated_images_batch, captions_batch,do_rescale=do_rescale)

    return metric.compute()


im2vecsweep_base_config = {
    "base_path": "results/Im2Vec",
    "im2vec_model_path": "checkpoints/last-v1.ckpt",
    "im2vec_config_path": "config.yaml",
    "out_base_dir": "results/Im2Vec",
    "dataset": "icons",
}

for class_name in ["figr8_full"]:
    if class_name in ["home", "camera", "book"]:
        continue
    print("doing now class_name: ", class_name)
    selected_config = im2vecsweep_base_config
    im2vecsweep_base_config["class_name"] = class_name

    base_path = os.path.join(selected_config["base_path"], class_name)
    im2vec_model_path = os.path.join(base_path, selected_config["im2vec_model_path"])
    im2vec_config_path = os.path.join(base_path, selected_config["im2vec_config_path"])
    dataset = selected_config["dataset"]
    out_base_dir = os.path.join(selected_config["out_base_dir"], class_name)

    assert dataset in ["fonts", "icons"]
    if dataset == "icons":
        get_prompt_template = lambda x: f"Black and white icon of {x}, vector graphic"
    else:
        get_prompt_template = lambda x: ""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(im2vec_config_path, "r") as f:
        try:
            im2vec_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    im2vec_config = map_wand_config(im2vec_config)

    im2vec_config["model_params"]["imsize"] = 128
    im2vec_config["data_params"]["img_size"] = 128

    ds = GenericRasterizedSVGDataset(**im2vec_config["data_params"], train=None)
    im2vec = VectorVAEnLayers(**im2vec_config["model_params"])
    state_dict = torch.load(im2vec_model_path, map_location=device)["state_dict"]

    num_samples = min(3000, len(ds))

    for subdir in ["reconstructions", "samples", "gt"]:
        os.makedirs(os.path.join(out_base_dir, subdir), exist_ok=True)

    try:
        im2vec.load_state_dict(state_dict)
    except:
        im2vec.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
    im2vec = im2vec.eval().to(device)
    im2vec.base_control_features = im2vec.base_control_features.to(device)

    original_images=[]
    original_images_filled=[]
    reconstruction_points=[]

    # generate
    with torch.no_grad():
        random.seed(42)
        random_idx = random.sample(range(len(ds)), num_samples)
        samples_points = im2vec.multishape_sample(num_samples, return_points=True, device=device)
        for i,idx in tqdm(enumerate(random_idx), total=len(random_idx)):
            gt_image = ds[idx][0].to(device)
            reconstruction_points.append(im2vec.generate(gt_image.unsqueeze(0), return_points=True))
            
            filled_original = ds._rasterize_svg(ds.df.iloc[idx]["simplified_svg_file_path"], 480, fill=True)
            original_images_filled.append(filled_original)
            save_image(filled_original, os.path.join(out_base_dir,"gt",f"gt_filled_{idx}.png"))

            unfilled_original = ds._rasterize_svg(ds.df.iloc[idx]["simplified_svg_file_path"], 480, fill=False)
            original_images.append(unfilled_original)
            save_image(unfilled_original, os.path.join(out_base_dir,"gt",f"gt_unfilled_{idx}.png"))

            save_im2vec_points_to_svg(im2vec, samples_points[i], 72, os.path.join(out_base_dir,"samples"),f"im2vec_sample_{idx}.svg")
            save_im2vec_points_to_svg(im2vec, reconstruction_points[i], 72,os.path.join(out_base_dir,"reconstructions"),f"im2vec_reconstruction_{idx}.svg")

    # evaluate
    from utils import svg_file_path_to_tensor
    svg_sample_paths = [os.path.join(out_base_dir,"samples",f"im2vec_sample_{idx}.svg") for idx in random_idx]
    svg_reconstruction_paths = [os.path.join(out_base_dir,"reconstructions",f"im2vec_reconstruction_{idx}.svg") for idx in random_idx]

    print("rendering svgs")
    sample_renders_filled = [svg_file_path_to_tensor(p, stroke_width=0.4, image_size=480, filling=True) for p in svg_sample_paths]
    reconstruction_renders_filled = [svg_file_path_to_tensor(p, stroke_width=0.4, image_size=480, filling=True) for p in svg_reconstruction_paths]

    sample_renders_unfilled = [svg_file_path_to_tensor(p, stroke_width=0.4, image_size=480, filling=False) for p in svg_sample_paths]
    reconstruction_renders_unfilled = [svg_file_path_to_tensor(p, stroke_width=0.4, image_size=480, filling=False) for p in svg_reconstruction_paths]

    print("Computing MSE...")
    mse_filled = torch.nn.functional.mse_loss(torch.stack(reconstruction_renders_filled), torch.stack(original_images_filled))
    mse_unfilled = torch.nn.functional.mse_loss(torch.stack(reconstruction_renders_unfilled), torch.stack(original_images))

    print(f"mse_filled: {mse_filled}")
    print(f"mse_unfilled: {mse_unfilled}")

    save_image(make_grid(original_images_filled, nrow=25), os.path.join(out_base_dir,"original_images_filled.png"))
    save_image(make_grid(original_images, nrow=25), os.path.join(out_base_dir,"original_images_unfilled.png"))
    save_image(make_grid(sample_renders_filled, nrow=25), os.path.join(out_base_dir,"sample_renders_filled.png"))
    save_image(make_grid(reconstruction_renders_filled, nrow=25), os.path.join(out_base_dir,"reconstruction_renders_filled.png"))
    save_image(make_grid(sample_renders_unfilled, nrow=25), os.path.join(out_base_dir,"sample_renders_unfilled.png"))
    save_image(make_grid(reconstruction_renders_unfilled, nrow=25), os.path.join(out_base_dir,"reconstruction_renders_unfilled.png"))

    print("computing FID...")
    fid_samples_filled = compute_fid_score(sample_renders_filled, original_images_filled, device)
    fid_reconstructions_filled = compute_fid_score(reconstruction_renders_filled, original_images_filled, device)

    fid_samples_unfilled = compute_fid_score(sample_renders_unfilled, original_images, device)
    fid_reconstructions_unfilled = compute_fid_score(reconstruction_renders_unfilled, original_images, device)

    print(f"fid_samples_filled: {fid_samples_filled}")
    print(f"fid_reconstructions_filled: {fid_reconstructions_filled}")
    print(f"fid_samples_unfilled: {fid_samples_unfilled}")
    print(f"fid_reconstructions_unfilled: {fid_reconstructions_unfilled}")

    print("computing CLIP score...")
    if dataset=="icons":
        clip_samples_filled_prompt = -1 if class_name == "figr8_full" else compute_clip_score(sample_renders_filled, [get_prompt_template(class_name) for idx in random_idx], device, do_rescale=False)
        clip_reconstructions_filled_prompt = -1 if class_name == "figr8_full" else compute_clip_score(reconstruction_renders_filled, [get_prompt_template(class_name) for idx in random_idx], device, do_rescale=False)
        clip_samples_unfilled_prompt = -1 if class_name == "figr8_full" else compute_clip_score(sample_renders_unfilled, [get_prompt_template(class_name) for idx in random_idx], device, do_rescale=False)
        clip_reconstructions_unfilled_prompt = -1 if class_name == "figr8_full" else compute_clip_score(reconstruction_renders_unfilled, [get_prompt_template(class_name) for idx in random_idx], device, do_rescale=False)
    else:
        clip_samples_filled_prompt, clip_reconstructions_filled_prompt, clip_samples_unfilled_prompt, clip_reconstructions_unfilled_prompt = -1, -1, -1, -1
    clip_samples_filled_class = -1 if class_name == "figr8_full" else compute_clip_score(sample_renders_filled, [class_name for idx in random_idx], device, do_rescale=False)
    clip_reconstructions_filled_class = -1 if class_name == "figr8_full" else compute_clip_score(reconstruction_renders_filled, [class_name for idx in random_idx], device, do_rescale=False)
    clip_samples_unfilled_class = -1 if class_name == "figr8_full" else compute_clip_score(sample_renders_unfilled, [class_name for idx in random_idx], device, do_rescale=False)
    clip_reconstructions_unfilled_class = -1 if class_name == "figr8_full" else compute_clip_score(reconstruction_renders_unfilled, [class_name for idx in random_idx], device, do_rescale=False)

    # clip_white_image_baseline = compute_clip_score([torch.ones(3,480,480) for idx in random_idx], ["star" for idx in random_idx], device, do_rescale=False)
    # clip_black_image_baseline = compute_clip_score([torch.zeros(3,480,480) for idx in random_idx], ["star" for idx in random_idx], device, do_rescale=False)

    clip_white_image_baseline, clip_black_image_baseline = -1, -1

    print(f"clip_samples_filled_prompt: {clip_samples_filled_prompt}")
    print(f"clip_reconstructions_filled_prompt: {clip_reconstructions_filled_prompt}")
    print(f"clip_samples_unfilled_prompt: {clip_samples_unfilled_prompt}")
    print(f"clip_reconstructions_unfilled_prompt: {clip_reconstructions_unfilled_prompt}")

    print(f"clip_samples_filled_class: {clip_samples_filled_class}")
    print(f"clip_reconstructions_filled_class: {clip_reconstructions_filled_class}")
    print(f"clip_samples_unfilled_class: {clip_samples_unfilled_class}")
    print(f"clip_reconstructions_unfilled_class: {clip_reconstructions_unfilled_class}")

    print(f"clip_white_image_baseline: {clip_white_image_baseline}")
    print(f"clip_black_image_baseline: {clip_black_image_baseline}")

    with open(os.path.join(out_base_dir, "im2vec_results.txt"), "w") as f:
        f.write(f"num_samples: {len(random_idx)}\n")
        f.write(f"used dataset: {dataset}\n")
        f.write(f"class for clip: {class_name}\n")
        f.write(f"prompt template: {get_prompt_template('X')}\n\n")

        f.write(f"mse_recons_filled: \t{mse_filled}\n")
        f.write(f"mse_recons_unfilled: \t{mse_unfilled}\n")
        f.write(f"fid_samples_filled: \t{fid_samples_filled}\n")
        f.write(f"fid_samples_unfilled: \t{fid_samples_unfilled}\n")
        f.write(f"fid_reconstructions_filled: \t{fid_reconstructions_filled}\n")
        f.write(f"fid_reconstructions_unfilled: \t{fid_reconstructions_unfilled}\n")
        f.write(f"clip_samples_filled_prompt: \t{clip_samples_filled_prompt}\n")
        f.write(f"clip_samples_unfilled_prompt: \t{clip_samples_unfilled_prompt}\n")
        f.write(f"clip_reconstructions_filled_prompt: \t{clip_reconstructions_filled_prompt}\n")
        f.write(f"clip_reconstructions_unfilled_prompt: \t{clip_reconstructions_unfilled_prompt}\n")
        f.write(f"clip_samples_filled_class: \t{clip_samples_filled_class}\n")
        f.write(f"clip_samples_unfilled_class: \t{clip_samples_unfilled_class}\n")
        f.write(f"clip_reconstructions_filled_class: \t{clip_reconstructions_filled_class}\n")
        f.write(f"clip_reconstructions_unfilled_class: \t{clip_reconstructions_unfilled_class}\n\n")
        f.write(f"clip_white_image_baseline: \t{clip_white_image_baseline}\n")
        f.write(f"clip_black_image_baseline: \t{clip_black_image_baseline}\n")

    # also write config:
    with open(os.path.join(out_base_dir, "im2vec_config.yaml"), "w") as f:
        yaml.dump(im2vec_config, f)
    print("done")
